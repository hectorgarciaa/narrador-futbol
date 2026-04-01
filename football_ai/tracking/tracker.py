import numpy as np
import supervision as sv
from time import perf_counter

from football_ai.detection import Detector
from football_ai.identification import TeamDetector
from football_ai.reference_points import PnLCalibFieldProjector
from football_ai.tracking.byte_tracker import ByteTrack
from football_ai.tracking.tracker_logic_mixin import TrackerLogicMixin
from football_ai.tracking.possession import PossessionConfig, TeamPossessionEstimator


class Tracker(TrackerLogicMixin):
    @staticmethod
    def _measure_phase_execution(fn, *args, **kwargs):
        start = perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (perf_counter() - start) * 1000.0
        return result, elapsed_ms

    @staticmethod
    def _print_frame_phase_profile(phase_rows):
        total_ms = sum(float(row[1]) for row in phase_rows)
        for phase_index, row in enumerate(phase_rows, start=1):
            phase_title, phase_ms = row[0], row[1]
            print(f"fase {phase_index}: {phase_title}: {phase_ms:.2f} ms", flush=True)
            subphase_rows = row[2] if len(row) > 2 else None
            if subphase_rows:
                for subphase_title, subphase_ms in subphase_rows:
                    print(
                        f"  subfase: {subphase_title}: {subphase_ms:.2f} ms",
                        flush=True,
                    )
        print(f"FRAME: TIEMPO TOTAL: {total_ms:.2f} ms", flush=True)
        print("#########################################", flush=True)

    @staticmethod
    def _normalize_detection_class_name(class_name):
        token = str(class_name or "").strip().lower()
        aliases = {
            "player": "player", "players": "player",
            "goalkeeper": "player", "gk": "player", "keeper": "player",
            "referee": "referee", "ref": "referee", "refs": "referee",
            "ball": "ball", "balls": "ball",
        }
        return aliases.get(token, token)

    @staticmethod
    def _normalize_optional_positive_int(value):
        if value is None:
            return None
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return None
        if normalized <= 0:
            return None
        return normalized

    @classmethod
    def _detection_class_candidates_from_metadata(cls, metadata):
        if not isinstance(metadata, dict):
            return []
        candidates = []
        for key in ("class_tracker", "class", "class_yolo"):
            value = metadata.get(key)
            if not value or value in candidates:
                continue
            candidates.append(value)
        return candidates

    @classmethod
    def _primary_class_from_metadata(cls, metadata):
        candidates = cls._detection_class_candidates_from_metadata(metadata)
        if not candidates:
            return None, []
        return candidates[0], candidates

    def __init__(self, model_path, detector_conf, team_detector_conf, bytetracker_conf,
                 ball_conf, tracker_conf, projector_conf, project_root):

        self.model = Detector(model_path, **detector_conf)
        self.team_detector = TeamDetector(**team_detector_conf)
        # Legacy behavior (pre-refactor): do not cap track activation inside ByteTrack.
        # We still apply limits later in the canonical-ID layer (`max_tracks_per_class`).
        enforce_internal_class_limits = bool(
            (tracker_conf or {}).get("enforce_internal_class_limits", False)
        )
        bytetracker_runtime_conf = dict(bytetracker_conf or {})
        if not enforce_internal_class_limits:
            bytetracker_runtime_conf.pop("max_tracks_per_class", None)
        self.tracker = ByteTrack(**bytetracker_runtime_conf)
        self.field_projector = None
        if projector_conf["enabled"]:
            self.field_projector = PnLCalibFieldProjector(project_root=project_root, **projector_conf["constructor"])

        self.ball_min_conf = ball_conf["ball_min_conf"]
        self.ball_expected_position_gate_px = ball_conf["ball_expected_position_gate_px"]
        self.ball_expected_position_gate_growth_per_frame = ball_conf["ball_expected_position_gate_growth_per_frame"]
        self.ball_expected_position_confidence_relax = ball_conf["ball_expected_position_confidence_relax"]
        self.ball_size_ratio_per_frame = ball_conf["ball_size_ratio_per_frame"]
        self.ball_size_min_samples = ball_conf["ball_size_min_samples"]
        self.ball_size_std_factor = ball_conf["ball_size_std_factor"]
        self.ball_size_std_floor = ball_conf["ball_size_std_floor"]
        self.ball_max_reassign_lost_frames = ball_conf["ball_max_reassign_lost_frames"]
        self.ball_high_conf_override = ball_conf["ball_high_conf_override"]

        self.reassign_motion_factor = tracker_conf["reassign_motion_factor"]
        self.reassign_min_distance = tracker_conf["reassign_min_distance"]
        self.reassign_min_samples = tracker_conf["reassign_min_samples"]
        self.reassign_motion_growth_cap_frames = tracker_conf["reassign_motion_growth_cap_frames"]
        self.motion_std_gate_enabled = tracker_conf["motion_std_gate_enabled"]
        self.motion_std_factor = tracker_conf["motion_std_factor"]
        self.motion_std_min_samples = tracker_conf["motion_std_min_samples"]
        self.motion_std_floor = tracker_conf["motion_std_floor"]
        self.class_motion_stats = {
            "player": {"count": 0, "mean": 0.0, "m2": 0.0},
            "goalkeeper": {"count": 0, "mean": 0.0, "m2": 0.0},
            "referee": {"count": 0, "mean": 0.0, "m2": 0.0},
            "ball": {"count": 0, "mean": 0.0, "m2": 0.0},
        }

        self.max_tracks_per_class = bytetracker_conf["max_tracks_per_class"]
        # IDs canónicos reservados para entidades "persona" (sin balón).
        # El balón usa siempre track_id 0 en `tracks["ball"][frame][0]`.
        self.max_total_tracks = sum(
            int(limit)
            for class_name, limit in self.max_tracks_per_class.items()
            if str(class_name) != "ball"
        )
        self.reserve_penalty_spot_seed_players = tracker_conf["reserve_penalty_spot_seed_players"]
        self.reserve_penalty_spot_seed_match_distance_m = tracker_conf["reserve_penalty_spot_seed_match_distance_m"]
        configured_special_seed_ids = tracker_conf.get("special_seed_canonical_ids", [1, 2])
        self.special_seed_canonical_ids = tuple(
            sorted(
                {
                    int(canonical_id)
                    for canonical_id in configured_special_seed_ids
                    if self._normalize_optional_positive_int(canonical_id) is not None
                }
            )
        )
        if not self.special_seed_canonical_ids:
            self.special_seed_canonical_ids = (1, 2)
        self.referee_recovery_max_lost_frames = tracker_conf["referee_recovery_max_lost_frames"]
        self.referee_recovery_max_distance = tracker_conf["referee_recovery_max_distance"]
        configured_referee_ids = tracker_conf.get("referee_canonical_ids", [23, 24, 25])
        self.referee_canonical_ids = tuple(
            sorted(
                {
                    int(canonical_id)
                    for canonical_id in configured_referee_ids
                    if self._normalize_optional_positive_int(canonical_id) is not None
                }
            )
        )
        if not self.referee_canonical_ids:
            self.referee_canonical_ids = (23, 24, 25)
        self.referee_sideline_band_distance_m = float(
            tracker_conf.get("referee_sideline_band_distance_m", 3.0)
        )
        self.referee_field_length_m = float(
            projector_conf.get("constructor", {}).get("field_length_m", 106.0)
        )
        self.referee_field_width_m = float(
            projector_conf.get("constructor", {}).get("field_width_m", 68.0)
        )
        self.use_shirt_color_for_reassign = bool(
            tracker_conf.get("use_shirt_color_for_reassign", False)
        )
        self.shirt_color_reassign_distance_gate = float(
            tracker_conf.get("shirt_color_reassign_distance_gate", 45.0)
        )
        self.shirt_color_reassign_distance_growth_per_frame = float(
            tracker_conf.get("shirt_color_reassign_distance_growth_per_frame", 0.0)
        )
        raw_shirt_color_reassign_distance_cap = tracker_conf.get(
            "shirt_color_reassign_distance_cap",
            None,
        )
        if raw_shirt_color_reassign_distance_cap is None:
            self.shirt_color_reassign_distance_cap = None
        else:
            shirt_color_cap = float(raw_shirt_color_reassign_distance_cap)
            self.shirt_color_reassign_distance_cap = (
                shirt_color_cap if np.isfinite(shirt_color_cap) and shirt_color_cap > 0.0 else None
            )

        self.use_field_positions = projector_conf["enabled"]
        self.field_position_classes = projector_conf["classes"]
        self.reassign_min_field_distance_m = projector_conf["reassign_min_field_distance_m"]
        self.field_distance_gate_m = projector_conf["match_distance_gate_m"]
        self.field_distance_gate_max_lost_frames = projector_conf["match_distance_max_lost_frames"]
        
        raw_field_distance_gate_cap_m = projector_conf["match_distance_cap_m"]        
        if raw_field_distance_gate_cap_m is None:
            self.field_distance_gate_cap_m = None
        else:
            gate_cap = float(raw_field_distance_gate_cap_m)
            self.field_distance_gate_cap_m = (gate_cap if np.isfinite(gate_cap) and gate_cap > 0.0 else None)
        
        self.field_distance_growth_mode = str(projector_conf["match_distance_growth_mode"].strip().lower())
        if self.field_distance_growth_mode not in {"power", "linear_decay"}:
            self.field_distance_growth_mode = "power"
        self.field_distance_lost_exponent = projector_conf["match_distance_lost_exponent"]
        if (not np.isfinite(self.field_distance_lost_exponent) or self.field_distance_lost_exponent <= 0.0):
            self.field_distance_lost_exponent = 1.0
        self.field_distance_decay_per_frame = projector_conf["match_distance_decay_per_frame"]
        if (not np.isfinite(self.field_distance_decay_per_frame) or self.field_distance_decay_per_frame < 0.0):
            self.field_distance_decay_per_frame = 0.0
        
        self.strict_person_class_separation = tracker_conf["strict_person_class_separation"]
        self.require_field_position_for_reassign = tracker_conf["require_field_position_for_reassign"]
        self.max_reassign_lost_frames = tracker_conf["max_reassign_lost_frames"]
        raw_max_reassign_lost_frames_by_class = tracker_conf["max_reassign_lost_frames_by_class"]
        self.max_reassign_lost_frames_by_class = {}
        for class_name, class_limit in raw_max_reassign_lost_frames_by_class.items():
            self.max_reassign_lost_frames_by_class[str(class_name)] = class_limit
        self.forced_absorption_enabled = bool(
            tracker_conf.get("forced_absorption_enabled", True)
        )
        self.forced_absorption_player_min_lost_frames = max(
            1,
            int(tracker_conf.get("forced_absorption_player_min_lost_frames", 20)),
        )
        self.forced_absorption_player_min_consistent_frames = max(
            1,
            int(tracker_conf.get("forced_absorption_player_min_consistent_frames", 10)),
        )
        self.visualization_debug_frames = []
        self.possession_config = PossessionConfig.from_mapping(
            tracker_conf.get("possession")
        )
        self.possession_estimator = TeamPossessionEstimator(self.possession_config)
        self._current_referee_central_x_bounds = None

    @staticmethod
    def _normalize_track_identifier(track_id):
        if track_id is None:
            return None
        try:
            return int(track_id)
        except (TypeError, ValueError):
            return str(track_id)

    @classmethod
    def _track_id_matches(cls, left_track_id, right_track_id):
        if left_track_id is None or right_track_id is None:
            return False
        return (
            cls._normalize_track_identifier(left_track_id)
            == cls._normalize_track_identifier(right_track_id)
        )

    def _phase_prepare_frame_inputs(self, detections, collect_visual_debug):
        subphase_rows = []

        t0 = perf_counter()
        detections.names = {
            k: self._normalize_detection_class_name(v)
            for k, v in detections.names.items()
        }
        subphase_rows.append(("Normalizar nombres de clase", (perf_counter() - t0) * 1000.0))

        t0 = perf_counter()
        detections_sv = sv.Detections.from_ultralytics(detections)
        frame_size = (
            None
            if detections.orig_img is None
            else (int(detections.orig_img.shape[1]), int(detections.orig_img.shape[0]))
        )
        subphase_rows.append(("Convertir a supervision + frame_size", (perf_counter() - t0) * 1000.0))

        t0 = perf_counter()
        raw_detections, detection_class_labels = self._get_raw_detections(
            detections,
            collect_visual_debug,
        )
        subphase_rows.append(("Extraer detecciones RAW", (perf_counter() - t0) * 1000.0))

        t0 = perf_counter()
        field_projection, field_positions, ground_points_projected = (
            self._get_field_projection_and_positions(
                detections_sv,
                detection_class_labels,
                detections.orig_img,
            )
        )
        subphase_rows.append(("Proyección de campo (PnLCalib)", (perf_counter() - t0) * 1000.0))
        return {
            "payload": (
                detections_sv,
                frame_size,
                raw_detections,
                detection_class_labels,
                field_projection,
                field_positions,
                ground_points_projected,
            ),
            "subphase_rows": subphase_rows,
        }

    def _phase_assign_team_and_enrich_metadata(
        self,
        detections,
        detections_sv,
        raw_detections,
        detection_class_labels,
        show_kmeans,
        field_positions,
        ground_points_projected,
    ):
        subphase_rows = []

        t0 = perf_counter()
        teams_of_detected_objects = self.team_detector.detect_teams(detections, show_kmeans)
        subphase_rows.append(("TeamDetector.detect_teams", (perf_counter() - t0) * 1000.0))

        t0 = perf_counter()
        yolo_class_labels = np.array(
            detection_class_labels,
            dtype=object,
        )
        self.team_detector.apply_referee_relabel_gate(
            teams_of_detected_objects,
            yolo_class_labels,
            field_positions,
            self.referee_field_width_m,
            self.referee_sideline_band_distance_m,
        )
        current_class_labels_after_referee_gate = np.array(
            [dicc["class"] for dicc in teams_of_detected_objects],
            dtype=object,
        )
        self.team_detector.apply_goalkeeper_relabel_gate(
            teams_of_detected_objects,
            current_class_labels_after_referee_gate,
            field_positions,
            self.referee_field_width_m,
        )
        teams_labels = np.array(
            [dicc["team"] for dicc in teams_of_detected_objects],
            dtype=object,
        )
        class_labels = np.array(
            [dicc["class"] for dicc in teams_of_detected_objects],
            dtype=object,
        )
        if raw_detections:
            for raw_idx, raw_detection in enumerate(raw_detections):
                class_yolo_value = (
                    str(yolo_class_labels[raw_idx])
                    if raw_idx < len(yolo_class_labels)
                    else raw_detection.get("class_yolo")
                )
                class_relabel_value = (
                    str(class_labels[raw_idx])
                    if raw_idx < len(class_labels)
                    else None
                )
                raw_detection["class_yolo"] = class_yolo_value
                raw_detection["class_relabel"] = class_relabel_value
                raw_detection["class_team_detector"] = class_relabel_value
                if raw_idx < len(teams_of_detected_objects):
                    raw_detection["referee_reassign_gate"] = teams_of_detected_objects[raw_idx].get(
                        "referee_reassign_gate"
                    )
                    raw_detection["goalkeeper_reassign_gate"] = teams_of_detected_objects[raw_idx].get(
                        "goalkeeper_reassign_gate"
                    )
        subphase_rows.append(("Construir arrays de labels", (perf_counter() - t0) * 1000.0))

        t0 = perf_counter()
        self._enrich_metadata(
            detections_sv,
            teams_labels,
            class_labels,
            yolo_class_labels,
            teams_of_detected_objects,
            field_positions,
            ground_points_projected,
        )
        subphase_rows.append(("Enriquecer metadata de detecciones", (perf_counter() - t0) * 1000.0))
        return {
            "payload": (
                teams_of_detected_objects,
                teams_labels,
                class_labels,
                yolo_class_labels,
            ),
            "subphase_rows": subphase_rows,
        }

    def _phase_update_bytetrack(
        self,
        detections_sv,
        teams_labels,
        class_labels,
        yolo_class_labels,
    ):
        return self.tracker.update_with_detections(
            detections_sv,
            teams_labels,
            class_labels,
            yolo_class_labels=yolo_class_labels,
        )

    @staticmethod
    def _phase_initialize_empty_frame_tracks(tracks):
        for key in tracks.keys():
            tracks[key].append({})

    def _phase_process_tracked_detections(
        self,
        sorted_tracked_detections,
        canonical_state,
        raw_to_canonical_id,
        canonical_to_raw_id,
        used_canonical_ids_in_frame,
        tracks,
        accepted_raw_detection_indexes,
        collect_visual_debug,
        n_frame,
        pending_detections,
        ball_candidates,
        bytetrack_discard_reason_by_raw_idx,
    ):
        for object_detected in sorted_tracked_detections:
            bbox, _, confidence, _, tracker_id, metadata = object_detected
            bbox = self._bbox_to_list(bbox)
            confidence = float(confidence)
            class_name, detection_class_candidates = self._primary_class_from_metadata(
                metadata
            )
            if class_name is None:
                continue
            detected_team = metadata["team"]
            field_position = metadata["field_position"]
            raw_detection_idx = metadata["raw_det_idx"]
            detection_shirt_color = metadata.get("shirt_color")

            if class_name == "ball":
                ball_candidates.append(
                    {
                        "bbox": bbox,
                        "confidence": confidence,
                        "metadata": metadata,
                        "source": "tracked",
                        "raw_det_idx": raw_detection_idx,
                    }
                )
                continue

            raw_tracker_id = int(tracker_id)
            canonical_id = raw_to_canonical_id.get(raw_tracker_id)
            output_class_name = class_name
            if canonical_id is not None:
                previous_state = canonical_state.get(canonical_id)
                if previous_state is None:
                    if collect_visual_debug and raw_detection_idx is not None:
                        bytetrack_discard_reason_by_raw_idx[int(raw_detection_idx)] = {
                            "reason_pre": "canonical_state_missing",
                            "class_tracker": output_class_name,
                        }
                    canonical_id = None
                elif canonical_id in used_canonical_ids_in_frame:
                    if collect_visual_debug and raw_detection_idx is not None:
                        bytetrack_discard_reason_by_raw_idx[int(raw_detection_idx)] = {
                            "reason_pre": "canonical_id_used_in_frame",
                            "class_tracker": output_class_name,
                        }
                    canonical_id = None
                else:
                    output_class_name = previous_state["class_name"]
                    if output_class_name not in tracks:
                        if collect_visual_debug and raw_detection_idx is not None:
                            bytetrack_discard_reason_by_raw_idx[int(raw_detection_idx)] = {
                                "reason_pre": "canonical_class_unsupported",
                                "class_tracker": output_class_name,
                            }
                        canonical_id = None
                    else:
                        resolved_class, resolved_reason = (
                            self._resolve_raw_tracker_continuity_for_detection_debug(
                                previous_state,
                                class_name,
                                detected_team,
                                bbox,
                                field_position,
                                n_frame,
                                detection_class_candidates=detection_class_candidates,
                                detection_shirt_color=detection_shirt_color,
                            )
                        )
                        if resolved_class is None:
                            if collect_visual_debug and raw_detection_idx is not None:
                                bytetrack_discard_reason_by_raw_idx[int(raw_detection_idx)] = {
                                    "reason_pre": str(resolved_reason or "canonical_gate_failed"),
                                    "class_tracker": output_class_name,
                                }
                            canonical_id = None
                        else:
                            output_class_name = resolved_class
                            self._commit_assignment(
                                canonical_to_raw_id,
                                raw_to_canonical_id,
                                raw_tracker_id,
                                canonical_state,
                                n_frame,
                                used_canonical_ids_in_frame,
                                tracks,
                                accepted_raw_detection_indexes,
                                collect_visual_debug,
                                canonical_id,
                                output_class_name,
                                bbox,
                                confidence,
                                detected_team,
                                field_position,
                                metadata,
                                raw_detection_idx=raw_detection_idx,
                                detection_class_candidates=detection_class_candidates,
                            )
                            continue

            if class_name in {"player", "goalkeeper"}:
                (
                    special_override_id,
                    special_override_class_name,
                    special_override_distance_sq,
                ) = (
                    self._best_special_penalty_seed_id(
                        bbox,
                        [
                            candidate_id
                            for candidate_id in canonical_state.keys()
                            if candidate_id not in used_canonical_ids_in_frame
                        ],
                        canonical_state,
                        class_name,
                        detected_team,
                        n_frame,
                        field_position=field_position,
                        detection_class_candidates=detection_class_candidates,
                        detection_shirt_color=detection_shirt_color,
                    )
                )
                should_apply_special_override = (
                    special_override_id is not None
                    and special_override_class_name is not None
                )
                if (
                    should_apply_special_override
                    and canonical_id is not None
                    and not canonical_state.get(canonical_id, {}).get(
                        "special_penalty_seed",
                        False,
                    )
                ):
                    current_state = canonical_state.get(canonical_id)
                    current_lost_frames = max(
                        0,
                        n_frame - int(current_state.get("last_frame", n_frame)),
                    ) if isinstance(current_state, dict) else 0
                    current_distance_sq = self._bbox_distance_sq(
                        current_state.get("bbox") if isinstance(current_state, dict) else None,
                        bbox,
                        class_name=output_class_name,
                        field_position_a=(
                            current_state.get("field_position")
                            if isinstance(current_state, dict)
                            else None
                        ),
                        field_position_b=field_position,
                    )
                    should_apply_special_override = (
                        current_lost_frames > 1
                        and special_override_distance_sq is not None
                        and (
                            current_distance_sq is None
                            or special_override_distance_sq < current_distance_sq
                        )
                    )
                if should_apply_special_override:
                    canonical_id = special_override_id
                    output_class_name = special_override_class_name

            if canonical_id is None:
                output_class_name = class_name
                if collect_visual_debug and raw_detection_idx is not None:
                    entry = bytetrack_discard_reason_by_raw_idx.setdefault(
                        int(raw_detection_idx),
                        {},
                    )
                    entry.setdefault("reason_pre", "raw_tracker_id_unmapped_or_rejected")
                    entry.setdefault("class_tracker", output_class_name)
                pending_detections.append(
                    {
                        "raw_tracker_id": raw_tracker_id,
                        "bbox": bbox,
                        "confidence": confidence,
                        "detected_team": detected_team,
                        "field_position": field_position,
                        "metadata": metadata,
                        "preferred_class_name": output_class_name,
                        "raw_detection_idx": raw_detection_idx,
                        "detection_class_candidates": detection_class_candidates,
                        "shirt_color": detection_shirt_color,
                    }
                )
                continue

            self._commit_assignment(
                canonical_to_raw_id,
                raw_to_canonical_id,
                raw_tracker_id,
                canonical_state,
                n_frame,
                used_canonical_ids_in_frame,
                tracks,
                accepted_raw_detection_indexes,
                collect_visual_debug,
                canonical_id,
                output_class_name,
                bbox,
                confidence,
                detected_team,
                field_position,
                metadata,
                raw_detection_idx=raw_detection_idx,
                detection_class_candidates=detection_class_candidates,
            )

    def _phase_assign_pending_detections(
        self,
        pending_detections,
        canonical_state,
        used_canonical_ids_in_frame,
        n_frame,
        collect_visual_debug,
        bytetrack_discard_reason_by_raw_idx,
        canonical_to_raw_id,
        raw_to_canonical_id,
        tracks,
        accepted_raw_detection_indexes,
        forced_absorption_state,
    ):
        available_ids = [
            candidate_id
            for candidate_id in canonical_state.keys()
            if candidate_id not in used_canonical_ids_in_frame
        ]
        pending_assignments = self._assign_pending_by_lost_order(
            pending_detections,
            available_ids,
            canonical_state,
            n_frame,
        )
        forced_absorption_assignments, active_forced_absorption_raw_ids = (
            self._build_forced_absorption_assignments(
                pending_detections,
                canonical_state,
                used_canonical_ids_in_frame,
                n_frame,
                forced_absorption_state,
                excluded_pending_indexes=set(pending_assignments.keys()),
                excluded_canonical_ids={
                    canonical_id
                    for canonical_id, _output_class_name in pending_assignments.values()
                },
            )
        )

        for pending_idx, pending in enumerate(pending_detections):
            assignment = pending_assignments.get(pending_idx)
            if assignment is not None:
                canonical_id, output_class_name = assignment
            else:
                forced_assignment = forced_absorption_assignments.get(pending_idx)
                if forced_assignment is not None:
                    canonical_id, output_class_name, forced_absorption_info = (
                        forced_assignment
                    )
                    self._commit_assignment(
                        canonical_to_raw_id,
                        raw_to_canonical_id,
                        pending["raw_tracker_id"],
                        canonical_state,
                        n_frame,
                        used_canonical_ids_in_frame,
                        tracks,
                        accepted_raw_detection_indexes,
                        collect_visual_debug,
                        canonical_id,
                        output_class_name,
                        pending["bbox"],
                        pending["confidence"],
                        pending["detected_team"],
                        pending["field_position"],
                        pending["metadata"],
                        raw_detection_idx=pending.get("raw_detection_idx"),
                        detection_class_candidates=pending.get(
                            "detection_class_candidates"
                        ),
                        forced_absorption_info=forced_absorption_info,
                    )
                    forced_absorption_state.pop(
                        int(pending["raw_tracker_id"]),
                        None,
                    )
                    active_forced_absorption_raw_ids.discard(
                        int(pending["raw_tracker_id"])
                    )
                    if collect_visual_debug and pending.get("raw_detection_idx") is not None:
                        try:
                            bytetrack_discard_reason_by_raw_idx.pop(
                                int(pending["raw_detection_idx"]),
                                None,
                            )
                        except (TypeError, ValueError):
                            pass
                    continue
                output_class_name = pending["preferred_class_name"]
                class_limit = self.max_tracks_per_class.get(output_class_name)
                if class_limit is not None:
                    class_count = self._count_ids_for_class(
                        canonical_state,
                        output_class_name,
                        current_frame=n_frame,
                    )
                    if class_count >= int(class_limit):
                        if collect_visual_debug and pending.get("raw_detection_idx") is not None:
                            entry = bytetrack_discard_reason_by_raw_idx.setdefault(
                                int(pending["raw_detection_idx"]),
                                {},
                            )
                            entry["reason_post"] = "canonical_class_limit_reached"
                        continue
                next_free_id = self._next_free_canonical_id(
                    canonical_state,
                    class_name=output_class_name,
                )
                if next_free_id is None:
                    if collect_visual_debug and pending.get("raw_detection_idx") is not None:
                        entry = bytetrack_discard_reason_by_raw_idx.setdefault(
                            int(pending["raw_detection_idx"]),
                            {},
                        )
                        entry["reason_post"] = "canonical_no_free_id"
                    continue
                canonical_id = next_free_id

            self._commit_assignment(
                canonical_to_raw_id,
                raw_to_canonical_id,
                pending["raw_tracker_id"],
                canonical_state,
                n_frame,
                used_canonical_ids_in_frame,
                tracks,
                accepted_raw_detection_indexes,
                collect_visual_debug,
                canonical_id,
                output_class_name,
                pending["bbox"],
                pending["confidence"],
                pending["detected_team"],
                pending["field_position"],
                pending["metadata"],
                raw_detection_idx=pending.get("raw_detection_idx"),
                detection_class_candidates=pending.get("detection_class_candidates"),
            )
            forced_absorption_state.pop(int(pending["raw_tracker_id"]), None)
            if collect_visual_debug and pending.get("raw_detection_idx") is not None:
                # Mark as accepted if it came from YOLO index.
                try:
                    bytetrack_discard_reason_by_raw_idx.pop(
                        int(pending["raw_detection_idx"]),
                        None,
                    )
                except (TypeError, ValueError):
                    pass
        self._finalize_forced_absorption_state(
            forced_absorption_state,
            active_forced_absorption_raw_ids,
        )

    def _update_referee_absorption_context(self, sorted_tracked_detections):
        player_x_positions = []
        for object_detected in sorted_tracked_detections or []:
            try:
                _, _, _, _, _, metadata = object_detected
            except (TypeError, ValueError):
                continue
            class_name, _ = self._primary_class_from_metadata(metadata)
            if class_name not in {"player", "goalkeeper"}:
                continue
            field_position = self._field_position_to_tuple(
                metadata.get("field_position") if isinstance(metadata, dict) else None
            )
            if field_position is None:
                continue
            player_x_positions.append(float(field_position[0]))

        if len(player_x_positions) < 4:
            self._current_referee_central_x_bounds = None
            return

        player_x_positions.sort()
        self._current_referee_central_x_bounds = (
            player_x_positions[1],
            player_x_positions[-2],
        )

    @staticmethod
    def _phase_emit_reserved_seed_tracks(
        canonical_state,
        used_canonical_ids_in_frame,
        tracks,
        n_frame,
        field_projection,
        build_reserved_seed_track_payload_fn,
    ):
        for canonical_id, state in canonical_state.items():
            if canonical_id in used_canonical_ids_in_frame:
                continue
            if not state.get("reserved_seed", False):
                continue
            tracks["goalkeeper"][n_frame][canonical_id] = build_reserved_seed_track_payload_fn(
                state,
                field_projection,
            )

    def _phase_collect_raw_ball_candidates(
        self,
        detections,
        field_positions,
        ground_points_projected,
        ball_candidates,
    ):
        if detections.boxes is None or len(detections.boxes) <= 0:
            return
        boxes = detections.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for raw_idx, (bbox, score, cid) in enumerate(zip(xyxy, conf, cls)):
            class_name = detections.names[cid]
            if class_name != "ball" or score < self.ball_min_conf:
                continue
            x1, y1, x2, y2 = bbox.tolist()
            ball_candidates.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(score),
                    "metadata": {
                        "team": None,
                        "distances": None,
                        "shirt_color": None,
                        "bbox_size": float((x2 - x1) * (y2 - y1)),
                        "field_position": (
                            field_positions[raw_idx]
                            if raw_idx < len(field_positions)
                            else None
                        ),
                        "ground_point_image": (
                            ground_points_projected[raw_idx]
                            if raw_idx < len(ground_points_projected)
                            else None
                        ),
                    },
                    "source": "raw",
                    "raw_det_idx": int(raw_idx),
                }
            )

    def _phase_select_and_update_ball_track(
        self,
        ball_candidates,
        ball_state,
        n_frame,
        frame_size,
        tracks,
        collect_visual_debug,
        accepted_raw_detection_indexes,
    ):
        selected_ball = self._select_ball_candidate(
            ball_candidates,
            ball_state,
            n_frame,
            frame_size=frame_size,
        )
        if selected_ball is None:
            return ball_state
        tracks["ball"][n_frame][0] = self._build_ball_track_payload(
            selected_ball["bbox"],
            selected_ball["confidence"],
            selected_ball["metadata"],
        )
        if collect_visual_debug and selected_ball.get("raw_det_idx") is not None:
            accepted_raw_detection_indexes.add(int(selected_ball["raw_det_idx"]))
        return self._update_ball_state(
            ball_state if self._ball_state_is_active(ball_state, n_frame) else None,
            selected_ball["bbox"],
            n_frame,
        )

    def _phase_update_possession(self, tracks, n_frame):
        frame_tracks_for_possession = {
            "player": tracks["player"][n_frame],
            "goalkeeper": tracks["goalkeeper"][n_frame],
            "referee": tracks["referee"][n_frame],
            "ball": tracks["ball"][n_frame],
        }
        possession_info = self.possession_estimator.update_frame(
            n_frame,
            frame_tracks_for_possession,
        )
        self._attach_possession_metadata(tracks, n_frame, possession_info)

    @staticmethod
    def _phase_build_visual_debug_frame(
        collect_visual_debug,
        raw_detections,
        accepted_raw_detection_indexes,
        bytetrack_raw_detection_indexes,
        bytetrack_id_by_raw_idx,
        bytetrack_discard_reason_by_raw_idx,
        visual_debug_frames,
    ):
        if not collect_visual_debug:
            return
        discarded = [
            raw_detection
            for raw_detection in raw_detections
            if int(raw_detection["raw_det_idx"]) not in accepted_raw_detection_indexes
        ]
        discarded_not_tracked = []
        discarded_tracked_no_canonical = []
        for det in discarded:
            raw_idx = int(det.get("raw_det_idx"))
            if raw_idx in bytetrack_raw_detection_indexes:
                payload = dict(det)
                payload["bytetrack_id"] = bytetrack_id_by_raw_idx.get(raw_idx)
                reason_info = bytetrack_discard_reason_by_raw_idx.get(raw_idx, {})
                payload["class_tracker"] = reason_info.get("class_tracker")
                reason_pre = reason_info.get("reason_pre")
                reason_post = reason_info.get("reason_post")
                if reason_pre and reason_post:
                    payload["discard_reason"] = f"{reason_pre}|{reason_post}"
                elif reason_post:
                    payload["discard_reason"] = str(reason_post)
                elif reason_pre:
                    payload["discard_reason"] = str(reason_pre)
                discarded_tracked_no_canonical.append(payload)
            else:
                discarded_not_tracked.append(det)
        visual_debug_frames.append(
            {
                "raw_detections": raw_detections,
                "discarded_detections": discarded,
                "discarded_yolo_not_tracked": discarded_not_tracked,
                "discarded_bytetrack_not_canonical": discarded_tracked_no_canonical,
            }
        )

    @staticmethod
    def _phase_run_frame_hook(frame_hook, tracks, n_frame):
        subphase_rows = []
        t0 = perf_counter()
        is_callable = callable(frame_hook)
        subphase_rows.append(("Comprobar callable(frame_hook)", (perf_counter() - t0) * 1000.0))

        if is_callable:
            t0 = perf_counter()
            frame_hook(tracks, n_frame)
            subphase_rows.append(("Ejecutar frame_hook.on_frame", (perf_counter() - t0) * 1000.0))
        else:
            subphase_rows.append(("Ejecutar frame_hook.on_frame", 0.0))
        return subphase_rows

    def _attach_possession_metadata(self, tracks, frame_id, possession_info):
        owning_team_id = possession_info.get("team_id")
        owning_player_id = possession_info.get("player_id")
        possession_reason = possession_info.get("reason")
        possession_ball_detected = bool(possession_info.get("ball_detected", False))
        possession_nearest_track_id = possession_info.get("nearest_track_id")
        possession_nearest_team_id = possession_info.get("nearest_team_id")

        for class_name in ("player", "goalkeeper", "referee", "ball"):
            class_frames = tracks.get(class_name)
            if not isinstance(class_frames, list) or frame_id >= len(class_frames):
                continue
            frame_map = class_frames[frame_id]
            if not isinstance(frame_map, dict):
                continue
            for track_id, payload in frame_map.items():
                if not isinstance(payload, dict):
                    continue
                is_possession_player = (
                    class_name in {"player", "goalkeeper"}
                    and self._track_id_matches(track_id, owning_player_id)
                )
                payload["is_possession_player"] = bool(is_possession_player)
                payload["ball_owning_team_id"] = owning_team_id
                payload["ball_owning_player_id"] = owning_player_id
                payload["player_id"] = owning_player_id
                payload["possession_reason"] = possession_reason
                payload["possession_ball_detected"] = possession_ball_detected
                payload["possession_nearest_track_id"] = possession_nearest_track_id
                payload["possession_nearest_team_id"] = possession_nearest_team_id

        possession_frames = tracks.get("possession")
        if isinstance(possession_frames, list) and frame_id < len(possession_frames):
            possession_frames[frame_id] = {
                "team_id": owning_team_id,
                "player_id": owning_player_id,
                "reason": possession_reason,
                "ball_detected": possession_ball_detected,
                "nearest_track_id": possession_nearest_track_id,
                "nearest_team_id": possession_nearest_team_id,
            }

    def get_tracks(
        self,
        video,
        show_kmeans=False,
        frame_hook=None,
        collect_visual_debug=False,
        profile_phases=False,
    ):
        self.possession_estimator = TeamPossessionEstimator(self.possession_config)
        tracks = {"player": [], "goalkeeper": [], "referee": [], "ball": [], "possession": []}

        model_detections = self.model.detect(video)
        
        visual_debug_frames = [] if collect_visual_debug else None
        
        raw_to_canonical_id = {}
        canonical_to_raw_id = {}
        canonical_state = {}
        forced_absorption_state = {}
        self._initialize_reserved_penalty_spot_players(canonical_state)
        
        ball_state = None
        for n_frame, detections in enumerate(model_detections):
            frame_phase_rows = []

            (
                prepare_phase_result,
                elapsed_ms,
            ) = self._measure_phase_execution(
                self._phase_prepare_frame_inputs,
                detections,
                collect_visual_debug,
            )
            (
                (
                    detections_sv,
                    frame_size,
                    raw_detections,
                    detection_class_labels,
                    field_projection,
                    field_positions,
                    ground_points_projected,
                )
            ) = prepare_phase_result["payload"]
            frame_phase_rows.append((
                "Preparar detecciones y proyección",
                elapsed_ms,
                prepare_phase_result.get("subphase_rows"),
            ))

            (
                assign_phase_result,
                elapsed_ms,
            ) = self._measure_phase_execution(
                self._phase_assign_team_and_enrich_metadata,
                detections,
                detections_sv,
                raw_detections,
                detection_class_labels,
                show_kmeans,
                field_positions,
                ground_points_projected,
            )
            (
                (
                    teams_of_detected_objects,
                    teams_labels,
                    class_labels,
                    yolo_class_labels,
                )
            ) = assign_phase_result["payload"]
            frame_phase_rows.append((
                "Asignar equipo y enriquecer metadata",
                elapsed_ms,
                assign_phase_result.get("subphase_rows"),
            ))

            tracks_detection, elapsed_ms = self._measure_phase_execution(
                self._phase_update_bytetrack,
                detections_sv,
                teams_labels,
                class_labels,
                yolo_class_labels,
            )
            frame_phase_rows.append(("Asociación ByteTrack", elapsed_ms))

            _, elapsed_ms = self._measure_phase_execution(
                self._phase_initialize_empty_frame_tracks,
                tracks,
            )
            frame_phase_rows.append(("Inicializar contenedores de frame", elapsed_ms))

            sorted_tracked_detections = self._sort_tracked_detections(tracks_detection)
            self._update_referee_absorption_context(sorted_tracked_detections)

            bytetrack_raw_detection_indexes, bytetrack_id_by_raw_idx = self._collect_bytetrack_detection_info(sorted_tracked_detections, collect_visual_debug)

            used_canonical_ids_in_frame = set()

            bytetrack_discard_reason_by_raw_idx = {}
            accepted_raw_detection_indexes = set()

            pending_detections = []
            ball_candidates = []

            _, elapsed_ms = self._measure_phase_execution(
                self._phase_process_tracked_detections,
                sorted_tracked_detections,
                canonical_state,
                raw_to_canonical_id,
                canonical_to_raw_id,
                used_canonical_ids_in_frame,
                tracks,
                accepted_raw_detection_indexes,
                collect_visual_debug,
                n_frame,
                pending_detections,
                ball_candidates,
                bytetrack_discard_reason_by_raw_idx,
            )
            frame_phase_rows.append(("Canonización inicial desde tracks detectados", elapsed_ms))

            _, elapsed_ms = self._measure_phase_execution(
                self._phase_assign_pending_detections,
                pending_detections,
                canonical_state,
                used_canonical_ids_in_frame,
                n_frame,
                collect_visual_debug,
                bytetrack_discard_reason_by_raw_idx,
                canonical_to_raw_id,
                raw_to_canonical_id,
                tracks,
                accepted_raw_detection_indexes,
                forced_absorption_state,
            )
            frame_phase_rows.append(("Resolver pendientes y asignar IDs canónicos", elapsed_ms))

            _, elapsed_ms = self._measure_phase_execution(
                self._phase_emit_reserved_seed_tracks,
                canonical_state,
                used_canonical_ids_in_frame,
                tracks,
                n_frame,
                field_projection,
                self._build_reserved_seed_track_payload,
            )
            frame_phase_rows.append(("Emitir tracks de seeds reservadas", elapsed_ms))

            _, elapsed_ms = self._measure_phase_execution(
                self._phase_collect_raw_ball_candidates,
                detections,
                field_positions,
                ground_points_projected,
                ball_candidates,
            )
            frame_phase_rows.append(("Añadir candidatas de balón YOLO crudo", elapsed_ms))

            ball_state, elapsed_ms = self._measure_phase_execution(
                self._phase_select_and_update_ball_track,
                ball_candidates,
                ball_state,
                n_frame,
                frame_size,
                tracks,
                collect_visual_debug,
                accepted_raw_detection_indexes,
            )
            frame_phase_rows.append(("Seleccionar balón y actualizar estado", elapsed_ms))

            _, elapsed_ms = self._measure_phase_execution(
                self._phase_update_possession,
                tracks,
                n_frame,
            )
            frame_phase_rows.append(("Actualizar posesión del frame", elapsed_ms))

            _, elapsed_ms = self._measure_phase_execution(
                self._phase_build_visual_debug_frame,
                collect_visual_debug,
                raw_detections,
                accepted_raw_detection_indexes,
                bytetrack_raw_detection_indexes,
                bytetrack_id_by_raw_idx,
                bytetrack_discard_reason_by_raw_idx,
                visual_debug_frames,
            )
            frame_phase_rows.append(("Construir debug visual del frame", elapsed_ms))

            hook_subphase_rows, elapsed_ms = self._measure_phase_execution(
                self._phase_run_frame_hook,
                frame_hook,
                tracks,
                n_frame,
            )
            frame_phase_rows.append(("Ejecutar frame hook", elapsed_ms, hook_subphase_rows))

            if profile_phases:
                self._print_frame_phase_profile(frame_phase_rows)
        self.visualization_debug_frames = visual_debug_frames if collect_visual_debug else []
        return tracks
    
    def _get_raw_detections(self, detections, collect_visual_debug):
        raw_detections = []        
        if detections.boxes is not None and len(detections.boxes) > 0:
            raw_cls = detections.boxes.cls.cpu().numpy().astype(int)
            detection_class_labels = [detections.names[class_id] for class_id in raw_cls]
            if collect_visual_debug: 
                raw_xyxy = detections.boxes.xyxy.cpu().numpy()
                raw_conf = detections.boxes.conf.cpu().numpy()
                for raw_idx, (bbox, score, cid) in enumerate(zip(raw_xyxy, raw_conf, raw_cls)):
                    raw_detections.append(
                        {
                            "raw_det_idx": int(raw_idx),
                            "class_yolo": detections.names[cid],
                            "bbox": [float(v) for v in bbox.tolist()],
                            "confidence": float(score),
                        }
                    )
        return raw_detections, detection_class_labels
    
    def _get_field_projection_and_positions(self, detections_sv, detection_class_labels, orig_img):
        field_projection = None
        field_positions = np.full((len(detections_sv), 2), np.nan, dtype=np.float32)
        ground_points_projected = np.full((len(detections_sv), 2), np.nan, dtype=np.float32)
        if self.field_projector is not None and orig_img is not None:
                field_projection = self.field_projector.project_detections(
                    orig_img,
                    detections_sv.xyxy,
                    class_names=detection_class_labels,
                )
                field_positions = field_projection.field_positions_m
                ground_points_projected = field_projection.ground_points_image_projected
    
        return field_projection, field_positions, ground_points_projected
    
    def _enrich_metadata(
        self,
        detections_sv,
        teams_labels,
        class_labels,
        yolo_class_labels,
        teams_of_detected_objects,
        field_positions,
        ground_points_projected,
    ):
        # Conserva metadatos por detección para recuperarlos tras filtrar por tracking.
        if detections_sv.data is None:
            detections_sv.data = {}
        detections_sv.data["team"] = teams_labels
        detections_sv.data["class"] = class_labels
        detections_sv.data["class_yolo"] = yolo_class_labels
        detections_sv.data["distances"] = np.array([dicc["distances"] for dicc in teams_of_detected_objects], dtype=object)
        detections_sv.data["shirt_color"] = np.array([dicc["shirt_color"] for dicc in teams_of_detected_objects], dtype=object)
        detections_sv.data["bbox_size"] = np.array([dicc["bbox_size"] for dicc in teams_of_detected_objects], dtype=float)
        detections_sv.data["referee_reassign_gate"] = np.array(
            [dicc.get("referee_reassign_gate") for dicc in teams_of_detected_objects],
            dtype=object,
        )
        detections_sv.data["goalkeeper_reassign_gate"] = np.array(
            [dicc.get("goalkeeper_reassign_gate") for dicc in teams_of_detected_objects],
            dtype=object,
        )

        detections_sv.data["field_position"] = np.asarray(field_positions, dtype=np.float32)
        detections_sv.data["ground_point_image"] = np.asarray(ground_points_projected, dtype=np.float32)
        detections_sv.data["raw_det_idx"] = np.arange(len(detections_sv), dtype=np.int32)

    def _sort_tracked_detections(self, tracks_detection, class_priority={"referee": 0, "goalkeeper": 1, "player": 2, "ball": 3}):
        def _priority_from_metadata(metadata):
            class_name, _ = self._primary_class_from_metadata(metadata)
            return class_priority.get(str(class_name), max(class_priority.values()) + 1)

        return sorted(
            list(tracks_detection), 
            key=lambda item: (
                _priority_from_metadata(item[5]),
                -float(item[2])
            )
        )

    def _collect_bytetrack_detection_info(self, sorted_tracked_detections, collect_visual_debug):
        bytetrack_raw_detection_indexes = set()
        bytetrack_id_by_raw_idx = {}
        
        if collect_visual_debug:
            for object_detected in sorted_tracked_detections:
                _, _, _, _, tracker_id, metadata = object_detected
                raw_detection_idx = metadata.get("raw_det_idx")
                if raw_detection_idx is None:
                    continue
                try:
                    raw_idx_int = int(raw_detection_idx)
                    bytetrack_raw_detection_indexes.add(raw_idx_int)
                    bytetrack_id_by_raw_idx[raw_idx_int] = int(tracker_id)
                except (TypeError, ValueError):
                    continue
        return bytetrack_raw_detection_indexes, bytetrack_id_by_raw_idx

    def _commit_assignment(
        self,
        canonical_to_raw_id,
        raw_to_canonical_id,
        raw_tracker_id,
        canonical_state,
        n_frame,
        used_canonical_ids_in_frame,
        tracks,
        accepted_raw_detection_indexes,
        collect_visual_debug,
        canonical_id,
        output_class_name,
        bbox,
        confidence,
        detected_team,
        field_position,
        metadata,
        raw_detection_idx=None,
        detection_class_candidates=None,
        forced_absorption_info=None,
    ):
        previous_owner_raw_id = canonical_to_raw_id.get(canonical_id)
        previous_canonical_id = raw_to_canonical_id.get(raw_tracker_id)
        canonical_relinked = False
        canonical_assignment_mode = "new_canonical"
        if forced_absorption_info is not None:
            canonical_assignment_mode = "forced_absorption"
        elif previous_canonical_id is not None and previous_canonical_id != canonical_id:
            canonical_assignment_mode = "raw_tracker_reassigned_to_other_canonical"
            canonical_relinked = True
        elif previous_owner_raw_id is not None and previous_owner_raw_id != raw_tracker_id:
            canonical_assignment_mode = "canonical_relinked_to_new_raw_tracker"
            canonical_relinked = True
        elif canonical_id in canonical_state:
            canonical_assignment_mode = "raw_continuity"
        if (
            previous_canonical_id is not None
            and previous_canonical_id != canonical_id
        ):
            canonical_to_raw_id.pop(previous_canonical_id, None)
        if (
            previous_owner_raw_id is not None
            and previous_owner_raw_id != raw_tracker_id
        ):
            raw_to_canonical_id.pop(previous_owner_raw_id, None)

        raw_to_canonical_id[raw_tracker_id] = canonical_id
        canonical_to_raw_id[canonical_id] = raw_tracker_id
        previous_state = canonical_state.get(canonical_id, {})
        prev_samples = int(previous_state.get("movement_samples", 0))
        prev_mean = float(previous_state.get("mean_step_distance", 0.0))
        prev_last_frame = int(previous_state.get("last_frame", n_frame))
        frame_gap = max(1, n_frame - prev_last_frame)
        prev_step_pf_count = int(
            previous_state.get("step_per_frame_count", 0)
        )
        prev_step_pf_mean = float(
            previous_state.get("step_per_frame_mean", 0.0)
        )
        prev_step_pf_m2 = float(previous_state.get("step_per_frame_m2", 0.0))
        step_distance = self._step_distance(
            previous_state.get("bbox"),
            bbox,
            class_name=output_class_name,
            previous_field_position=previous_state.get("field_position"),
            new_field_position=field_position,
        )
        if step_distance is not None:
            movement_samples = prev_samples + 1
            if prev_samples <= 0:
                mean_step_distance = step_distance
            else:
                mean_step_distance = (
                    (prev_mean * prev_samples) + step_distance
                ) / movement_samples
        else:
            movement_samples = prev_samples
            mean_step_distance = prev_mean

        if step_distance is not None:
            step_per_frame = step_distance / frame_gap
            (
                step_per_frame_count,
                step_per_frame_mean,
                step_per_frame_m2,
            ) = self._update_running_stats(
                prev_step_pf_count,
                prev_step_pf_mean,
                prev_step_pf_m2,
                step_per_frame,
            )
            class_stats = self.class_motion_stats.setdefault(
                output_class_name,
                {"count": 0, "mean": 0.0, "m2": 0.0},
            )
            (
                class_stats["count"],
                class_stats["mean"],
                class_stats["m2"],
            ) = self._update_running_stats(
                class_stats["count"],
                class_stats["mean"],
                class_stats["m2"],
                step_per_frame,
            )
        else:
            step_per_frame_count = prev_step_pf_count
            step_per_frame_mean = prev_step_pf_mean
            step_per_frame_m2 = prev_step_pf_m2
        special_penalty_seed = bool(previous_state.get("special_penalty_seed", False))
        resolved_team = (
            detected_team
            if detected_team is not None
            else previous_state.get("team")
        )
        if special_penalty_seed:
            resolved_team = None
        resolved_field_position = (
            self._field_position_to_tuple(field_position)
            or previous_state.get("field_position")
        )
        referee_role_zone = previous_state.get("referee_role_zone")
        if output_class_name == "referee":
            referee_role_zone = (
                self._referee_zone_from_field_position(resolved_field_position)
                or referee_role_zone
            )
        else:
            referee_role_zone = None
        canonical_state[canonical_id] = {
            "bbox": bbox,
            "class_name": output_class_name,
            "last_frame": n_frame,
            "team": resolved_team,
            "shirt_color": self._shirt_color_to_tuple(metadata.get("shirt_color")),
            "field_position": resolved_field_position,
            "movement_samples": movement_samples,
            "mean_step_distance": mean_step_distance,
            "step_per_frame_count": step_per_frame_count,
            "step_per_frame_mean": step_per_frame_mean,
            "step_per_frame_m2": step_per_frame_m2,
            "reserved_seed": False,
            "special_penalty_seed": special_penalty_seed,
            "class_candidates": list(detection_class_candidates or []),
            "referee_role_zone": referee_role_zone,
            "last_raw_tracker_id": int(raw_tracker_id),
            "canonical_assignment_mode": str(canonical_assignment_mode),
            "canonical_relinked": bool(canonical_relinked),
        }
        used_canonical_ids_in_frame.add(canonical_id)

        tracks[output_class_name][n_frame][canonical_id] = {
            "bbox": bbox,
            "confidence": confidence,
            "team": resolved_team,
            "distances": metadata.get("distances"),
            "shirt_color": metadata.get("shirt_color"),
            "class_tracker": output_class_name,
            "class_relabel": metadata.get("class"),
            "class_yolo": metadata.get("class_yolo"),
            "referee_reassign_gate": metadata.get("referee_reassign_gate"),
            "goalkeeper_reassign_gate": metadata.get("goalkeeper_reassign_gate"),
            "bbox_size": metadata.get("bbox_size"),
            "source_raw_tracker_id": int(raw_tracker_id),
            "canonical_assignment_mode": str(canonical_assignment_mode),
            "canonical_relinked": bool(canonical_relinked),
            "field_position_m": (
                list(self._field_position_to_tuple(field_position))
                if self._field_position_to_tuple(field_position) is not None
                else None
            ),
            "ground_point_image": (
                metadata.get("ground_point_image").tolist()
                if hasattr(metadata.get("ground_point_image"), "tolist")
                else metadata.get("ground_point_image")
            ),
            "reserved_seed": False,
            "special_penalty_seed": special_penalty_seed,
            "forced_absorption": bool(forced_absorption_info),
            "forced_absorption_raw_tracker_streak_frames": (
                int(forced_absorption_info["raw_tracker_streak_frames"])
                if forced_absorption_info is not None
                else None
            ),
            "forced_absorption_canonical_lost_frames": (
                int(forced_absorption_info["canonical_lost_frames"])
                if forced_absorption_info is not None
                else None
            ),
            "forced_absorption_source_raw_tracker_id": (
                int(forced_absorption_info["raw_tracker_id"])
                if forced_absorption_info is not None
                else None
            ),
            "forced_absorption_mode": (
                str(forced_absorption_info.get("mode"))
                if forced_absorption_info is not None
                else None
            ),
            "forced_absorption_reference_frame": (
                int(forced_absorption_info["reference_frame"])
                if forced_absorption_info is not None
                and forced_absorption_info.get("reference_frame") is not None
                else None
            ),
            "forced_absorption_distance_sq": (
                float(forced_absorption_info["distance_sq"])
                if forced_absorption_info is not None
                and forced_absorption_info.get("distance_sq") is not None
                else None
            ),
        }

        if collect_visual_debug and raw_detection_idx is not None:
            accepted_raw_detection_indexes.add(int(raw_detection_idx))
