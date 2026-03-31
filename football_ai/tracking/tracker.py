import numpy as np
import supervision as sv

from football_ai.detection import Detector
from football_ai.identification import TeamDetector
from football_ai.reference_points import PnLCalibFieldProjector
from football_ai.tracking.byte_tracker import ByteTrack
from football_ai.tracking.tracker_logic_mixin import TrackerLogicMixin
from football_ai.tracking.possession import PossessionConfig, TeamPossessionEstimator


class Tracker(TrackerLogicMixin):
    @staticmethod
    def _normalize_detection_class_name(class_name):
        token = str(class_name or "").strip().lower()
        aliases = {
            "player": "player",
            "players": "player",
            "goalkeeper": "goalkeeper",
            "gk": "goalkeeper",
            "keeper": "goalkeeper",
            "referee": "referee",
            "ref": "referee",
            "refs": "referee",
            "ball": "ball",
            "balls": "ball",
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

    def __init__(self, model_path, detector_conf, team_detector_conf, bytetracker_conf,
                 ball_conf, tracker_conf, projector_conf, project_root,
    ):

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
        self.referee_recovery_max_lost_frames = tracker_conf["referee_recovery_max_lost_frames"]
        self.referee_recovery_max_distance = tracker_conf["referee_recovery_max_distance"]

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
        self.visualization_debug_frames = []
        self.possession_config = PossessionConfig.from_mapping(
            tracker_conf.get("possession")
        )
        self.possession_estimator = TeamPossessionEstimator(self.possession_config)

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

    def get_tracks(self, video, show_kmeans=False, frame_hook=None, collect_visual_debug=False):
        self.possession_estimator = TeamPossessionEstimator(self.possession_config)
        model_detections = self.model.detect(video)
        tracks = {"player": [], "goalkeeper": [], "referee": [], "ball": [], "possession": []}
        visual_debug_frames = [] if collect_visual_debug else None
        raw_to_canonical_id = {}
        canonical_to_raw_id = {}
        canonical_state = {}
        self._initialize_reserved_penalty_spot_players(canonical_state)
        ball_state = None
        for n_frame, detections in enumerate(model_detections):
            detections_sv = sv.Detections.from_ultralytics(detections)
            frame_size = None
            original_frame_bgr = getattr(detections, "orig_img", None)
            if original_frame_bgr is not None:
                frame_size = (
                    int(original_frame_bgr.shape[1]),
                    int(original_frame_bgr.shape[0]),
                )
            raw_detections = []
            accepted_raw_detection_indexes = set()
            bytetrack_raw_detection_indexes = set()
            bytetrack_discard_reason_by_raw_idx = {}
            bytetrack_id_by_raw_idx = {}
            if collect_visual_debug and detections.boxes is not None and len(detections.boxes) > 0:
                raw_xyxy = detections.boxes.xyxy.cpu().numpy()
                raw_conf = detections.boxes.conf.cpu().numpy()
                raw_cls = detections.boxes.cls.cpu().numpy().astype(int)
                for raw_idx, (bbox, score, cid) in enumerate(zip(raw_xyxy, raw_conf, raw_cls)):
                    class_name = self._normalize_detection_class_name(detections.names[cid])
                    raw_detections.append(
                        {
                            "raw_det_idx": int(raw_idx),
                            "class_name": class_name,
                            "bbox": [float(v) for v in bbox.tolist()],
                            "confidence": float(score),
                        }
                    )

            teams_of_detected_objects = self.team_detector.detect_teams(detections, show_kmeans)
            teams_labels = [dicc["team"] for dicc in teams_of_detected_objects]
            class_labels = [
                self._normalize_detection_class_name(dicc["class"])
                for dicc in teams_of_detected_objects
            ]
            field_projection = None
            field_positions = np.full((len(detections_sv), 2), np.nan, dtype=np.float32)
            ground_points_projected = np.full((len(detections_sv), 2), np.nan, dtype=np.float32)
            if self.field_projector is not None:
                if original_frame_bgr is not None:
                    field_projection = self.field_projector.project_detections(
                        original_frame_bgr,
                        detections_sv.xyxy,
                        class_names=class_labels,
                    )
                    field_positions = field_projection.field_positions_m
                    ground_points_projected = field_projection.ground_points_image_projected

            # Conserva metadatos por detección para recuperarlos tras filtrar por tracking.
            if detections_sv.data is None:
                detections_sv.data = {}
            detections_sv.data["team"] = np.array(
                [dicc["team"] for dicc in teams_of_detected_objects], dtype=object
            )
            detections_sv.data["distances"] = np.array(
                [dicc["distances"] for dicc in teams_of_detected_objects], dtype=object
            )
            detections_sv.data["shirt_color"] = np.array(
                [dicc["shirt_color"] for dicc in teams_of_detected_objects], dtype=object
            )
            detections_sv.data["bbox_size"] = np.array(
                [dicc["bbox_size"] for dicc in teams_of_detected_objects], dtype=float
            )
            detections_sv.data["field_position"] = np.asarray(
                field_positions, dtype=np.float32
            )
            detections_sv.data["ground_point_image"] = np.asarray(
                ground_points_projected, dtype=np.float32
            )
            detections_sv.data["raw_det_idx"] = np.arange(len(detections_sv), dtype=np.int32)

            tracks_detection = self.tracker.update_with_detections(
                detections_sv,
                teams_labels,
                class_labels,
            )
            
            for key in tracks.keys():
                tracks[key].append({})

            used_canonical_ids_in_frame = set()
            class_priority = {
                "referee": 0,
                "goalkeeper": 1,
                "player": 2,
                "ball": 3,
            }
            sorted_tracked_detections = sorted(
                list(tracks_detection),
                key=lambda item: (
                    class_priority.get(
                        str(item[5].get("class_name", "")),
                        99,
                    ),
                    -float(item[2]),
                ),
            )
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
            pending_detections = []
            ball_candidates = []

            def commit_assignment(
                raw_tracker_id,
                canonical_id,
                output_class_name,
                bbox,
                confidence,
                detected_team,
                field_position,
                metadata,
                raw_detection_idx=None,
            ):
                previous_owner_raw_id = canonical_to_raw_id.get(canonical_id)
                previous_canonical_id = raw_to_canonical_id.get(raw_tracker_id)
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
                canonical_state[canonical_id] = {
                    "bbox": bbox,
                    "class_name": output_class_name,
                    "last_frame": n_frame,
                    "team": resolved_team,
                    "field_position": resolved_field_position,
                    "movement_samples": movement_samples,
                    "mean_step_distance": mean_step_distance,
                    "step_per_frame_count": step_per_frame_count,
                    "step_per_frame_mean": step_per_frame_mean,
                    "step_per_frame_m2": step_per_frame_m2,
                    "reserved_seed": False,
                    "special_penalty_seed": special_penalty_seed,
                }
                used_canonical_ids_in_frame.add(canonical_id)

                tracks[output_class_name][n_frame][canonical_id] = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "team": resolved_team,
                    "distances": metadata.get("distances"),
                    "shirt_color": metadata.get("shirt_color"),
                    "bbox_size": metadata.get("bbox_size"),
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
                }
                if collect_visual_debug and raw_detection_idx is not None:
                    accepted_raw_detection_indexes.add(int(raw_detection_idx))

            for object_detected in sorted_tracked_detections:
                bbox, _, confidence, _, tracker_id, metadata = object_detected
                class_name = metadata["class_name"]
                if class_name not in tracks:
                    if collect_visual_debug:
                        raw_detection_idx = metadata.get("raw_det_idx")
                        if raw_detection_idx is not None:
                            try:
                                bytetrack_discard_reason_by_raw_idx[int(raw_detection_idx)] = {
                                    "reason_pre": "class_not_supported",
                                }
                            except (TypeError, ValueError):
                                pass
                    continue
                bbox = self._bbox_to_list(bbox)
                confidence = float(confidence)
                detected_team = metadata.get("team")
                field_position = metadata.get("field_position")
                raw_detection_idx = metadata.get("raw_det_idx")

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
                            }
                        canonical_id = None
                    elif canonical_id in used_canonical_ids_in_frame:
                        if collect_visual_debug and raw_detection_idx is not None:
                            bytetrack_discard_reason_by_raw_idx[int(raw_detection_idx)] = {
                                "reason_pre": "canonical_id_used_in_frame",
                            }
                        canonical_id = None
                    else:
                        output_class_name = previous_state["class_name"]
                        if output_class_name not in tracks:
                            if collect_visual_debug and raw_detection_idx is not None:
                                bytetrack_discard_reason_by_raw_idx[int(raw_detection_idx)] = {
                                    "reason_pre": "canonical_class_unsupported",
                                }
                            canonical_id = None
                        else:
                            resolved_class, resolved_reason = (
                                self._resolve_candidate_class_for_detection_debug(
                                    previous_state,
                                    class_name,
                                    detected_team,
                                    bbox,
                                    field_position,
                                    n_frame,
                                )
                            )
                            if resolved_class is None:
                                if collect_visual_debug and raw_detection_idx is not None:
                                    bytetrack_discard_reason_by_raw_idx[int(raw_detection_idx)] = {
                                        "reason_pre": str(resolved_reason or "canonical_gate_failed"),
                                    }
                            canonical_id = None

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
                        }
                    )
                    continue

                commit_assignment(
                    raw_tracker_id,
                    canonical_id,
                    output_class_name,
                    bbox,
                    confidence,
                    detected_team,
                    field_position,
                    metadata,
                    raw_detection_idx=raw_detection_idx,
                )

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

            for pending_idx, pending in enumerate(pending_detections):
                assignment = pending_assignments.get(pending_idx)
                if assignment is not None:
                    canonical_id, output_class_name = assignment
                else:
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
                    next_free_id = self._next_free_canonical_id(canonical_state)
                    if next_free_id is None:
                        if collect_visual_debug and pending.get("raw_detection_idx") is not None:
                            entry = bytetrack_discard_reason_by_raw_idx.setdefault(
                                int(pending["raw_detection_idx"]),
                                {},
                            )
                            entry["reason_post"] = "canonical_no_free_id"
                        continue
                    canonical_id = next_free_id

                commit_assignment(
                    pending["raw_tracker_id"],
                    canonical_id,
                    output_class_name,
                    pending["bbox"],
                    pending["confidence"],
                    pending["detected_team"],
                    pending.get("field_position"),
                    pending["metadata"],
                    raw_detection_idx=pending.get("raw_detection_idx"),
                )
                if collect_visual_debug and pending.get("raw_detection_idx") is not None:
                    # Mark as accepted if it came from YOLO index.
                    try:
                        bytetrack_discard_reason_by_raw_idx.pop(
                            int(pending["raw_detection_idx"]),
                            None,
                        )
                    except (TypeError, ValueError):
                        pass

            for canonical_id, state in canonical_state.items():
                if canonical_id in used_canonical_ids_in_frame:
                    continue
                if not state.get("reserved_seed", False):
                    continue
                tracks["player"][n_frame][canonical_id] = self._build_reserved_seed_track_payload(
                    state,
                    field_projection,
                )

            if detections.boxes is not None and len(detections.boxes) > 0:
                boxes = detections.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                for raw_idx, (bbox, score, cid) in enumerate(zip(xyxy, conf, cls)):
                    class_name = self._normalize_detection_class_name(detections.names[cid])
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
                                "ground_point_image": None,
                            },
                            "source": "raw",
                            "raw_det_idx": int(raw_idx),
                        }
                    )

            selected_ball = self._select_ball_candidate(
                ball_candidates,
                ball_state,
                n_frame,
                frame_size=frame_size,
            )
            if selected_ball is not None:
                tracks["ball"][n_frame][0] = self._build_ball_track_payload(
                    selected_ball["bbox"],
                    selected_ball["confidence"],
                    selected_ball["metadata"],
                )
                if collect_visual_debug and selected_ball.get("raw_det_idx") is not None:
                    accepted_raw_detection_indexes.add(int(selected_ball["raw_det_idx"]))
                ball_state = self._update_ball_state(
                    ball_state if self._ball_state_is_active(ball_state, n_frame) else None,
                    selected_ball["bbox"],
                    n_frame,
                )

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

            if collect_visual_debug:
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

            if callable(frame_hook):
                frame_hook(tracks, n_frame)
        self.visualization_debug_frames = visual_debug_frames if collect_visual_debug else []
        return tracks
