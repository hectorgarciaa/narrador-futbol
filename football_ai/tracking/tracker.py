import numpy as np
import supervision as sv

from football_ai.detection import Detector
from football_ai.identification import TeamDetector
from football_ai.reference_points import PnLCalibFieldProjector
from football_ai.tracking.byte_tracker import ByteTrack

class Tracker:
    def __init__(
        self,
        model_path,
        conf,
        tracker_conf,
        team_colors,
        ball_min_conf=0.01,
        max_tracks_per_class=None,
        field_tracking_conf=None,
        project_root=None,
    ):
        if max_tracks_per_class is None:
            max_tracks_per_class = {"player": 22, "ball": 1, "referee": 3}
        if field_tracking_conf is None:
            field_tracking_conf = {}

        self.model = Detector(model_path, conf)
        # Compatibilidad: nueva convención snake_case y alias legacy camelCase.
        self.team_detector = TeamDetector(team_colors)
        self.teamDetector = self.team_detector
        self.max_tracks_per_class = max_tracks_per_class
        self.max_total_tracks = int(
            tracker_conf.get("max_total_tracks", sum(max_tracks_per_class.values()))
        )
        self.enforce_internal_class_limits = bool(
            tracker_conf.get("enforce_internal_class_limits", False)
        )
        internal_class_limits = (
            max_tracks_per_class if self.enforce_internal_class_limits else None
        )
        self.tracker = ByteTrack(
            tracker_conf["track_thresh"],
            tracker_conf["track_buffer"],
            tracker_conf["match_thresh"],
            tracker_conf["frame_rate"],
            tracker_conf["minimum_consecutive_frames"],
            max_tracks_per_class=internal_class_limits,
            team_mismatch_penalty=tracker_conf.get("team_mismatch_penalty", 1000.0),
            second_match_threshold=tracker_conf.get("second_match_threshold", 0.7),
            unconfirmed_match_threshold=tracker_conf.get("unconfirmed_match_threshold", 0.8),
            use_field_positions=field_tracking_conf.get("enabled", False),
            field_position_classes=field_tracking_conf.get(
                "classes", ["player", "goalkeeper"]
            ),
            field_distance_gate_m=field_tracking_conf.get(
                "match_distance_gate_m", 8.0
            ),
            field_distance_weight=field_tracking_conf.get(
                "match_distance_weight", 0.25
            ),
        )
        self.ball_min_conf = ball_min_conf
        self.reassign_motion_factor = float(
            tracker_conf.get("reassign_motion_factor", 4.0)
        )
        self.reassign_min_distance = float(
            tracker_conf.get("reassign_min_distance", 25.0)
        )
        self.reassign_min_samples = int(
            tracker_conf.get("reassign_min_samples", 3)
        )
        self.referee_recovery_max_lost_frames = int(
            tracker_conf.get("referee_recovery_max_lost_frames", 3)
        )
        self.referee_recovery_max_distance = float(
            tracker_conf.get("referee_recovery_max_distance", 45.0)
        )
        self.use_field_positions = bool(field_tracking_conf.get("enabled", False))
        self.field_position_classes = frozenset(
            field_tracking_conf.get("classes", ["player", "goalkeeper"])
        )
        self.reassign_min_field_distance_m = float(
            field_tracking_conf.get("reassign_min_field_distance_m", 4.0)
        )
        self.field_projector = None
        if self.use_field_positions:
            method = field_tracking_conf.get("method", "pnlcalib")
            if method != "pnlcalib":
                raise ValueError(
                    f"Método de proyección de campo no soportado: {method}"
                )
            self.field_projector = PnLCalibFieldProjector(
                project_root=project_root,
                field_length_m=float(field_tracking_conf.get("field_length_m", 106.0)),
                field_width_m=float(field_tracking_conf.get("field_width_m", 68.0)),
                max_width=int(field_tracking_conf.get("max_width", 1280)),
                bottom_offset_ratio=float(
                    field_tracking_conf.get("bottom_offset_ratio", 0.04)
                ),
                keypoint_threshold=float(
                    field_tracking_conf.get("keypoint_threshold", 0.3434)
                ),
                line_threshold=float(
                    field_tracking_conf.get("line_threshold", 0.7867)
                ),
                pnl_refine=bool(field_tracking_conf.get("pnl_refine", True)),
                temporal_blend=float(field_tracking_conf.get("temporal_blend", 0.20)),
                pixels_per_meter=int(field_tracking_conf.get("pixels_per_meter", 8)),
                device=field_tracking_conf.get("device"),
            )

    @staticmethod
    def _bbox_to_list(bbox):
        if bbox is None:
            return None
        if hasattr(bbox, "tolist"):
            return bbox.tolist()
        return list(bbox)

    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @staticmethod
    def _field_position_to_tuple(field_position):
        if field_position is None:
            return None
        field_position = np.asarray(field_position, dtype=np.float32).reshape(-1)
        if field_position.size < 2 or not np.all(np.isfinite(field_position[:2])):
            return None
        return (float(field_position[0]), float(field_position[1]))

    def _use_field_position_for_class(self, class_name, field_position=None):
        if not self.use_field_positions:
            return False
        if class_name not in self.field_position_classes:
            return False
        return self._field_position_to_tuple(field_position) is not None

    def _next_free_canonical_id(self, canonical_state):
        for canonical_id in range(1, self.max_total_tracks + 1):
            if canonical_id not in canonical_state:
                return canonical_id
        return None

    @staticmethod
    def _is_compatible_class(previous_class, new_class):
        person_classes = {"player", "goalkeeper"}
        if previous_class in person_classes and new_class in person_classes:
            return True
        return previous_class == new_class

    @staticmethod
    def _is_team_compatible(previous_team, new_team, class_name):
        # Solo forzamos consistencia de equipo para clases de jugadores.
        if class_name not in {"player", "goalkeeper"}:
            return True
        if previous_team is None or new_team is None:
            return True
        return previous_team == new_team

    def _step_distance(
        self,
        previous_bbox,
        new_bbox,
        class_name=None,
        previous_field_position=None,
        new_field_position=None,
    ):
        if self._use_field_position_for_class(
            class_name,
            previous_field_position,
        ) and self._use_field_position_for_class(class_name, new_field_position):
            px, py = self._field_position_to_tuple(previous_field_position)
            nx, ny = self._field_position_to_tuple(new_field_position)
            return float(((nx - px) ** 2 + (ny - py) ** 2) ** 0.5)
        if previous_bbox is None or new_bbox is None:
            return None
        px, py = self._bbox_center(previous_bbox)
        nx, ny = self._bbox_center(new_bbox)
        return float(((nx - px) ** 2 + (ny - py) ** 2) ** 0.5)

    def _is_motion_compatible(
        self,
        previous_state,
        new_bbox,
        current_frame,
        class_name=None,
        new_field_position=None,
    ):
        effective_class = class_name or previous_state.get("class_name")
        previous_bbox = previous_state.get("bbox")
        previous_field_position = previous_state.get("field_position")
        step_distance = self._step_distance(
            previous_bbox,
            new_bbox,
            class_name=effective_class,
            previous_field_position=previous_field_position,
            new_field_position=new_field_position,
        )
        if step_distance is None:
            return True

        lost_frames = max(
            1,
            current_frame - int(previous_state.get("last_frame", current_frame)),
        )
        samples = int(previous_state.get("movement_samples", 0))
        mean_step_distance = float(previous_state.get("mean_step_distance", 0.0))

        if self._use_field_position_for_class(effective_class, previous_field_position):
            max_allowed_jump = self.reassign_min_field_distance_m
        else:
            max_allowed_jump = self.reassign_min_distance
        if samples >= self.reassign_min_samples:
            expected_jump = mean_step_distance * lost_frames
            max_allowed_jump = max(
                max_allowed_jump,
                expected_jump * self.reassign_motion_factor,
            )

        return step_distance <= max_allowed_jump

    def _bbox_distance_sq(
        self,
        bbox_a,
        bbox_b,
        class_name=None,
        field_position_a=None,
        field_position_b=None,
    ):
        if self._use_field_position_for_class(
            class_name,
            field_position_a,
        ) and self._use_field_position_for_class(class_name, field_position_b):
            ax, ay = self._field_position_to_tuple(field_position_a)
            bx, by = self._field_position_to_tuple(field_position_b)
            return (ax - bx) ** 2 + (ay - by) ** 2
        if bbox_a is None or bbox_b is None:
            return None
        ax, ay = self._bbox_center(bbox_a)
        bx, by = self._bbox_center(bbox_b)
        return (ax - bx) ** 2 + (ay - by) ** 2

    def _resolve_candidate_class_for_detection(
        self,
        candidate_state,
        detection_class,
        detection_team,
        detection_bbox,
        detection_field_position,
        current_frame,
    ):
        candidate_class = candidate_state.get("class_name")
        if candidate_class is None:
            return None

        if self._is_compatible_class(candidate_class, detection_class):
            if not self._is_team_compatible(
                candidate_state.get("team"),
                detection_team,
                candidate_class,
            ):
                return None
            if not self._is_motion_compatible(
                candidate_state,
                detection_bbox,
                current_frame,
                class_name=candidate_class,
                new_field_position=detection_field_position,
            ):
                return None

            # En árbitros, exigimos también cercanía espacial absoluta para evitar swaps lejanos.
            if candidate_class == "referee":
                distance_sq = self._bbox_distance_sq(
                    candidate_state.get("bbox"),
                    detection_bbox,
                    class_name=candidate_class,
                    field_position_a=candidate_state.get("field_position"),
                    field_position_b=detection_field_position,
                )
                if distance_sq is None:
                    return None
                if distance_sq > (self.referee_recovery_max_distance ** 2):
                    return None
            return candidate_class

        # Recuperación controlada de árbitro cuando entra como player por error.
        if candidate_class == "referee" and detection_class == "player":
            last_frame = int(candidate_state.get("last_frame", current_frame))
            lost_frames = max(0, current_frame - last_frame)
            if lost_frames > self.referee_recovery_max_lost_frames:
                return None
            if not self._is_motion_compatible(
                candidate_state,
                detection_bbox,
                current_frame,
                class_name=candidate_class,
                new_field_position=detection_field_position,
            ):
                return None
            distance_sq = self._bbox_distance_sq(
                candidate_state.get("bbox"),
                detection_bbox,
                class_name=candidate_class,
                field_position_a=candidate_state.get("field_position"),
                field_position_b=detection_field_position,
            )
            if distance_sq is None:
                return None
            if distance_sq > (self.referee_recovery_max_distance ** 2):
                return None
            return "referee"

        return None

    def _assign_pending_by_lost_order(
        self,
        pending_detections,
        available_ids,
        canonical_state,
        current_frame,
    ):
        assignments = {}
        assigned_pending_indexes = set()
        sorted_candidate_ids = sorted(
            available_ids,
            key=lambda canonical_id: (
                max(
                    0,
                    current_frame
                    - int(canonical_state[canonical_id].get("last_frame", current_frame)),
                ),
                canonical_id,
            ),
        )

        for canonical_id in sorted_candidate_ids:
            candidate_state = canonical_state[canonical_id]
            best_pending_idx = None
            best_distance_sq = None
            best_output_class = None

            for pending_idx, pending in enumerate(pending_detections):
                if pending_idx in assigned_pending_indexes:
                    continue

                output_class_name = self._resolve_candidate_class_for_detection(
                    candidate_state,
                    pending["preferred_class_name"],
                    pending["detected_team"],
                    pending["bbox"],
                    pending.get("field_position"),
                    current_frame,
                )
                if output_class_name is None:
                    continue

                distance_sq = self._bbox_distance_sq(
                    candidate_state.get("bbox"),
                    pending["bbox"],
                    class_name=output_class_name,
                    field_position_a=candidate_state.get("field_position"),
                    field_position_b=pending.get("field_position"),
                )
                if distance_sq is None:
                    continue
                if best_distance_sq is None or distance_sq < best_distance_sq:
                    best_distance_sq = distance_sq
                    best_pending_idx = pending_idx
                    best_output_class = output_class_name

            if best_pending_idx is not None:
                assignments[best_pending_idx] = (canonical_id, best_output_class)
                assigned_pending_indexes.add(best_pending_idx)

        return assignments

    @staticmethod
    def _priority_tuple(class_name, lost_frames, distance_sq):
        if class_name == "referee":
            # Para árbitros priorizamos cercanía espacial para evitar swaps lejanos.
            return (distance_sq, lost_frames)
        return (lost_frames, distance_sq)

    def _candidate_priority(
        self,
        bbox,
        candidate_state,
        current_frame,
        class_name=None,
        field_position=None,
    ):
        prev_bbox = candidate_state.get("bbox")
        prev_field_position = candidate_state.get("field_position")
        if prev_bbox is None and prev_field_position is None:
            return None
        effective_class = class_name or candidate_state.get("class_name")
        distance = self._bbox_distance_sq(
            bbox,
            prev_bbox,
            class_name=effective_class,
            field_position_a=field_position,
            field_position_b=prev_field_position,
        )
        if distance is None:
            return None
        last_frame = int(candidate_state.get("last_frame", current_frame))
        lost_frames = max(0, current_frame - last_frame)
        return self._priority_tuple(effective_class, lost_frames, distance)

    def _nearest_recent_referee_id(
        self,
        bbox,
        candidate_ids,
        canonical_state,
        current_frame,
        field_position=None,
    ):
        best_id = None
        best_priority = None
        max_distance_sq = self.referee_recovery_max_distance ** 2

        for canonical_id in candidate_ids:
            state = canonical_state[canonical_id]
            if state.get("class_name") != "referee":
                continue
            last_frame = int(state.get("last_frame", current_frame))
            lost_frames = max(0, current_frame - last_frame)
            if lost_frames > self.referee_recovery_max_lost_frames:
                continue
            if not self._is_motion_compatible(
                state,
                bbox,
                current_frame,
                class_name="referee",
                new_field_position=field_position,
            ):
                continue

            priority = self._candidate_priority(
                bbox,
                state,
                current_frame,
                class_name="referee",
                field_position=field_position,
            )
            if priority is None:
                continue
            prev_bbox = state.get("bbox")
            if prev_bbox is None:
                continue
            distance_sq = self._bbox_distance_sq(
                bbox,
                prev_bbox,
                class_name="referee",
                field_position_a=field_position,
                field_position_b=state.get("field_position"),
            )
            if distance_sq is None:
                continue
            if distance_sq > max_distance_sq:
                continue

            if best_priority is None or priority < best_priority:
                best_priority = priority
                best_id = canonical_id

        return best_id, best_priority

    def _count_ids_for_class(self, canonical_state, class_name):
        return sum(
            1
            for state in canonical_state.values()
            if state.get("class_name") == class_name
        )

    def _nearest_id_same_class(
        self,
        bbox,
        candidate_ids,
        canonical_state,
        class_name,
        current_frame,
        require_motion=True,
        max_distance=None,
        field_position=None,
    ):
        best_id = None
        best_priority = None
        max_distance_sq = None
        if max_distance is not None:
            max_distance_sq = float(max_distance) ** 2

        for canonical_id in candidate_ids:
            state = canonical_state[canonical_id]
            if state.get("class_name") != class_name:
                continue
            if require_motion and not self._is_motion_compatible(
                state,
                bbox,
                current_frame,
                class_name=class_name,
                new_field_position=field_position,
            ):
                continue

            priority = self._candidate_priority(
                bbox,
                state,
                current_frame,
                class_name=class_name,
                field_position=field_position,
            )
            if priority is None:
                continue
            if max_distance_sq is not None:
                prev_bbox = state.get("bbox")
                distance_sq = self._bbox_distance_sq(
                    bbox,
                    prev_bbox,
                    class_name=class_name,
                    field_position_a=field_position,
                    field_position_b=state.get("field_position"),
                )
                if distance_sq is None:
                    continue
                if distance_sq > max_distance_sq:
                    continue
            if best_priority is None or priority < best_priority:
                best_priority = priority
                best_id = canonical_id

        return best_id, best_priority

    def _nearest_available_canonical_id(
        self,
        bbox,
        candidate_ids,
        canonical_state,
        class_name,
        team_name,
        current_frame,
        field_position=None,
    ):
        if not candidate_ids:
            return None

        class_compatible_ids = [
            canonical_id
            for canonical_id in candidate_ids
            if self._is_compatible_class(
                canonical_state[canonical_id]["class_name"], class_name
            )
        ]
        if not class_compatible_ids:
            return None

        team_compatible_ids = [
            canonical_id
            for canonical_id in class_compatible_ids
            if self._is_team_compatible(
                canonical_state[canonical_id].get("team"),
                team_name,
                class_name,
            )
        ]
        if not team_compatible_ids:
            return None

        motion_compatible_ids = [
            canonical_id
            for canonical_id in team_compatible_ids
            if self._is_motion_compatible(
                canonical_state[canonical_id],
                bbox,
                current_frame,
                class_name=class_name,
                new_field_position=field_position,
            )
        ]
        if not motion_compatible_ids:
            return None

        best_id = None
        best_priority = None
        for canonical_id in motion_compatible_ids:
            priority = self._candidate_priority(
                bbox,
                canonical_state[canonical_id],
                current_frame,
                class_name=class_name,
                field_position=field_position,
            )
            if priority is None:
                continue
            priority = (*priority, canonical_id)
            if best_priority is None or priority < best_priority:
                best_priority = priority
                best_id = canonical_id

        return best_id
    
    def get_tracks(self, video, show_kmeans=False):
        model_detections = self.model.detect(video)
        tracks = {"player": [], "goalkeeper": [], "referee": [], "ball": [] }
        raw_to_canonical_id = {}
        canonical_to_raw_id = {}
        canonical_state = {}
        for n_frame, detections in enumerate(model_detections):
            detections_sv = sv.Detections.from_ultralytics(detections)

            teams_of_detected_objects = self.team_detector.detect_teams(detections, show_kmeans)
            teams_labels = [dicc["team"] for dicc in teams_of_detected_objects]
            class_labels = [dicc["class"] for dicc in teams_of_detected_objects]
            field_projection = None
            field_positions = np.full((len(detections_sv), 2), np.nan, dtype=np.float32)
            ground_points_projected = np.full((len(detections_sv), 2), np.nan, dtype=np.float32)
            if self.field_projector is not None:
                original_frame_bgr = getattr(detections, "orig_img", None)
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

            tracks_detection = self.tracker.update_with_detections(
                detections_sv,
                teams_labels,
                class_labels,
            )
            
            for key in tracks.keys():
                tracks[key].append({})

            has_tracked_ball = False
            used_canonical_ids_in_frame = set()
            sorted_tracked_detections = sorted(
                list(tracks_detection),
                key=lambda item: float(item[2]),
                reverse=True,
            )
            pending_detections = []

            def commit_assignment(
                raw_tracker_id,
                canonical_id,
                output_class_name,
                bbox,
                confidence,
                detected_team,
                field_position,
                metadata,
            ):
                previous_owner_raw_id = canonical_to_raw_id.get(canonical_id)
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
                resolved_team = (
                    detected_team
                    if detected_team is not None
                    else previous_state.get("team")
                )
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
                }
                used_canonical_ids_in_frame.add(canonical_id)

                tracks[output_class_name][n_frame][canonical_id] = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "team": metadata.get("team"),
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
                }

            for object_detected in sorted_tracked_detections:
                bbox, _, confidence, _, tracker_id, metadata = object_detected
                class_name = metadata["class_name"]
                if class_name not in tracks:
                    continue
                bbox = self._bbox_to_list(bbox)
                confidence = float(confidence)
                detected_team = metadata.get("team")
                field_position = metadata.get("field_position")

                if class_name == "ball":
                    has_tracked_ball = True
                    current_ball = tracks["ball"][n_frame].get(0)
                    if current_ball is None or confidence > float(current_ball["confidence"]):
                        tracks["ball"][n_frame][0] = {
                            "bbox": bbox,
                            "confidence": confidence,
                            "team": metadata.get("team"),
                            "distances": metadata.get("distances"),
                            "shirt_color": metadata.get("shirt_color"),
                            "bbox_size": metadata.get("bbox_size"),
                            "field_position_m": None,
                            "ground_point_image": None,
                        }
                    continue

                raw_tracker_id = int(tracker_id)
                canonical_id = raw_to_canonical_id.get(raw_tracker_id)
                output_class_name = class_name
                if canonical_id is not None:
                    previous_state = canonical_state.get(canonical_id)
                    if previous_state is None:
                        canonical_id = None
                    elif canonical_id in used_canonical_ids_in_frame:
                        canonical_id = None
                    else:
                        output_class_name = previous_state["class_name"]
                        if output_class_name not in tracks:
                            canonical_id = None
                        elif self._resolve_candidate_class_for_detection(
                            previous_state,
                            output_class_name,
                            detected_team,
                            bbox,
                            field_position,
                            n_frame,
                        ) is None:
                            canonical_id = None

                if canonical_id is None:
                    pending_detections.append(
                        {
                            "raw_tracker_id": raw_tracker_id,
                            "bbox": bbox,
                            "confidence": confidence,
                            "detected_team": detected_team,
                            "field_position": field_position,
                            "metadata": metadata,
                            "preferred_class_name": output_class_name,
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
                        )
                        if class_count >= int(class_limit):
                            continue
                    next_free_id = self._next_free_canonical_id(canonical_state)
                    if next_free_id is None:
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
                )

            # Si el tracker no activó el balón, usar detecciones crudas (sin tracking)
            if not has_tracked_ball and detections.boxes is not None and len(detections.boxes) > 0:
                boxes = detections.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                best_ball = None
                for bbox, score, cid in zip(xyxy, conf, cls):
                    class_name = detections.names[cid]
                    if class_name != "ball" or score < self.ball_min_conf:
                        continue
                    if best_ball is None or score > best_ball[1]:
                        best_ball = (bbox, score)

                if best_ball is not None:
                    bbox, score = best_ball
                    x1, y1, x2, y2 = bbox.tolist()
                    tracks["ball"][n_frame][0] = {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(score),
                        "team": None,
                        "distances": None,
                        "shirt_color": None,
                        "bbox_size": float((x2 - x1) * (y2 - y1)),
                        "field_position_m": None,
                        "ground_point_image": None,
                    }
        
        return tracks
