import logging
from time import perf_counter

import numpy as np
import supervision as sv

from football_ai.detection import Detector
from football_ai.identification import TeamDetector
from football_ai.reference_points import PnLCalibFieldProjector
from football_ai.tracking.byte_tracker import ByteTrack

logger = logging.getLogger(__name__)

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
        self.team_inference_cache_enabled = bool(
            tracker_conf.get("team_inference_cache_enabled", False)
        )
        self.team_inference_cache_iou_threshold = float(
            tracker_conf.get("team_inference_cache_iou_threshold", 0.35)
        )
        self.team_inference_classes = frozenset(
            tracker_conf.get("team_inference_classes", ["player", "goalkeeper"])
        )
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
        self.timing_enabled = bool(tracker_conf.get("timing_enabled", False))
        self.timing_log_every_n_frames = max(
            1, int(tracker_conf.get("timing_log_every_n_frames", 25))
        )
        self.timing_save_per_frame = bool(
            tracker_conf.get("timing_save_per_frame", True)
        )
        self.last_timing_summary = {}
        self.last_timing_frames = []

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
    def _bbox_iou(bbox_a, bbox_b):
        ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a]
        bx1, by1, bx2, by2 = [float(v) for v in bbox_b]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    def _build_cached_team_assignments(
        self,
        detections_xyxy,
        class_labels,
        canonical_state,
        current_frame,
        previous_frame_candidates=None,
    ):
        if not self.team_inference_cache_enabled:
            return {}

        detections_array = np.asarray(detections_xyxy, dtype=np.float32).reshape(-1, 4)
        if len(detections_array) == 0:
            return {}

        assignments = {}
        used_previous_candidate_indices = set()
        if previous_frame_candidates:
            for detection_index, detection_bbox in enumerate(detections_array):
                if detection_index >= len(class_labels):
                    continue
                class_name = class_labels[detection_index]
                if class_name not in self.team_inference_classes:
                    continue

                best_candidate_index = None
                best_iou = 0.0
                for candidate_index, candidate in enumerate(previous_frame_candidates):
                    if candidate_index in used_previous_candidate_indices:
                        continue
                    if candidate.get("class_name") != class_name:
                        continue
                    if candidate.get("team") is None:
                        continue
                    candidate_bbox = candidate.get("bbox")
                    if candidate_bbox is None:
                        continue
                    iou = self._bbox_iou(detection_bbox, candidate_bbox)
                    if iou < self.team_inference_cache_iou_threshold:
                        continue
                    if iou > best_iou:
                        best_iou = iou
                        best_candidate_index = candidate_index

                if best_candidate_index is None:
                    continue

                candidate = previous_frame_candidates[best_candidate_index]
                used_previous_candidate_indices.add(best_candidate_index)
                x1, y1, x2, y2 = [float(v) for v in detection_bbox]
                assignments[detection_index] = {
                    "team": candidate.get("team"),
                    "distances": None,
                    "shirt_color": candidate.get("shirt_color"),
                    "bbox_size": float(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
                }

        previous_state_candidates = []
        for canonical_id, state in canonical_state.items():
            if state.get("team") is None:
                continue
            if state.get("bbox") is None:
                continue
            class_name = state.get("class_name")
            if class_name not in self.team_inference_classes:
                continue
            if int(state.get("last_frame", -999999)) != int(current_frame) - 1:
                continue
            previous_state_candidates.append((canonical_id, state))

        if not previous_state_candidates:
            return assignments

        used_state_candidate_ids = set()
        for detection_index, detection_bbox in enumerate(detections_array):
            if detection_index in assignments:
                continue
            if detection_index >= len(class_labels):
                continue
            class_name = class_labels[detection_index]
            if class_name not in self.team_inference_classes:
                continue

            best_candidate = None
            best_iou = 0.0
            for canonical_id, state in previous_state_candidates:
                if canonical_id in used_state_candidate_ids:
                    continue
                if state.get("class_name") != class_name:
                    continue
                iou = self._bbox_iou(detection_bbox, state.get("bbox"))
                if iou < self.team_inference_cache_iou_threshold:
                    continue
                if iou > best_iou:
                    best_iou = iou
                    best_candidate = (canonical_id, state)

            if best_candidate is None:
                continue

            canonical_id, state = best_candidate
            used_state_candidate_ids.add(canonical_id)
            x1, y1, x2, y2 = [float(v) for v in detection_bbox]
            assignments[detection_index] = {
                "team": state.get("team"),
                "distances": None,
                "shirt_color": state.get("shirt_color"),
                "bbox_size": float(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
            }
        return assignments

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

    @staticmethod
    def _summarize_timing_frames(timing_frames):
        if not timing_frames:
            return {"frames": 0, "phases": {}, "metrics": {}}

        keys = [key for key in timing_frames[0].keys() if key != "frame_index"]
        phase_names = [key for key in keys if key.endswith("_s")]
        metric_names = [key for key in keys if not key.endswith("_s")]

        phase_summary = {}
        for phase_name in phase_names:
            phase_values = np.array(
                [frame_timing[phase_name] for frame_timing in timing_frames],
                dtype=np.float64,
            )
            phase_summary[phase_name] = {
                "total_s": float(np.sum(phase_values)),
                "mean_s": float(np.mean(phase_values)),
                "max_s": float(np.max(phase_values)),
                "p95_s": float(np.percentile(phase_values, 95)),
            }

        metrics_summary = {}
        for metric_name in metric_names:
            metric_values = np.array(
                [frame_timing[metric_name] for frame_timing in timing_frames],
                dtype=np.float64,
            )
            metrics_summary[metric_name] = {
                "mean": float(np.mean(metric_values)),
                "min": float(np.min(metric_values)),
                "max": float(np.max(metric_values)),
                "p95": float(np.percentile(metric_values, 95)),
            }

        return {
            "frames": len(timing_frames),
            "phases": phase_summary,
            "metrics": metrics_summary,
        }
    
    def get_tracks(self, video, show_kmeans=False):
        model_detections_iter = iter(self.model.detect(video))
        tracks = {"player": [], "goalkeeper": [], "referee": [], "ball": []}
        raw_to_canonical_id = {}
        canonical_to_raw_id = {}
        canonical_state = {}
        previous_frame_team_candidates = []
        timing_frames = []
        n_frame = 0

        while True:
            frame_start = perf_counter()
            try:
                detections = next(model_detections_iter)
            except StopIteration:
                break
            detection_fetch_s = perf_counter() - frame_start

            sv_start = perf_counter()
            detections_sv = sv.Detections.from_ultralytics(detections)
            detections_to_sv_s = perf_counter() - sv_start

            class_ids = (
                np.asarray(detections_sv.class_id, dtype=np.int32).reshape(-1)
                if detections_sv.class_id is not None
                else np.zeros((len(detections_sv),), dtype=np.int32)
            )
            class_labels = [
                detections.names[int(class_id)]
                for class_id in class_ids
            ]
            cached_team_assignments = self._build_cached_team_assignments(
                detections_xyxy=detections_sv.xyxy,
                class_labels=class_labels,
                canonical_state=canonical_state,
                current_frame=n_frame,
                previous_frame_candidates=previous_frame_team_candidates,
            )

            team_detection_start = perf_counter()
            teams_of_detected_objects = self.team_detector.detect_teams(
                detections,
                show_kmeans,
                cached_assignments=cached_team_assignments,
                compute_for_classes=self.team_inference_classes,
            )
            team_timing = getattr(self.team_detector, "last_timing_detail", {}) or {}
            teams_labels = [dicc["team"] for dicc in teams_of_detected_objects]
            team_detection_s = perf_counter() - team_detection_start

            previous_frame_team_candidates = []
            detections_xyxy_array = np.asarray(
                detections_sv.xyxy,
                dtype=np.float32,
            ).reshape(-1, 4)
            for detection_index, detection_bbox in enumerate(detections_xyxy_array):
                if detection_index >= len(class_labels):
                    continue
                if detection_index >= len(teams_of_detected_objects):
                    continue
                class_name = class_labels[detection_index]
                if class_name not in self.team_inference_classes:
                    continue
                team_info = teams_of_detected_objects[detection_index]
                team_name = team_info.get("team")
                if team_name is None:
                    continue
                x1, y1, x2, y2 = [float(value) for value in detection_bbox]
                previous_frame_team_candidates.append(
                    {
                        "class_name": class_name,
                        "bbox": [x1, y1, x2, y2],
                        "team": team_name,
                        "shirt_color": team_info.get("shirt_color"),
                        "bbox_size": float(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
                    }
                )

            field_projection_start = perf_counter()
            field_positions = np.full((len(detections_sv), 2), np.nan, dtype=np.float32)
            ground_points_projected = np.full(
                (len(detections_sv), 2), np.nan, dtype=np.float32
            )
            field_timing = {}
            if self.field_projector is not None:
                original_frame_bgr = getattr(detections, "orig_img", None)
                if original_frame_bgr is not None:
                    field_projection = self.field_projector.project_detections(
                        original_frame_bgr,
                        detections_sv.xyxy,
                        class_names=class_labels,
                    )
                    field_positions = field_projection.field_positions_m
                    ground_points_projected = (
                        field_projection.ground_points_image_projected
                    )
                    field_timing = (
                        getattr(self.field_projector, "last_timing_detail", {}) or {}
                    )
            field_projection_s = perf_counter() - field_projection_start

            metadata_pack_start = perf_counter()
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
                [
                    (
                        tuple(
                            float(channel)
                            for channel in np.asarray(dicc["shirt_color"]).reshape(-1)[:3]
                        )
                        if dicc["shirt_color"] is not None
                        else None
                    )
                    for dicc in teams_of_detected_objects
                ],
                dtype=object,
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
            metadata_pack_s = perf_counter() - metadata_pack_start

            tracker_update_start = perf_counter()
            tracks_detection = self.tracker.update_with_detections(
                detections_sv,
                teams_labels,
                class_labels,
            )
            bytetrack_timing = getattr(self.tracker, "last_timing_detail", {}) or {}
            tracker_update_s = perf_counter() - tracker_update_start

            assignment_start = perf_counter()
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
                resolved_shirt_color = (
                    metadata.get("shirt_color")
                    if metadata.get("shirt_color") is not None
                    else previous_state.get("shirt_color")
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
                    "shirt_color": resolved_shirt_color,
                    "field_position": resolved_field_position,
                    "movement_samples": movement_samples,
                    "mean_step_distance": mean_step_distance,
                }
                used_canonical_ids_in_frame.add(canonical_id)

                tracks[output_class_name][n_frame][canonical_id] = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "team": resolved_team,
                    "distances": metadata.get("distances"),
                    "shirt_color": resolved_shirt_color,
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
                    if current_ball is None or confidence > float(
                        current_ball["confidence"]
                    ):
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
            id_assignment_s = perf_counter() - assignment_start

            ball_fallback_start = perf_counter()
            # Si el tracker no activó el balón, usar detecciones crudas (sin tracking)
            if (
                not has_tracked_ball
                and detections.boxes is not None
                and len(detections.boxes) > 0
            ):
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
            ball_fallback_s = perf_counter() - ball_fallback_start

            if self.timing_enabled:
                track_total_s = tracker_update_s + id_assignment_s + ball_fallback_s
                frame_timing = {
                    "frame_index": int(n_frame),
                    "detection_fetch_s": float(detection_fetch_s),
                    "detection_inference_s": float(detection_fetch_s),
                    "detections_to_sv_s": float(detections_to_sv_s),
                    "team_detection_s": float(team_detection_s),
                    "team_kmeans_s": float(team_timing.get("team_kmeans_s", 0.0)),
                    "team_clustering_s": float(
                        team_timing.get("team_clustering_s", 0.0)
                    ),
                    "team_kmeans_fit_s": float(
                        team_timing.get("team_kmeans_fit_s", 0.0)
                    ),
                    "team_color_update_s": float(
                        team_timing.get("team_color_update_s", 0.0)
                    ),
                    "team_assign_s": float(team_timing.get("team_assign_s", 0.0)),
                    "team_crop_player_s": float(
                        team_timing.get("team_crop_player_s", 0.0)
                    ),
                    "team_crop_shirt_s": float(
                        team_timing.get("team_crop_shirt_s", 0.0)
                    ),
                    "team_objects_count": float(
                        team_timing.get("team_objects_count", 0.0)
                    ),
                    "team_objects_valid_crop_count": float(
                        team_timing.get("team_objects_valid_crop_count", 0.0)
                    ),
                    "team_reused_count": float(
                        team_timing.get("team_reused_count", 0.0)
                    ),
                    "team_kmeans_computed_count": float(
                        team_timing.get("team_kmeans_computed_count", 0.0)
                    ),
                    "team_kmeans_skipped_count": float(
                        team_timing.get("team_kmeans_skipped_count", 0.0)
                    ),
                    "field_projection_s": float(field_projection_s),
                    "field_homography_estimation_s": float(
                        field_timing.get("field_homography_estimation_s", 0.0)
                    ),
                    "field_pnl_total_s": float(field_timing.get("pnl_total_s", 0.0)),
                    "field_pnl_streaming_total_s": float(
                        field_timing.get("pnl_streaming_total_s", 0.0)
                    ),
                    "field_pnl_preprocess_color_convert_s": float(
                        field_timing.get("pnl_preprocess_color_convert_s", 0.0)
                    ),
                    "field_pnl_preprocess_tensor_build_s": float(
                        field_timing.get("pnl_preprocess_tensor_build_s", 0.0)
                    ),
                    "field_pnl_preprocess_resize_s": float(
                        field_timing.get("pnl_preprocess_resize_s", 0.0)
                    ),
                    "field_pnl_preprocess_to_device_s": float(
                        field_timing.get("pnl_preprocess_to_device_s", 0.0)
                    ),
                    "field_pnl_model_kp_inference_s": float(
                        field_timing.get("pnl_model_kp_inference_s", 0.0)
                    ),
                    "field_pnl_model_line_inference_s": float(
                        field_timing.get("pnl_model_line_inference_s", 0.0)
                    ),
                    "field_pnl_decode_keypoints_s": float(
                        field_timing.get("pnl_decode_keypoints_s", 0.0)
                    ),
                    "field_pnl_decode_lines_s": float(
                        field_timing.get("pnl_decode_lines_s", 0.0)
                    ),
                    "field_pnl_complete_keypoints_s": float(
                        field_timing.get("pnl_complete_keypoints_s", 0.0)
                    ),
                    "field_pnl_calibrator_init_s": float(
                        field_timing.get("pnl_calibrator_init_s", 0.0)
                    ),
                    "field_pnl_calibrator_update_s": float(
                        field_timing.get("pnl_calibrator_update_s", 0.0)
                    ),
                    "field_pnl_ground_voting_s": float(
                        field_timing.get("pnl_ground_voting_s", 0.0)
                    ),
                    "field_pnl_camera_voting_s": float(
                        field_timing.get("pnl_camera_voting_s", 0.0)
                    ),
                    "field_pnl_homography_select_s": float(
                        field_timing.get("pnl_homography_select_s", 0.0)
                    ),
                    "field_pnl_centered_to_field_s": float(
                        field_timing.get("pnl_centered_to_field_s", 0.0)
                    ),
                    "field_pnl_template_homography_s": float(
                        field_timing.get("pnl_template_homography_s", 0.0)
                    ),
                    "field_pnl_projection_matrix_s": float(
                        field_timing.get("pnl_projection_matrix_s", 0.0)
                    ),
                    "field_pnl_temporal_smoothing_s": float(
                        field_timing.get("pnl_temporal_smoothing_s", 0.0)
                    ),
                    "field_pnl_previous_fallback_s": float(
                        field_timing.get("pnl_previous_fallback_s", 0.0)
                    ),
                    "field_resize_s": float(field_timing.get("field_resize_s", 0.0)),
                    "field_project_points_s": float(
                        field_timing.get("field_project_points_s", 0.0)
                    ),
                    "field_ground_points_s": float(
                        field_timing.get("field_ground_points_s", 0.0)
                    ),
                    "field_scale_points_s": float(
                        field_timing.get("field_scale_points_s", 0.0)
                    ),
                    "field_class_filter_s": float(
                        field_timing.get("field_class_filter_s", 0.0)
                    ),
                    "field_has_homography": float(
                        field_timing.get("field_has_homography", 0.0)
                    ),
                    "field_pnl_visible_keypoints_count": float(
                        field_timing.get("pnl_visible_keypoints_count", 0.0)
                    ),
                    "field_pnl_visible_lines_count": float(
                        field_timing.get("pnl_visible_lines_count", 0.0)
                    ),
                    "field_pnl_used_ground_solution": float(
                        field_timing.get("pnl_used_ground_solution", 0.0)
                    ),
                    "field_pnl_used_camera_fallback": float(
                        field_timing.get("pnl_used_camera_fallback", 0.0)
                    ),
                    "field_pnl_temporal_smoothed": float(
                        field_timing.get("pnl_temporal_smoothed", 0.0)
                    ),
                    "field_pnl_previous_fallback_applied": float(
                        field_timing.get("pnl_previous_fallback_applied", 0.0)
                    ),
                    "metadata_pack_s": float(metadata_pack_s),
                    "tracker_update_s": float(tracker_update_s),
                    "track_total_s": float(track_total_s),
                    "bytetrack_total_s": float(
                        bytetrack_timing.get("bytetrack_total_s", 0.0)
                    ),
                    "bytetrack_preprocess_s": float(
                        bytetrack_timing.get("bytetrack_preprocess_s", 0.0)
                    ),
                    "bytetrack_update_tensors_s": float(
                        bytetrack_timing.get("bytetrack_update_tensors_s", 0.0)
                    ),
                    "bytetrack_output_match_s": float(
                        bytetrack_timing.get("bytetrack_output_match_s", 0.0)
                    ),
                    "bytetrack_tracks_output_count": float(
                        bytetrack_timing.get("bytetrack_tracks_output_count", 0.0)
                    ),
                    "id_assignment_s": float(id_assignment_s),
                    "ball_fallback_s": float(ball_fallback_s),
                    "frame_total_s": float(perf_counter() - frame_start),
                }
                timing_frames.append(frame_timing)
                if (n_frame + 1) % self.timing_log_every_n_frames == 0:
                    window = timing_frames[-self.timing_log_every_n_frames :]
                    mean_total = float(
                        np.mean([entry["frame_total_s"] for entry in window])
                    )
                    mean_detection = float(
                        np.mean([entry["detection_fetch_s"] for entry in window])
                    )
                    mean_team = float(
                        np.mean([entry["team_detection_s"] for entry in window])
                    )
                    mean_projection = float(
                        np.mean([entry["field_projection_s"] for entry in window])
                    )
                    mean_tracker = float(
                        np.mean([entry["tracker_update_s"] for entry in window])
                    )
                    logger.info(
                        (
                            "Timing tracking frames %d-%d | frame=%.4fs | detect=%.4fs "
                            "| team=%.4fs | field=%.4fs | bytetrack=%.4fs"
                        ),
                        n_frame + 1 - len(window),
                        n_frame,
                        mean_total,
                        mean_detection,
                        mean_team,
                        mean_projection,
                        mean_tracker,
                    )

            n_frame += 1

        if self.timing_enabled:
            self.last_timing_summary = self._summarize_timing_frames(timing_frames)
            self.last_timing_frames = (
                timing_frames if self.timing_save_per_frame else []
            )
        else:
            self.last_timing_summary = {}
            self.last_timing_frames = []

        return tracks
