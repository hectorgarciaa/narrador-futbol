import math

import numpy as np
import supervision as sv

from football_ai.detection import Detector
from football_ai.identification import TeamDetector
from football_ai.reference_points import PnLCalibFieldProjector
from football_ai.tracking.byte_tracker import ByteTrack

class Tracker:
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

    def __init__(
        self,
        model_path,
        conf,
        tracker_conf,
        team_colors,
        team_detector_conf=None,
        ball_min_conf=0.01,
        max_tracks_per_class=None,
        field_tracking_conf=None,
        project_root=None,
    ):
        if max_tracks_per_class is None:
            max_tracks_per_class = {"player": 22, "ball": 1, "referee": 3}
        if field_tracking_conf is None:
            field_tracking_conf = {}
        if team_detector_conf is None:
            team_detector_conf = {}

        self.model = Detector(model_path, conf)
        # Compatibilidad: nueva convención snake_case y alias legacy camelCase.
        self.team_detector = TeamDetector(team_colors, **team_detector_conf)
        self.teamDetector = self.team_detector
        self.max_tracks_per_class = max_tracks_per_class
        self.max_total_tracks = int(
            tracker_conf.get("max_total_tracks", sum(max_tracks_per_class.values()))
        )
        self.reserve_penalty_spot_seed_players = bool(
            tracker_conf.get("reserve_penalty_spot_seed_players", False)
        )
        self.reserve_penalty_spot_seed_match_distance_m = float(
            tracker_conf.get("reserve_penalty_spot_seed_match_distance_m", 12.0)
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
            field_distance_gate_max_lost_frames=field_tracking_conf.get(
                "match_distance_max_lost_frames"
            ),
            field_distance_gate_cap_m=field_tracking_conf.get(
                "match_distance_cap_m"
            ),
            field_distance_growth_mode=field_tracking_conf.get(
                "match_distance_growth_mode", "power"
            ),
            field_distance_lost_exponent=field_tracking_conf.get(
                "match_distance_lost_exponent", 1.0
            ),
            field_distance_decay_per_frame=field_tracking_conf.get(
                "match_distance_decay_per_frame", 0.0
            ),
            lost_time_penalty_weight=tracker_conf.get(
                "lost_time_penalty_weight", 0.0
            ),
            lost_time_penalty_max_frames=tracker_conf.get(
                "lost_time_penalty_max_frames", 10
            ),
            use_field_position_as_primary_cost=tracker_conf.get(
                "use_field_position_as_primary_cost", False
            ),
            use_bbox_center_for_matching=tracker_conf.get(
                "use_bbox_center_for_matching", True
            ),
            bbox_center_distance_weight=tracker_conf.get(
                "bbox_center_distance_weight", 0.7
            ),
            bbox_center_distance_gate_px=tracker_conf.get(
                "bbox_center_distance_gate_px", 120.0
            ),
        )
        self.ball_min_conf = ball_min_conf
        self.ball_expected_position_gate_px = float(
            tracker_conf.get("ball_expected_position_gate_px", 90.0)
        )
        self.ball_expected_position_gate_growth_per_frame = float(
            tracker_conf.get("ball_expected_position_gate_growth_per_frame", 35.0)
        )
        self.ball_expected_position_confidence_relax = float(
            tracker_conf.get("ball_expected_position_confidence_relax", 1.4)
        )
        self.ball_size_ratio_per_frame = float(
            tracker_conf.get("ball_size_ratio_per_frame", 1.8)
        )
        self.ball_size_min_samples = int(
            tracker_conf.get("ball_size_min_samples", 5)
        )
        self.ball_size_std_factor = float(
            tracker_conf.get("ball_size_std_factor", 3.0)
        )
        self.ball_size_std_floor = float(
            tracker_conf.get("ball_size_std_floor", 1.0)
        )
        self.ball_max_reassign_lost_frames = self._normalize_optional_positive_int(
            tracker_conf.get("ball_max_reassign_lost_frames", 4)
        )
        self.ball_high_conf_override = float(
            tracker_conf.get("ball_high_conf_override", 0.6)
        )
        self.reassign_motion_factor = float(
            tracker_conf.get("reassign_motion_factor", 4.0)
        )
        self.reassign_min_distance = float(
            tracker_conf.get("reassign_min_distance", 25.0)
        )
        self.reassign_min_samples = int(
            tracker_conf.get("reassign_min_samples", 3)
        )
        self.reassign_motion_growth_cap_frames = self._normalize_optional_positive_int(
            tracker_conf.get("reassign_motion_growth_cap_frames", 12)
        )
        self.motion_std_gate_enabled = bool(
            tracker_conf.get("motion_std_gate_enabled", True)
        )
        self.motion_std_factor = float(
            tracker_conf.get("motion_std_factor", 10.0)
        )
        self.motion_std_min_samples = int(
            tracker_conf.get("motion_std_min_samples", 8)
        )
        self.motion_std_floor = float(
            tracker_conf.get("motion_std_floor", 0.5)
        )
        self.class_motion_stats = {
            "player": {"count": 0, "mean": 0.0, "m2": 0.0},
            "goalkeeper": {"count": 0, "mean": 0.0, "m2": 0.0},
            "referee": {"count": 0, "mean": 0.0, "m2": 0.0},
            "ball": {"count": 0, "mean": 0.0, "m2": 0.0},
        }
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
        self.field_distance_gate_m = float(
            field_tracking_conf.get("match_distance_gate_m", 8.0)
        )
        self.field_distance_gate_max_lost_frames = self._normalize_optional_positive_int(
            field_tracking_conf.get("match_distance_max_lost_frames")
        )
        raw_field_distance_gate_cap_m = field_tracking_conf.get("match_distance_cap_m")
        if raw_field_distance_gate_cap_m is None:
            self.field_distance_gate_cap_m = None
        else:
            gate_cap = float(raw_field_distance_gate_cap_m)
            self.field_distance_gate_cap_m = (
                gate_cap if np.isfinite(gate_cap) and gate_cap > 0.0 else None
            )
        self.field_distance_growth_mode = str(
            field_tracking_conf.get("match_distance_growth_mode", "power") or "power"
        ).strip().lower()
        if self.field_distance_growth_mode not in {"power", "linear_decay"}:
            self.field_distance_growth_mode = "power"
        self.field_distance_lost_exponent = float(
            field_tracking_conf.get("match_distance_lost_exponent", 1.0)
        )
        if (
            not np.isfinite(self.field_distance_lost_exponent)
            or self.field_distance_lost_exponent <= 0.0
        ):
            self.field_distance_lost_exponent = 1.0
        self.field_distance_decay_per_frame = float(
            field_tracking_conf.get("match_distance_decay_per_frame", 0.0)
        )
        if (
            not np.isfinite(self.field_distance_decay_per_frame)
            or self.field_distance_decay_per_frame < 0.0
        ):
            self.field_distance_decay_per_frame = 0.0
        self.strict_person_class_separation = bool(
            tracker_conf.get("strict_person_class_separation", True)
        )
        self.require_field_position_for_reassign = bool(
            tracker_conf.get("require_field_position_for_reassign", True)
        )
        self.max_reassign_lost_frames = self._normalize_optional_positive_int(
            tracker_conf.get("max_reassign_lost_frames")
        )
        raw_max_reassign_lost_frames_by_class = (
            tracker_conf.get("max_reassign_lost_frames_by_class", {}) or {}
        )
        self.max_reassign_lost_frames_by_class = {}
        for class_name, class_limit in raw_max_reassign_lost_frames_by_class.items():
            self.max_reassign_lost_frames_by_class[str(class_name)] = (
                self._normalize_optional_positive_int(class_limit)
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
    def _bbox_area(bbox):
        x1, y1, x2, y2 = bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    def _ball_size_measure(self, bbox):
        return float(math.sqrt(max(self._bbox_area(bbox), 0.0)))

    @staticmethod
    def _field_position_to_tuple(field_position):
        if field_position is None:
            return None
        field_position = np.asarray(field_position, dtype=np.float32).reshape(-1)
        if field_position.size < 2 or not np.all(np.isfinite(field_position[:2])):
            return None
        return (float(field_position[0]), float(field_position[1]))

    @staticmethod
    def _update_running_stats(prev_count, prev_mean, prev_m2, value):
        """Update running mean and variance accumulator (Welford)."""
        count = int(prev_count) + 1
        delta = float(value) - float(prev_mean)
        mean = float(prev_mean) + (delta / count)
        delta2 = float(value) - mean
        m2 = float(prev_m2) + (delta * delta2)
        return count, mean, m2

    def _motion_limit_per_frame(self, stats_count, stats_mean, stats_m2):
        """Compute per-frame motion cap from running mean/std stats."""
        variance = float(stats_m2) / max(1, int(stats_count) - 1)
        std_per_frame = max(
            float(np.sqrt(max(variance, 0.0))),
            self.motion_std_floor,
        )
        return float(stats_mean) + (self.motion_std_factor * std_per_frame)

    def _use_field_position_for_class(self, class_name, field_position=None):
        if not self.use_field_positions:
            return False
        if class_name not in self.field_position_classes:
            return False
        return self._field_position_to_tuple(field_position) is not None

    def _must_use_field_position_for_class(self, class_name):
        if not self.require_field_position_for_reassign:
            return False
        return bool(self.use_field_positions and class_name in self.field_position_classes)

    def _max_lost_frames_for_class(self, class_name):
        if class_name in self.max_reassign_lost_frames_by_class:
            class_limit = self.max_reassign_lost_frames_by_class.get(class_name)
            return class_limit
        return self.max_reassign_lost_frames

    def _next_free_canonical_id(self, canonical_state):
        for canonical_id in range(1, self.max_total_tracks + 1):
            if canonical_id not in canonical_state:
                return canonical_id
        return None

    def _reserved_penalty_spot_field_positions(self):
        if (
            not self.reserve_penalty_spot_seed_players
            or not self.use_field_positions
            or self.field_projector is None
        ):
            return []

        geometry = getattr(self.field_projector, "geometry", None)
        if geometry is None:
            return []

        center_y_m = float(getattr(geometry, "center_y_m", geometry.field_width_m / 2.0))
        penalty_mark_distance_m = float(getattr(geometry, "penalty_mark_distance_m", 11.0))
        field_length_m = float(getattr(geometry, "field_length_m", 106.0))
        return [
            (penalty_mark_distance_m, center_y_m),
            (field_length_m - penalty_mark_distance_m, center_y_m),
        ]

    def _build_reserved_penalty_spot_state(self, field_position):
        return {
            "bbox": None,
            "class_name": "player",
            "last_frame": 0,
            "team": None,
            "field_position": tuple(float(v) for v in field_position),
            "movement_samples": 0,
            "mean_step_distance": 0.0,
            "step_per_frame_count": 0,
            "step_per_frame_mean": 0.0,
            "step_per_frame_m2": 0.0,
            "reserved_seed": True,
            "reserved_seed_match_distance_m": float(
                max(0.0, self.reserve_penalty_spot_seed_match_distance_m)
            ),
            "reserved_seed_origin": "penalty_spot",
        }

    @staticmethod
    def _project_homography_point(point_xy, homography):
        if point_xy is None or homography is None:
            return None
        point = np.asarray(point_xy, dtype=np.float32).reshape(-1)
        if point.size < 2 or not np.all(np.isfinite(point[:2])):
            return None
        try:
            inverse_homography = np.linalg.inv(np.asarray(homography, dtype=np.float32))
        except np.linalg.LinAlgError:
            return None

        homogeneous_point = np.array([point[0], point[1], 1.0], dtype=np.float32)
        image_point = inverse_homography @ homogeneous_point
        if not np.all(np.isfinite(image_point)) or abs(float(image_point[2])) < 1e-6:
            return None
        return (
            float(image_point[0] / image_point[2]),
            float(image_point[1] / image_point[2]),
        )

    @staticmethod
    def _scale_point(point_xy, source_shape_hw, target_shape_hw):
        if point_xy is None:
            return None
        source_height, source_width = source_shape_hw
        target_height, target_width = target_shape_hw
        if (
            source_height <= 0
            or source_width <= 0
            or target_height <= 0
            or target_width <= 0
        ):
            return None
        scale_x = float(target_width) / float(source_width)
        scale_y = float(target_height) / float(source_height)
        return (float(point_xy[0]) * scale_x, float(point_xy[1]) * scale_y)

    @staticmethod
    def _seed_bbox_from_ground_point(image_point_original, frame_shape_original):
        if image_point_original is None or frame_shape_original is None:
            return None
        frame_height, frame_width = frame_shape_original
        if frame_height <= 0 or frame_width <= 0:
            return None

        x_coord = float(image_point_original[0])
        y_coord = float(image_point_original[1])
        if (
            not np.isfinite(x_coord)
            or not np.isfinite(y_coord)
            or x_coord < 0.0
            or x_coord > float(frame_width)
            or y_coord < 0.0
            or y_coord > float(frame_height)
        ):
            return None

        bbox_width = max(8.0, min(float(frame_width) * 0.012, 24.0))
        bbox_height = max(18.0, min(float(frame_height) * 0.05, 48.0))
        x1 = max(0.0, x_coord - (bbox_width / 2.0))
        x2 = min(float(frame_width - 1), x_coord + (bbox_width / 2.0))
        y2 = min(float(frame_height - 1), y_coord)
        y1 = max(0.0, y2 - bbox_height)
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _build_reserved_seed_track_payload(self, state, field_projection):
        field_position = self._field_position_to_tuple(state.get("field_position"))
        projected_ground_point = None
        original_ground_point = None
        frame_shape_original = None
        if field_projection is not None:
            projected_ground_point = self._project_homography_point(
                field_position,
                field_projection.homography_image_to_field,
            )
            frame_shape_original = getattr(field_projection, "frame_shape_original", None)
            frame_shape_projected = getattr(field_projection, "frame_shape_projected", None)
            if (
                projected_ground_point is not None
                and frame_shape_original is not None
                and frame_shape_projected is not None
            ):
                original_ground_point = self._scale_point(
                    projected_ground_point,
                    frame_shape_projected,
                    frame_shape_original,
                )

        bbox = self._seed_bbox_from_ground_point(original_ground_point, frame_shape_original)
        bbox_size = float(self._bbox_area(bbox)) if bbox is not None else 0.0
        return {
            "bbox": bbox,
            "confidence": 0.0,
            "team": None,
            "distances": None,
            "shirt_color": None,
            "bbox_size": bbox_size,
            "field_position_m": list(field_position) if field_position is not None else None,
            "ground_point_image": (
                [float(projected_ground_point[0]), float(projected_ground_point[1])]
                if projected_ground_point is not None
                else None
            ),
            "synthetic_seed": True,
        }

    def _initialize_reserved_penalty_spot_players(self, canonical_state):
        for field_position in self._reserved_penalty_spot_field_positions():
            next_free_id = self._next_free_canonical_id(canonical_state)
            if next_free_id is None:
                break
            canonical_state[next_free_id] = self._build_reserved_penalty_spot_state(
                field_position
            )

    def _is_compatible_class(self, previous_class, new_class):
        if previous_class is None or new_class is None:
            return previous_class == new_class
        person_classes = {"player", "goalkeeper"}
        if previous_class in person_classes and new_class in person_classes:
            if self.strict_person_class_separation:
                return previous_class == new_class
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
        if self._must_use_field_position_for_class(class_name):
            previous_field_position = self._field_position_to_tuple(previous_field_position)
            new_field_position = self._field_position_to_tuple(new_field_position)
            if previous_field_position is None or new_field_position is None:
                return None
            px, py = previous_field_position
            nx, ny = new_field_position
            return float(((nx - px) ** 2 + (ny - py) ** 2) ** 0.5)
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

    def _field_distance_gate_for_lost_frames(self, lost_frames):
        gate_lost_frames = max(1, int(lost_frames))
        if self.field_distance_gate_max_lost_frames is not None:
            gate_lost_frames = min(
                gate_lost_frames,
                self.field_distance_gate_max_lost_frames,
            )
        if self.field_distance_growth_mode == "linear_decay":
            step_base = self.field_distance_gate_m
            decay = self.field_distance_decay_per_frame
            gate = 0.0
            for step_idx in range(gate_lost_frames):
                step = step_base - (decay * step_idx)
                if step <= 0.0:
                    break
                gate += step
            gate = max(step_base, gate)
        else:
            growth = float(gate_lost_frames) ** self.field_distance_lost_exponent
            gate = self.field_distance_gate_m * growth

        if self.field_distance_gate_cap_m is not None:
            gate = min(gate, self.field_distance_gate_cap_m)
        return float(gate)

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
        if self._must_use_field_position_for_class(effective_class):
            previous_field_position_tuple = self._field_position_to_tuple(previous_field_position)
            new_field_position_tuple = self._field_position_to_tuple(new_field_position)
            if (
                previous_field_position_tuple is None
                or new_field_position_tuple is None
            ):
                return False
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
        motion_growth_frames = lost_frames
        if self.reassign_motion_growth_cap_frames is not None:
            motion_growth_frames = min(
                lost_frames,
                self.reassign_motion_growth_cap_frames,
            )
        samples = int(previous_state.get("movement_samples", 0))
        mean_step_distance = float(previous_state.get("mean_step_distance", 0.0))
        reserved_seed = bool(previous_state.get("reserved_seed", False))
        uses_field_position = self._use_field_position_for_class(
            effective_class,
            previous_field_position,
        )

        if not reserved_seed:
            max_lost_frames = self._max_lost_frames_for_class(effective_class)
            if max_lost_frames is not None and lost_frames > max_lost_frames:
                return False

        if uses_field_position:
            max_allowed_jump = max(
                self.reassign_min_field_distance_m,
                self._field_distance_gate_for_lost_frames(lost_frames),
            )
        else:
            max_allowed_jump = self.reassign_min_distance

        if reserved_seed:
            max_allowed_jump = max(
                max_allowed_jump,
                float(previous_state.get("reserved_seed_match_distance_m", 0.0)),
            )

        if not uses_field_position and samples >= self.reassign_min_samples:
            expected_jump = mean_step_distance * motion_growth_frames
            max_allowed_jump = max(
                max_allowed_jump,
                expected_jump * self.reassign_motion_factor,
            )

        # Gate estadístico por velocidad (distancia por frame):
        # bloquea saltos extremos respecto al histórico del propio track.
        if self.motion_std_gate_enabled and not uses_field_position:
            limits_per_frame = []

            track_stats_count = int(previous_state.get("step_per_frame_count", 0))
            if track_stats_count >= self.motion_std_min_samples:
                track_stats_mean = float(
                    previous_state.get("step_per_frame_mean", 0.0)
                )
                track_stats_m2 = float(previous_state.get("step_per_frame_m2", 0.0))
                limits_per_frame.append(
                    self._motion_limit_per_frame(
                        track_stats_count,
                        track_stats_mean,
                        track_stats_m2,
                    )
                )

            class_stats = self.class_motion_stats.get(effective_class, {})
            class_stats_count = int(class_stats.get("count", 0))
            if class_stats_count >= self.motion_std_min_samples:
                limits_per_frame.append(
                    self._motion_limit_per_frame(
                        class_stats_count,
                        float(class_stats.get("mean", 0.0)),
                        float(class_stats.get("m2", 0.0)),
                    )
                )

            if limits_per_frame:
                max_per_frame = min(limits_per_frame)
                stats_jump_limit = max_per_frame * motion_growth_frames
                # Allow statistical gate to be stricter than base min jump.
                # Keep only a small numerical floor to avoid over-constraining.
                stats_jump_limit = max(
                    stats_jump_limit,
                    self.motion_std_floor * motion_growth_frames,
                )
                max_allowed_jump = min(max_allowed_jump, stats_jump_limit)

        return step_distance <= max_allowed_jump

    def _bbox_distance_sq(
        self,
        bbox_a,
        bbox_b,
        class_name=None,
        field_position_a=None,
        field_position_b=None,
    ):
        if self._must_use_field_position_for_class(class_name):
            field_position_a = self._field_position_to_tuple(field_position_a)
            field_position_b = self._field_position_to_tuple(field_position_b)
            if field_position_a is None or field_position_b is None:
                return None
            ax, ay = field_position_a
            bx, by = field_position_b
            return (ax - bx) ** 2 + (ay - by) ** 2
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

    def _ball_state_is_active(self, ball_state, current_frame):
        if ball_state is None:
            return False
        if self.ball_max_reassign_lost_frames is None:
            return True
        lost_frames = current_frame - int(ball_state.get("last_frame", current_frame))
        return lost_frames <= self.ball_max_reassign_lost_frames

    def _predict_ball_center(self, ball_state, current_frame):
        if ball_state is None:
            return None
        previous_bbox = ball_state.get("bbox")
        if previous_bbox is None:
            return None
        last_center = self._bbox_center(previous_bbox)
        prev_center = ball_state.get("prev_center")
        prev_frame = ball_state.get("prev_frame")
        last_frame = int(ball_state.get("last_frame", current_frame))
        gap_frames = max(1, current_frame - last_frame)
        if prev_center is None or prev_frame is None:
            return last_center
        frame_delta = max(1, last_frame - int(prev_frame))
        vx = (last_center[0] - float(prev_center[0])) / frame_delta
        vy = (last_center[1] - float(prev_center[1])) / frame_delta
        return (
            last_center[0] + (vx * gap_frames),
            last_center[1] + (vy * gap_frames),
        )

    def _ball_prediction_gate_px(self, ball_state, current_frame):
        base_gate = float(self.ball_expected_position_gate_px)
        if ball_state is None:
            return base_gate
        last_frame = int(ball_state.get("last_frame", current_frame))
        gap_frames = max(1, current_frame - last_frame)
        mean_step = float(ball_state.get("step_per_frame_mean", 0.0))
        dynamic_gate = base_gate + (
            self.ball_expected_position_gate_growth_per_frame * max(0, gap_frames - 1)
        )
        if mean_step > 0.0:
            dynamic_gate = max(dynamic_gate, mean_step * 2.5 * gap_frames)
        return float(dynamic_gate)

    @staticmethod
    def _clamp_coordinate(value, lower_bound, upper_bound):
        return max(float(lower_bound), min(float(value), float(upper_bound)))

    def _ball_prediction_reference(self, ball_state, current_frame, frame_size=None):
        predicted_center = self._predict_ball_center(ball_state, current_frame)
        if predicted_center is None:
            return None, {}
        if frame_size is None:
            return predicted_center, {}

        frame_width = max(1.0, float(frame_size[0]))
        frame_height = max(1.0, float(frame_size[1]))
        max_x = frame_width - 1.0
        max_y = frame_height - 1.0
        px, py = predicted_center
        edge_constraints = {}

        if px < 0.0:
            edge_constraints["left"] = True
        elif px > max_x:
            edge_constraints["right"] = True

        if py < 0.0:
            edge_constraints["top"] = True
        elif py > max_y:
            edge_constraints["bottom"] = True

        if not edge_constraints:
            return predicted_center, {}

        return (
            self._clamp_coordinate(px, 0.0, max_x),
            self._clamp_coordinate(py, 0.0, max_y),
        ), edge_constraints

    def _matches_ball_edge_constraints(
        self,
        candidate_center,
        edge_constraints,
        frame_size,
        prediction_gate,
    ):
        if not edge_constraints or frame_size is None:
            return True

        frame_width = max(1.0, float(frame_size[0]))
        frame_height = max(1.0, float(frame_size[1]))
        edge_band = max(40.0, min(float(prediction_gate) * 0.5, 140.0))
        cx, cy = candidate_center

        if edge_constraints.get("left") and cx > edge_band:
            return False
        if edge_constraints.get("right") and cx < (frame_width - edge_band):
            return False
        if edge_constraints.get("top") and cy > edge_band:
            return False
        if edge_constraints.get("bottom") and cy < (frame_height - edge_band):
            return False

        return True

    def _is_ball_size_compatible(self, previous_state, new_bbox, confidence):
        if previous_state is None:
            return True

        prev_bbox = previous_state.get("bbox")
        if prev_bbox is None:
            return True

        last_frame = int(previous_state.get("last_frame", 0))
        new_size = self._ball_size_measure(new_bbox)
        prev_size = self._ball_size_measure(prev_bbox)
        if prev_size <= 0.0 or new_size <= 0.0:
            return True

        ratio_limit = float(self.ball_size_ratio_per_frame)
        ratio_limit = max(1.0, ratio_limit)
        size_ratio = max(new_size, prev_size) / max(min(new_size, prev_size), 1e-6)

        if size_ratio <= ratio_limit:
            return True

        size_count = int(previous_state.get("ball_size_count", 0))
        if size_count >= self.ball_size_min_samples:
            size_mean = float(previous_state.get("ball_size_mean", prev_size))
            size_m2 = float(previous_state.get("ball_size_m2", 0.0))
            size_variance = size_m2 / max(1, size_count - 1)
            size_std = max(
                float(np.sqrt(max(size_variance, 0.0))),
                self.ball_size_std_floor,
            )
            lower_bound = max(0.0, size_mean - (self.ball_size_std_factor * size_std))
            upper_bound = size_mean + (self.ball_size_std_factor * size_std)
            if lower_bound <= new_size <= upper_bound:
                return True

        if confidence >= self.ball_high_conf_override:
            relaxed_ratio_limit = ratio_limit * self.ball_expected_position_confidence_relax
            if size_ratio <= relaxed_ratio_limit:
                return True

        return False

    def _build_ball_track_payload(self, bbox, confidence, metadata):
        return {
            "bbox": bbox,
            "confidence": float(confidence),
            "team": metadata.get("team"),
            "distances": metadata.get("distances"),
            "shirt_color": metadata.get("shirt_color"),
            "bbox_size": metadata.get("bbox_size"),
            "field_position_m": None,
            "ground_point_image": None,
        }

    def _update_ball_state(self, previous_state, bbox, current_frame):
        previous_bbox = previous_state.get("bbox") if previous_state is not None else None
        prev_last_frame = (
            int(previous_state.get("last_frame", current_frame))
            if previous_state is not None
            else current_frame
        )
        frame_gap = max(1, current_frame - prev_last_frame)

        prev_samples = int(previous_state.get("movement_samples", 0)) if previous_state else 0
        prev_mean = float(previous_state.get("mean_step_distance", 0.0)) if previous_state else 0.0
        prev_step_pf_count = (
            int(previous_state.get("step_per_frame_count", 0)) if previous_state else 0
        )
        prev_step_pf_mean = (
            float(previous_state.get("step_per_frame_mean", 0.0)) if previous_state else 0.0
        )
        prev_step_pf_m2 = (
            float(previous_state.get("step_per_frame_m2", 0.0)) if previous_state else 0.0
        )

        step_distance = self._step_distance(
            previous_bbox,
            bbox,
            class_name="ball",
            previous_field_position=None,
            new_field_position=None,
        )
        if step_distance is not None:
            movement_samples = prev_samples + 1
            if prev_samples <= 0:
                mean_step_distance = step_distance
            else:
                mean_step_distance = (
                    (prev_mean * prev_samples) + step_distance
                ) / movement_samples
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
                "ball",
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
            movement_samples = prev_samples
            mean_step_distance = prev_mean
            step_per_frame_count = prev_step_pf_count
            step_per_frame_mean = prev_step_pf_mean
            step_per_frame_m2 = prev_step_pf_m2

        new_size = self._ball_size_measure(bbox)
        prev_size_count = int(previous_state.get("ball_size_count", 0)) if previous_state else 0
        prev_size_mean = float(previous_state.get("ball_size_mean", 0.0)) if previous_state else 0.0
        prev_size_m2 = float(previous_state.get("ball_size_m2", 0.0)) if previous_state else 0.0
        (
            ball_size_count,
            ball_size_mean,
            ball_size_m2,
        ) = self._update_running_stats(
            prev_size_count,
            prev_size_mean,
            prev_size_m2,
            new_size,
        )

        return {
            "bbox": bbox,
            "class_name": "ball",
            "last_frame": current_frame,
            "field_position": None,
            "team": None,
            "movement_samples": movement_samples,
            "mean_step_distance": mean_step_distance,
            "step_per_frame_count": step_per_frame_count,
            "step_per_frame_mean": step_per_frame_mean,
            "step_per_frame_m2": step_per_frame_m2,
            "prev_center": (
                self._bbox_center(previous_bbox) if previous_bbox is not None else None
            ),
            "prev_frame": prev_last_frame if previous_bbox is not None else None,
            "ball_size_count": ball_size_count,
            "ball_size_mean": ball_size_mean,
            "ball_size_m2": ball_size_m2,
        }

    def _select_ball_candidate(
        self,
        candidates,
        ball_state,
        current_frame,
        frame_size=None,
    ):
        reference_state = (
            ball_state if self._ball_state_is_active(ball_state, current_frame) else None
        )
        evaluated_candidates = []

        for candidate in candidates:
            bbox = candidate["bbox"]
            confidence = float(candidate["confidence"])
            if not self._is_ball_size_compatible(reference_state, bbox, confidence):
                continue

            expected_error = 0.0
            if reference_state is not None:
                predicted_center, edge_constraints = self._ball_prediction_reference(
                    reference_state,
                    current_frame,
                    frame_size=frame_size,
                )
                if predicted_center is not None:
                    candidate_center = self._bbox_center(bbox)
                    prediction_gate = self._ball_prediction_gate_px(
                        reference_state,
                        current_frame,
                    )
                    if confidence >= self.ball_high_conf_override:
                        prediction_gate *= self.ball_expected_position_confidence_relax
                    if not self._matches_ball_edge_constraints(
                        candidate_center,
                        edge_constraints,
                        frame_size,
                        prediction_gate,
                    ):
                        continue
                    expected_error = float(
                        ((candidate_center[0] - predicted_center[0]) ** 2 + (candidate_center[1] - predicted_center[1]) ** 2) ** 0.5
                    )
                    if expected_error > prediction_gate:
                        continue

            evaluated_candidates.append(
                (
                    expected_error,
                    -confidence,
                    0 if candidate["source"] == "tracked" else 1,
                    candidate,
                )
            )

        if evaluated_candidates:
            evaluated_candidates.sort(key=lambda item: item[:3])
            return evaluated_candidates[0][3]

        # Antes de ver el balón por primera vez, o tras demasiados frames perdidos,
        # permitimos redetección libre por confianza máxima.
        if reference_state is None and candidates:
            bootstrap_candidates = sorted(
                candidates,
                key=lambda candidate: (
                    -float(candidate["confidence"]),
                    0 if candidate["source"] == "tracked" else 1,
                ),
            )
            return bootstrap_candidates[0]

        return None

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

        if (
            candidate_state.get("reserved_seed")
            and candidate_class == "player"
            and detection_class in {"player", "goalkeeper"}
        ):
            if not self._is_motion_compatible(
                candidate_state,
                detection_bbox,
                current_frame,
                class_name=detection_class,
                new_field_position=detection_field_position,
            ):
                return None
            return detection_class

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
                0 if canonical_state[canonical_id].get("reserved_seed") else 1,
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
            and not state.get("reserved_seed", False)
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

            teams_of_detected_objects = self.team_detector.detect_teams(detections, show_kmeans)
            teams_labels = [dicc["team"] for dicc in teams_of_detected_objects]
            class_labels = [dicc["class"] for dicc in teams_of_detected_objects]
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

            tracks_detection = self.tracker.update_with_detections(
                detections_sv,
                teams_labels,
                class_labels,
            )
            
            for key in tracks.keys():
                tracks[key].append({})

            used_canonical_ids_in_frame = set()
            sorted_tracked_detections = sorted(
                list(tracks_detection),
                key=lambda item: float(item[2]),
                reverse=True,
            )
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
                    "step_per_frame_count": step_per_frame_count,
                    "step_per_frame_mean": step_per_frame_mean,
                    "step_per_frame_m2": step_per_frame_m2,
                    "reserved_seed": False,
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
                    ball_candidates.append(
                        {
                            "bbox": bbox,
                            "confidence": confidence,
                            "metadata": metadata,
                            "source": "tracked",
                        }
                    )
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
                            class_name,
                            detected_team,
                            bbox,
                            field_position,
                            n_frame,
                        ) is None:
                            canonical_id = None

                if canonical_id is None:
                    output_class_name = class_name
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
                for bbox, score, cid in zip(xyxy, conf, cls):
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
                                "ground_point_image": None,
                            },
                            "source": "raw",
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
                ball_state = self._update_ball_state(
                    ball_state if self._ball_state_is_active(ball_state, n_frame) else None,
                    selected_ball["bbox"],
                    n_frame,
                )
        
        return tracks
