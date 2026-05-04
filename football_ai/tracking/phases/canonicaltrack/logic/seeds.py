from __future__ import annotations

import numpy as np


class CanonicalSeedMixin:
    def _reserved_penalty_spot_field_positions(self):
        if (
            not self.reserve_penalty_spot_seed_players
            or not self.use_field_positions
            or self.field_projector is None
        ):
            return []

        geometry = self.field_projector.geometry
        center_y_m = float(geometry.center_y_m)
        penalty_mark_distance_m = float(geometry.penalty_mark_distance_m)
        field_length_m = float(geometry.field_length_m)
        return [
            (penalty_mark_distance_m, center_y_m),
            (field_length_m - penalty_mark_distance_m, center_y_m),
        ]

    def _build_reserved_penalty_spot_state(self, field_position):
        return {
            "bbox": None,
            "class_name": "goalkeeper",
            "last_frame": 0,
            "team": None,
            "field_position": tuple(float(v) for v in field_position),
            "movement_samples": 0,
            "mean_step_distance": 0.0,
            "step_per_frame_count": 0,
            "step_per_frame_mean": 0.0,
            "step_per_frame_m2": 0.0,
            "reserved_seed": True,
            "special_penalty_seed": True,
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

    def _build_reserved_seed_track_payload(self, state, reference_packet):
        field_position = self._field_position_to_tuple(state.get("field_position"))
        projected_ground_point = None
        original_ground_point = None
        frame_shape_original = None
        reference_clean = dict((reference_packet or {}).get("clean") or {})
        if reference_clean:
            homography_image_to_field = np.asarray(
                reference_clean.get("homography_image_to_field_3x3") or [],
                dtype=np.float64,
            )
            projected_ground_point = self._project_homography_point(
                field_position,
                homography_image_to_field if homography_image_to_field.shape == (3, 3) else None,
            )
            frame_shape_original = (
                int((reference_packet or {}).get("image_height", 0) or 0),
                int((reference_packet or {}).get("image_width", 0) or 0),
            )
            original_ground_point = projected_ground_point

        bbox = self._seed_bbox_from_ground_point(original_ground_point, frame_shape_original)
        bbox_size = float(self._bbox_area(bbox)) if bbox is not None else 0.0
        return {
            "bbox": bbox,
            "confidence": 0.0,
            "team": None,
            "distances": None,
            "shirt_color": None,
            "bbox_size": bbox_size,
            "class_tracker": "goalkeeper",
            "class_td": "goalkeeper",
            "class_yolo": "goalkeeper",
            "source_raw_tracker_id": None,
            "canonical_assignment_mode": "reserved_seed",
            "canonical_relinked": False,
            "forced_absorption": False,
            "field_position_m": list(field_position) if field_position is not None else None,
            "ground_point_image": (
                [float(projected_ground_point[0]), float(projected_ground_point[1])]
                if projected_ground_point is not None
                else None
            ),
            "reserved_seed": True,
            "special_penalty_seed": bool(state.get("special_penalty_seed", False)),
            "synthetic_seed": True,
        }

    def _initialize_reserved_penalty_spot_players(self, canonical_state):
        for next_free_id, field_position in zip(
            self.special_seed_canonical_ids,
            self._reserved_penalty_spot_field_positions(),
        ):
            if next_free_id in canonical_state:
                continue
            canonical_state[next_free_id] = self._build_reserved_penalty_spot_state(
                field_position
            )
