from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from football_ai.reference_points.pnlcalib_runtime import (
    PitchGeometry,
    StreamingPnLCalibEstimator,
    _to_template_homography,
    estimate_pnlcalib_frame,
    ensure_pnlcalib_repo,
    ensure_pnlcalib_weights,
    get_default_pnlcalib_repo_path,
    get_default_pnlcalib_weight_paths,
    load_pnlcalib_runtime,
    project_image_points,
    resize_frame,
)


FIELD_POSITION_CLASSES = frozenset({"player", "goalkeeper", "referee", "ball"})
GROUND_POINT_BOTTOM_OFFSET_BY_CLASS = {
    "player": 0.04,
    "goalkeeper": 0.04,
    "referee": 0.04,
    "ball": 0.0,
}
PNLCALIB_TEMPLATE_FIELD_LENGTH_M = 105.0
PNLCALIB_TEMPLATE_FIELD_WIDTH_M = 68.0
PNLCALIB_MAIN_KEYPOINT_WORLD_COORDS_M = [
    (0.0, 0.0),
    (52.5, 0.0),
    (105.0, 0.0),
    (0.0, 13.84),
    (16.5, 13.84),
    (88.5, 13.84),
    (105.0, 13.84),
    (0.0, 24.84),
    (5.5, 24.84),
    (99.5, 24.84),
    (105.0, 24.84),
    (0.0, 30.34),
    (0.0, 30.34),
    (105.0, 30.34),
    (105.0, 30.34),
    (0.0, 37.66),
    (0.0, 37.66),
    (105.0, 37.66),
    (105.0, 37.66),
    (0.0, 43.16),
    (5.5, 43.16),
    (99.5, 43.16),
    (105.0, 43.16),
    (0.0, 54.16),
    (16.5, 54.16),
    (88.5, 54.16),
    (105.0, 54.16),
    (0.0, 68.0),
    (52.5, 68.0),
    (105.0, 68.0),
    (16.5, 26.68),
    (52.5, 24.85),
    (88.5, 26.68),
    (16.5, 41.31),
    (52.5, 43.15),
    (88.5, 41.31),
    (19.99, 32.29),
    (43.68, 31.53),
    (61.31, 31.53),
    (85.0, 32.29),
    (19.99, 35.7),
    (43.68, 36.46),
    (61.31, 36.46),
    (85.0, 35.7),
    (11.0, 34.0),
    (16.5, 34.0),
    (20.15, 34.0),
    (46.03, 27.53),
    (58.97, 27.53),
    (43.35, 34.0),
    (52.5, 34.0),
    (61.5, 34.0),
    (46.03, 40.47),
    (58.97, 40.47),
    (84.85, 34.0),
    (88.5, 34.0),
    (94.0, 34.0),
]
PNLCALIB_AUX_KEYPOINT_WORLD_COORDS_M = [
    (5.5, 0.0),
    (16.5, 0.0),
    (88.5, 0.0),
    (99.5, 0.0),
    (5.5, 13.84),
    (99.5, 13.84),
    (16.5, 24.84),
    (88.5, 24.84),
    (16.5, 43.16),
    (88.5, 43.16),
    (5.5, 54.16),
    (99.5, 54.16),
    (5.5, 68.0),
    (16.5, 68.0),
    (88.5, 68.0),
    (99.5, 68.0),
]
PNLCALIB_LINE_WORLD_COORDS_M = {
    1: ((0.0, 54.16, 0.0), (16.5, 54.16, 0.0)),
    2: ((16.5, 13.84, 0.0), (16.5, 54.16, 0.0)),
    3: ((16.5, 13.84, 0.0), (0.0, 13.84, 0.0)),
    4: ((88.5, 54.16, 0.0), (105.0, 54.16, 0.0)),
    5: ((88.5, 13.84, 0.0), (88.5, 54.16, 0.0)),
    6: ((88.5, 13.84, 0.0), (105.0, 13.84, 0.0)),
    7: ((0.0, 37.66, -2.44), (0.0, 30.34, -2.44)),
    8: ((0.0, 37.66, 0.0), (0.0, 37.66, -2.44)),
    9: ((0.0, 30.34, 0.0), (0.0, 30.34, -2.44)),
    10: ((105.0, 37.66, -2.44), (105.0, 30.34, -2.44)),
    11: ((105.0, 30.34, 0.0), (105.0, 30.34, -2.44)),
    12: ((105.0, 37.66, 0.0), (105.0, 37.66, -2.44)),
    13: ((52.5, 0.0, 0.0), (52.5, 68.0, 0.0)),
    14: ((0.0, 68.0, 0.0), (105.0, 68.0, 0.0)),
    15: ((0.0, 0.0, 0.0), (0.0, 68.0, 0.0)),
    16: ((105.0, 0.0, 0.0), (105.0, 68.0, 0.0)),
    17: ((0.0, 0.0, 0.0), (105.0, 0.0, 0.0)),
    18: ((0.0, 43.16, 0.0), (5.5, 43.16, 0.0)),
    19: ((5.5, 43.16, 0.0), (5.5, 24.84, 0.0)),
    20: ((5.5, 24.84, 0.0), (0.0, 24.84, 0.0)),
    21: ((99.5, 43.16, 0.0), (105.0, 43.16, 0.0)),
    22: ((99.5, 43.16, 0.0), (99.5, 24.84, 0.0)),
    23: ((99.5, 24.84, 0.0), (105.0, 24.84, 0.0)),
}
logger = logging.getLogger(__name__)


@dataclass
class FieldProjectionResult:
    frame_shape_original: Tuple[int, int]
    frame_shape_projected: Tuple[int, int]
    homography_image_to_field: Optional[np.ndarray]
    field_positions_m: np.ndarray
    ground_points_image_original: np.ndarray
    ground_points_image_projected: np.ndarray
    estimation_mode: str
    reprojection_error: Optional[float]
    visible_keypoints_count: int = 0
    visible_lines_count: int = 0
    keypoint_threshold_used: Optional[float] = None
    line_threshold_used: Optional[float] = None
    quality_diagnostics: Optional[Dict[str, Any]] = None
    homography_quality_score: Optional[float] = None
    homography_quality_status: str = "rejected"
    field_positions_usable_for_tracking: bool = True

    @property
    def has_homography(self) -> bool:
        return self.homography_image_to_field is not None


class PnLCalibFieldProjector:
    def __init__(
        self,
        project_root: Optional[Path] = None,
        field_length_m: float = 106.0,
        field_width_m: float = 68.0,
        max_width: int = 1280,
        bottom_offset_ratio: float = 0.04,
        keypoint_threshold: float = 0.3434,
        line_threshold: float = 0.7867,
        pnl_refine: bool = True,
        temporal_blend: float = 0.20,
        pixels_per_meter: int = 8,
        device: Optional[str] = None,
        adaptive_thresholds_enabled: bool = True,
        adaptive_max_attempts: int = 4,
        adaptive_keypoint_threshold_step: float = 0.05,
        adaptive_line_threshold_step: float = 0.10,
        adaptive_keypoint_threshold_min: float = 0.15,
        adaptive_line_threshold_min: float = 0.25,
        adaptive_min_visible_keypoints: int = 6,
        adaptive_min_visible_lines: int = 1,
        validation_score_threshold: float = 0.55,
        validation_geometry_fit_min: float = 0.35,
        validation_support_quality_min: float = 0.35,
        validation_coverage_quality_min: float = 0.20,
        validation_reprojection_error_threshold_px: float = 8.0,
        validation_homography_condition_number_threshold: float = 1.0e8,
        validation_keypoint_world_error_threshold_m: float = 3.0,
        validation_line_world_error_threshold_m: float = 4.0,
        validation_target_visible_keypoints: int = 14,
        validation_target_visible_lines: int = 5,
        validation_target_image_point_hull_area_ratio: float = 0.18,
        validation_target_image_x_span_ratio: float = 0.55,
        validation_target_image_y_span_ratio: float = 0.40,
        validation_target_field_coverage_ratio: float = 0.20,
        tracking_field_positions_require_attempt0: bool = True,
        repo_path: Optional[Path] = None,
        weights_kp_path: Optional[Path] = None,
        weights_line_path: Optional[Path] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve() if project_root is not None else None
        self.geometry = PitchGeometry(
            field_length_m=field_length_m,
            field_width_m=field_width_m,
        )
        self.max_width = int(max_width)
        self.bottom_offset_ratio = float(bottom_offset_ratio)
        self.keypoint_threshold = float(keypoint_threshold)
        self.line_threshold = float(line_threshold)
        self.adaptive_thresholds_enabled = bool(adaptive_thresholds_enabled)
        self.adaptive_max_attempts = max(1, int(adaptive_max_attempts))
        self.adaptive_keypoint_threshold_step = float(
            max(0.0, adaptive_keypoint_threshold_step)
        )
        self.adaptive_line_threshold_step = float(max(0.0, adaptive_line_threshold_step))
        self.adaptive_keypoint_threshold_min = float(adaptive_keypoint_threshold_min)
        self.adaptive_line_threshold_min = float(adaptive_line_threshold_min)
        self.adaptive_min_visible_keypoints = max(
            0, int(adaptive_min_visible_keypoints)
        )
        self.adaptive_min_visible_lines = max(0, int(adaptive_min_visible_lines))
        self.validation_score_threshold = float(max(0.0, validation_score_threshold))
        self.validation_geometry_fit_min = float(max(0.0, validation_geometry_fit_min))
        self.validation_support_quality_min = float(max(0.0, validation_support_quality_min))
        self.validation_coverage_quality_min = float(max(0.0, validation_coverage_quality_min))
        self.validation_reprojection_error_threshold_px = float(
            max(1e-6, validation_reprojection_error_threshold_px)
        )
        self.validation_homography_condition_number_threshold = float(
            max(1.0, validation_homography_condition_number_threshold)
        )
        self.validation_keypoint_world_error_threshold_m = float(
            max(1e-6, validation_keypoint_world_error_threshold_m)
        )
        self.validation_line_world_error_threshold_m = float(
            max(1e-6, validation_line_world_error_threshold_m)
        )
        self.validation_target_visible_keypoints = max(
            self.adaptive_min_visible_keypoints,
            int(validation_target_visible_keypoints),
        )
        self.validation_target_visible_lines = max(
            self.adaptive_min_visible_lines,
            int(validation_target_visible_lines),
        )
        self.validation_target_image_point_hull_area_ratio = float(
            max(1e-6, validation_target_image_point_hull_area_ratio)
        )
        self.validation_target_image_x_span_ratio = float(
            max(1e-6, validation_target_image_x_span_ratio)
        )
        self.validation_target_image_y_span_ratio = float(
            max(1e-6, validation_target_image_y_span_ratio)
        )
        self.validation_target_field_coverage_ratio = float(
            max(1e-6, validation_target_field_coverage_ratio)
        )
        self.tracking_field_positions_require_attempt0 = bool(
            tracking_field_positions_require_attempt0
        )

        resolved_repo_path = (
            Path(repo_path)
            if repo_path is not None
            else get_default_pnlcalib_repo_path(self.project_root)
        )
        default_weights_kp_path, default_weights_line_path = get_default_pnlcalib_weight_paths(
            self.project_root
        )
        resolved_weights_kp_path = (
            Path(weights_kp_path) if weights_kp_path is not None else default_weights_kp_path
        )
        resolved_weights_line_path = (
            Path(weights_line_path) if weights_line_path is not None else default_weights_line_path
        )

        repo_dir = ensure_pnlcalib_repo(resolved_repo_path)
        weights_kp_path, weights_line_path = ensure_pnlcalib_weights(
            resolved_weights_kp_path,
            resolved_weights_line_path,
        )
        runtime = load_pnlcalib_runtime(
            repo_dir=repo_dir,
            weights_kp_path=weights_kp_path,
            weights_line_path=weights_line_path,
            device=device,
        )
        self.estimator = StreamingPnLCalibEstimator(
            runtime=runtime,
            geometry=self.geometry,
            keypoint_threshold=keypoint_threshold,
            line_threshold=line_threshold,
            pnl_refine=pnl_refine,
            pixels_per_meter=pixels_per_meter,
            temporal_blend=temporal_blend,
        )

    def _adaptive_threshold_schedule(self) -> list[tuple[float, float]]:
        schedule = [(self.keypoint_threshold, self.line_threshold)]
        if not self.adaptive_thresholds_enabled:
            return schedule

        for attempt_idx in range(1, self.adaptive_max_attempts):
            candidate_keypoint_threshold = max(
                self.adaptive_keypoint_threshold_min,
                self.keypoint_threshold
                - (attempt_idx * self.adaptive_keypoint_threshold_step),
            )
            candidate_line_threshold = max(
                self.adaptive_line_threshold_min,
                self.line_threshold - (attempt_idx * self.adaptive_line_threshold_step),
            )
            candidate_pair = (
                float(candidate_keypoint_threshold),
                float(candidate_line_threshold),
            )
            if candidate_pair not in schedule:
                schedule.append(candidate_pair)
        return schedule

    @staticmethod
    def _project_with_homography(
        ground_points_projected: np.ndarray,
        homography_image_to_field: Optional[np.ndarray],
    ) -> tuple[np.ndarray, Optional[str]]:
        projected_positions = np.full(
            (len(ground_points_projected), 2),
            np.nan,
            dtype=np.float32,
        )
        if homography_image_to_field is None or len(ground_points_projected) == 0:
            return projected_positions, None
        try:
            projected_positions = project_image_points(
                ground_points_projected,
                homography_image_to_field,
            ).astype(np.float32)
            return projected_positions, None
        except np.linalg.LinAlgError:
            return projected_positions, "singular_homography"
        except Exception as exc:  # pragma: no cover - guard defensivo
            return projected_positions, f"projection_error:{type(exc).__name__}"

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _weighted_average(items: Sequence[tuple[Optional[float], float]]) -> Optional[float]:
        valid_items = [
            (float(value), float(weight))
            for value, weight in items
            if value is not None and np.isfinite(float(value)) and weight > 0.0
        ]
        if not valid_items:
            return None
        total_weight = sum(weight for _, weight in valid_items)
        if total_weight <= 0.0:
            return None
        return float(
            sum(value * weight for value, weight in valid_items) / total_weight
        )

    @classmethod
    def _inverse_error_score(
        cls,
        value: Optional[float],
        threshold: float,
    ) -> Optional[float]:
        if value is None or not np.isfinite(float(value)):
            return None
        safe_threshold = max(float(threshold), 1e-6)
        return cls._clamp01(safe_threshold / (safe_threshold + max(float(value), 0.0)))

    @classmethod
    def _range_score(
        cls,
        value: float,
        minimum: float,
        target: float,
    ) -> float:
        value = float(value)
        minimum = float(minimum)
        target = float(target)
        if target <= minimum:
            return 1.0 if value >= target else 0.0
        return cls._clamp01((value - minimum) / (target - minimum))

    @staticmethod
    def _filter_finite_points(points_xy: Sequence[Sequence[float]]) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        finite_mask = np.all(np.isfinite(points), axis=1)
        return points[finite_mask]

    @staticmethod
    def _convex_hull_area(points_xy: Sequence[Sequence[float]]) -> float:
        points = PnLCalibFieldProjector._filter_finite_points(points_xy)
        if len(points) < 3:
            return 0.0
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        return float(abs(cv2.contourArea(hull)))

    @staticmethod
    def _span_ratio(
        values: np.ndarray,
        extent: float,
    ) -> float:
        if values.size == 0 or not np.any(np.isfinite(values)):
            return 0.0
        safe_extent = max(float(extent), 1e-6)
        span = float(np.nanmax(values) - np.nanmin(values))
        return float(max(0.0, span) / safe_extent)

    def _scale_world_point_to_geometry(
        self,
        point_xy: tuple[float, float],
    ) -> tuple[float, float]:
        scale_x = float(self.geometry.field_length_m) / PNLCALIB_TEMPLATE_FIELD_LENGTH_M
        scale_y = float(self.geometry.field_width_m) / PNLCALIB_TEMPLATE_FIELD_WIDTH_M
        return (
            float(point_xy[0]) * scale_x,
            float(point_xy[1]) * scale_y,
        )

    def _keypoint_world_point_m(
        self,
        keypoint_id: int,
    ) -> Optional[tuple[float, float]]:
        if 1 <= keypoint_id <= len(PNLCALIB_MAIN_KEYPOINT_WORLD_COORDS_M):
            return self._scale_world_point_to_geometry(
                PNLCALIB_MAIN_KEYPOINT_WORLD_COORDS_M[keypoint_id - 1]
            )
        aux_idx = keypoint_id - (len(PNLCALIB_MAIN_KEYPOINT_WORLD_COORDS_M) + 1)
        if 0 <= aux_idx < len(PNLCALIB_AUX_KEYPOINT_WORLD_COORDS_M):
            return self._scale_world_point_to_geometry(
                PNLCALIB_AUX_KEYPOINT_WORLD_COORDS_M[aux_idx]
            )
        return None

    def _line_world_segment_m(
        self,
        line_id: int,
    ) -> Optional[tuple[tuple[float, float], tuple[float, float], bool]]:
        segment = PNLCALIB_LINE_WORLD_COORDS_M.get(int(line_id))
        if segment is None:
            return None
        (x1, y1, z1), (x2, y2, z2) = segment
        point_1 = self._scale_world_point_to_geometry((x1, y1))
        point_2 = self._scale_world_point_to_geometry((x2, y2))
        return point_1, point_2, bool(abs(z1) < 1e-6 and abs(z2) < 1e-6)

    def _collect_keypoint_image_points(
        self,
        estimate,
    ) -> np.ndarray:
        points = [
            (float(point["x"]), float(point["y"]))
            for point in getattr(estimate, "keypoints_dict", {}).values()
            if "x" in point and "y" in point
        ]
        return self._filter_finite_points(points)

    def _collect_line_image_points(
        self,
        estimate,
    ) -> np.ndarray:
        points = []
        for line in getattr(estimate, "lines_dict", {}).values():
            if all(key in line for key in ("x_1", "y_1", "x_2", "y_2")):
                points.append((float(line["x_1"]), float(line["y_1"])))
                points.append((float(line["x_2"]), float(line["y_2"])))
        return self._filter_finite_points(points)

    def _keypoint_world_alignment_error_m(
        self,
        estimate,
        homography_image_to_field: Optional[np.ndarray],
    ) -> Optional[float]:
        if homography_image_to_field is None:
            return None
        projected_errors = []
        for keypoint_id, point in getattr(estimate, "keypoints_dict", {}).items():
            if "x" not in point or "y" not in point:
                continue
            world_point = self._keypoint_world_point_m(int(keypoint_id))
            if world_point is None:
                continue
            projected_point, projection_error = self._project_with_homography(
                np.asarray([[float(point["x"]), float(point["y"])]], dtype=np.float32),
                homography_image_to_field,
            )
            if projection_error is not None or len(projected_point) == 0:
                continue
            projected_xy = projected_point[0]
            if not np.all(np.isfinite(projected_xy)):
                continue
            projected_errors.append(
                float(np.linalg.norm(projected_xy - np.asarray(world_point, dtype=np.float32)))
            )
        if not projected_errors:
            return None
        return float(np.median(np.asarray(projected_errors, dtype=np.float32)))

    def _line_world_alignment_error_m(
        self,
        estimate,
        homography_image_to_field: Optional[np.ndarray],
    ) -> Optional[float]:
        if homography_image_to_field is None:
            return None
        line_errors = []
        for line_id, line in getattr(estimate, "lines_dict", {}).items():
            world_segment = self._line_world_segment_m(int(line_id))
            if world_segment is None:
                continue
            world_p1, world_p2, is_planar = world_segment
            if not is_planar:
                continue
            if not all(key in line for key in ("x_1", "y_1", "x_2", "y_2")):
                continue
            projected_points, projection_error = self._project_with_homography(
                np.asarray(
                    [
                        [float(line["x_1"]), float(line["y_1"])],
                        [float(line["x_2"]), float(line["y_2"])],
                    ],
                    dtype=np.float32,
                ),
                homography_image_to_field,
            )
            if projection_error is not None or len(projected_points) != 2:
                continue
            if not np.all(np.isfinite(projected_points)):
                continue
            world_segment_array = np.asarray([world_p1, world_p2], dtype=np.float32)
            direct_error = float(
                np.mean(np.linalg.norm(projected_points - world_segment_array, axis=1))
            )
            flipped_error = float(
                np.mean(
                    np.linalg.norm(
                        projected_points - world_segment_array[::-1],
                        axis=1,
                    )
                )
            )
            line_errors.append(min(direct_error, flipped_error))
        if not line_errors:
            return None
        return float(np.median(np.asarray(line_errors, dtype=np.float32)))

    def _line_family_diversity_score(
        self,
        estimate,
    ) -> float:
        families = set()
        for line_id in getattr(estimate, "lines_dict", {}).keys():
            world_segment = self._line_world_segment_m(int(line_id))
            if world_segment is None:
                continue
            point_1, point_2, is_planar = world_segment
            if not is_planar:
                families.add("goal_structure")
                continue
            delta_x = abs(float(point_2[0]) - float(point_1[0]))
            delta_y = abs(float(point_2[1]) - float(point_1[1]))
            families.add("horizontal" if delta_x >= delta_y else "vertical")
        return self._clamp01(len(families) / 3.0)

    def _keypoint_family_diversity_score(
        self,
        estimate,
    ) -> tuple[float, int, int]:
        x_bins = set()
        y_bins = set()
        for keypoint_id in getattr(estimate, "keypoints_dict", {}).keys():
            world_point = self._keypoint_world_point_m(int(keypoint_id))
            if world_point is None:
                continue
            x_ratio = float(world_point[0]) / max(float(self.geometry.field_length_m), 1e-6)
            y_ratio = float(world_point[1]) / max(float(self.geometry.field_width_m), 1e-6)
            x_bins.add(min(2, max(0, int(x_ratio * 3.0))))
            y_bins.add(min(2, max(0, int(y_ratio * 3.0))))
        score = self._weighted_average(
            [
                (len(x_bins) / 3.0, 0.5),
                (len(y_bins) / 3.0, 0.5),
            ]
        )
        return float(score or 0.0), len(x_bins), len(y_bins)

    @staticmethod
    def _keypoint_confidence_score(
        estimate,
    ) -> Optional[float]:
        confidences = [
            float(point["p"])
            for point in getattr(estimate, "keypoints_dict", {}).values()
            if "p" in point and np.isfinite(float(point["p"]))
        ]
        if not confidences:
            return None
        return float(np.mean(np.asarray(confidences, dtype=np.float32)))

    @staticmethod
    def _line_confidence_score(
        estimate,
    ) -> Optional[float]:
        confidences = []
        for line in getattr(estimate, "lines_dict", {}).values():
            sample = []
            if "p_1" in line and np.isfinite(float(line["p_1"])):
                sample.append(float(line["p_1"]))
            if "p_2" in line and np.isfinite(float(line["p_2"])):
                sample.append(float(line["p_2"]))
            if sample:
                confidences.append(float(np.mean(np.asarray(sample, dtype=np.float32))))
        if not confidences:
            return None
        return float(np.mean(np.asarray(confidences, dtype=np.float32)))

    def _geometry_fit_score(
        self,
        estimate,
        homography_image_to_field: Optional[np.ndarray],
    ) -> tuple[Optional[float], Dict[str, Any], list[str]]:
        details: Dict[str, Any] = {}
        rejection_reasons: list[str] = []
        if homography_image_to_field is None:
            return None, details, ["no_homography"]

        reprojection_error = getattr(estimate, "reprojection_error", None)
        details["reprojection_error"] = reprojection_error
        reprojection_score = self._inverse_error_score(
            reprojection_error,
            self.validation_reprojection_error_threshold_px,
        )
        details["reprojection_score"] = reprojection_score

        homography = np.asarray(homography_image_to_field, dtype=np.float64)
        if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
            return None, details, ["invalid_homography_matrix"]
        normalized_homography = homography.copy()
        if abs(float(normalized_homography[2, 2])) < 1e-9:
            return None, details, ["homography_scale_zero"]
        normalized_homography /= normalized_homography[2, 2]
        condition_number = float(np.linalg.cond(normalized_homography))
        details["homography_condition_number"] = condition_number
        if not np.isfinite(condition_number):
            return None, details, ["invalid_homography_condition_number"]
        if condition_number > self.validation_homography_condition_number_threshold:
            rejection_reasons.append("ill_conditioned_homography")
        log_condition_threshold = max(
            np.log10(self.validation_homography_condition_number_threshold),
            1e-6,
        )
        condition_score = self._clamp01(
            1.0 - (np.log10(max(condition_number, 1.0)) / log_condition_threshold)
        )
        details["condition_score"] = condition_score

        keypoint_world_error_m = self._keypoint_world_alignment_error_m(
            estimate,
            homography_image_to_field,
        )
        details["keypoint_world_error_m"] = keypoint_world_error_m
        keypoint_alignment_score = self._inverse_error_score(
            keypoint_world_error_m,
            self.validation_keypoint_world_error_threshold_m,
        )
        details["keypoint_alignment_score"] = keypoint_alignment_score

        line_world_error_m = self._line_world_alignment_error_m(
            estimate,
            homography_image_to_field,
        )
        details["line_world_error_m"] = line_world_error_m
        line_alignment_score = self._inverse_error_score(
            line_world_error_m,
            self.validation_line_world_error_threshold_m,
        )
        details["line_alignment_score"] = line_alignment_score

        geometry_fit = self._weighted_average(
            [
                (reprojection_score, 0.35),
                (condition_score, 0.15),
                (keypoint_alignment_score, 0.30),
                (line_alignment_score, 0.20),
            ]
        )
        details["geometry_fit"] = geometry_fit
        return geometry_fit, details, rejection_reasons

    def _support_quality_score(
        self,
        estimate,
    ) -> tuple[Optional[float], Dict[str, Any]]:
        keypoint_count = int(getattr(estimate, "visible_keypoints_count", 0))
        line_count = int(getattr(estimate, "visible_lines_count", 0))
        keypoint_count_score = self._range_score(
            keypoint_count,
            self.adaptive_min_visible_keypoints,
            self.validation_target_visible_keypoints,
        )
        line_count_score = self._range_score(
            line_count,
            self.adaptive_min_visible_lines,
            self.validation_target_visible_lines,
        )
        keypoint_confidence_score = self._keypoint_confidence_score(estimate)
        line_confidence_score = self._line_confidence_score(estimate)
        keypoint_family_diversity_score, x_family_count, y_family_count = (
            self._keypoint_family_diversity_score(estimate)
        )
        line_family_diversity_score = self._line_family_diversity_score(estimate)

        support_quality = self._weighted_average(
            [
                (keypoint_count_score, 0.25),
                (line_count_score, 0.20),
                (keypoint_confidence_score, 0.15),
                (line_confidence_score, 0.10),
                (keypoint_family_diversity_score, 0.15),
                (line_family_diversity_score, 0.15),
            ]
        )
        return support_quality, {
            "keypoint_count_score": keypoint_count_score,
            "line_count_score": line_count_score,
            "keypoint_confidence_score": keypoint_confidence_score,
            "line_confidence_score": line_confidence_score,
            "keypoint_family_diversity_score": keypoint_family_diversity_score,
            "line_family_diversity_score": line_family_diversity_score,
            "keypoint_family_x_count": int(x_family_count),
            "keypoint_family_y_count": int(y_family_count),
            "support_quality": support_quality,
        }

    def _field_semantic_coverage_points(
        self,
        estimate,
    ) -> np.ndarray:
        field_points = []
        for keypoint_id in getattr(estimate, "keypoints_dict", {}).keys():
            world_point = self._keypoint_world_point_m(int(keypoint_id))
            if world_point is not None:
                field_points.append(world_point)
        for line_id in getattr(estimate, "lines_dict", {}).keys():
            world_segment = self._line_world_segment_m(int(line_id))
            if world_segment is None:
                continue
            point_1, point_2, is_planar = world_segment
            if not is_planar:
                continue
            field_points.extend([point_1, point_2])
        return self._filter_finite_points(field_points)

    def _coverage_quality_score(
        self,
        estimate,
        frame_shape_hw: Tuple[int, int],
    ) -> tuple[Optional[float], Dict[str, Any]]:
        frame_height, frame_width = frame_shape_hw
        keypoint_image_points = self._collect_keypoint_image_points(estimate)
        line_image_points = self._collect_line_image_points(estimate)
        image_points = self._filter_finite_points(
            np.vstack([keypoint_image_points, line_image_points])
            if len(keypoint_image_points) > 0 or len(line_image_points) > 0
            else np.zeros((0, 2), dtype=np.float32)
        )
        image_hull_area_ratio = 0.0
        image_x_span_ratio = 0.0
        image_y_span_ratio = 0.0
        if len(image_points) > 0:
            image_hull_area_ratio = float(
                self._convex_hull_area(image_points)
                / max(float(frame_width * frame_height), 1e-6)
            )
            image_x_span_ratio = self._span_ratio(image_points[:, 0], float(frame_width))
            image_y_span_ratio = self._span_ratio(image_points[:, 1], float(frame_height))

        field_points = self._field_semantic_coverage_points(estimate)
        field_hull_area_ratio = 0.0
        if len(field_points) > 0:
            field_hull_area_ratio = float(
                self._convex_hull_area(field_points)
                / max(
                    float(self.geometry.field_length_m * self.geometry.field_width_m),
                    1e-6,
                )
            )

        image_hull_score = self._clamp01(
            image_hull_area_ratio / self.validation_target_image_point_hull_area_ratio
        )
        image_x_span_score = self._clamp01(
            image_x_span_ratio / self.validation_target_image_x_span_ratio
        )
        image_y_span_score = self._clamp01(
            image_y_span_ratio / self.validation_target_image_y_span_ratio
        )
        field_hull_score = self._clamp01(
            field_hull_area_ratio / self.validation_target_field_coverage_ratio
        )

        coverage_quality = self._weighted_average(
            [
                (image_hull_score, 0.40),
                (image_x_span_score, 0.20),
                (image_y_span_score, 0.20),
                (field_hull_score, 0.20),
            ]
        )
        return coverage_quality, {
            "image_point_hull_area_ratio": image_hull_area_ratio,
            "image_x_span_ratio": image_x_span_ratio,
            "image_y_span_ratio": image_y_span_ratio,
            "field_semantic_hull_area_ratio": field_hull_area_ratio,
            "image_hull_score": image_hull_score,
            "image_x_span_score": image_x_span_score,
            "image_y_span_score": image_y_span_score,
            "field_hull_score": field_hull_score,
            "coverage_quality": coverage_quality,
        }

    @staticmethod
    def _rejection_type(rejection_reasons: Sequence[str]) -> Optional[str]:
        reasons = set(rejection_reasons)
        if not reasons:
            return None
        if "no_homography" in reasons:
            return "no_solution"
        if (
            "ill_conditioned_homography" in reasons
            or "invalid_homography_matrix" in reasons
            or "homography_scale_zero" in reasons
            or "invalid_homography_condition_number" in reasons
            or "low_geometry_fit" in reasons
        ):
            return "bad_geometry"
        if (
            "insufficient_keypoints" in reasons
            or "insufficient_lines" in reasons
            or "low_support_quality" in reasons
            or "low_coverage_quality" in reasons
        ):
            return "low_support"
        return "low_score"

    def _evaluate_projection_quality(
        self,
        estimate,
        frame_shape_hw: Tuple[int, int],
        keypoint_threshold: float,
        line_threshold: float,
        attempt_index: int,
    ) -> Dict[str, Any]:
        geometry_fit, geometry_details, rejection_reasons = self._geometry_fit_score(
            estimate,
            estimate.homography_image_to_field,
        )
        support_quality, support_details = self._support_quality_score(estimate)
        coverage_quality, coverage_details = self._coverage_quality_score(
            estimate,
            frame_shape_hw=frame_shape_hw,
        )

        if estimate.homography_image_to_field is None:
            rejection_reasons.append("no_homography")
        if estimate.visible_keypoints_count < self.adaptive_min_visible_keypoints:
            rejection_reasons.append("insufficient_keypoints")
        if estimate.visible_lines_count < self.adaptive_min_visible_lines:
            rejection_reasons.append("insufficient_lines")

        has_homography = estimate.homography_image_to_field is not None
        if not has_homography:
            geometry_fit = 0.0
            quality_score = 0.0
        else:
            quality_score = self._weighted_average(
                [
                    (geometry_fit, 0.45),
                    (support_quality, 0.30),
                    (coverage_quality, 0.25),
                ]
            )
        if geometry_fit is None or geometry_fit < self.validation_geometry_fit_min:
            rejection_reasons.append("low_geometry_fit")
        if support_quality is None or support_quality < self.validation_support_quality_min:
            rejection_reasons.append("low_support_quality")
        if coverage_quality is None or coverage_quality < self.validation_coverage_quality_min:
            rejection_reasons.append("low_coverage_quality")
        if quality_score is None or quality_score < self.validation_score_threshold:
            rejection_reasons.append("low_quality_score")

        unique_rejection_reasons = list(dict.fromkeys(rejection_reasons))
        rejection_type = self._rejection_type(unique_rejection_reasons)
        quality_status = (
            "good"
            if has_homography and len(unique_rejection_reasons) == 0
            else "rejected"
        )

        return {
            "attempt_index": int(attempt_index),
            "accepted": quality_status == "good",
            "quality_status": quality_status,
            "rejection_type": rejection_type,
            "rejection_reasons": unique_rejection_reasons,
            "keypoint_threshold": float(keypoint_threshold),
            "line_threshold": float(line_threshold),
            "visible_keypoints_count": int(estimate.visible_keypoints_count),
            "visible_lines_count": int(estimate.visible_lines_count),
            "estimation_mode": str(getattr(estimate, "estimation_mode", "unknown")),
            "reprojection_error": getattr(estimate, "reprojection_error", None),
            "quality_score": quality_score,
            "geometry_fit": geometry_fit,
            "support_quality": support_quality,
            "coverage_quality": coverage_quality,
            "geometry_details": geometry_details,
            "support_details": support_details,
            "coverage_details": coverage_details,
        }

    def _estimate_with_adaptive_thresholds(
        self,
        projected_frame: np.ndarray,
        ground_points_projected: np.ndarray,
        class_names: Optional[Sequence[Optional[str]]],
    ):
        del ground_points_projected, class_names
        attempt_records = []

        def _safe_metric(diagnostics: Dict[str, Any], key: str) -> float:
            value = diagnostics.get(key)
            if value is None:
                return float("-inf")
            try:
                value = float(value)
            except (TypeError, ValueError):
                return float("-inf")
            if not np.isfinite(value):
                return float("-inf")
            return value

        for attempt_index, (
            keypoint_threshold,
            line_threshold,
        ) in enumerate(self._adaptive_threshold_schedule()):
            raw_estimate = estimate_pnlcalib_frame(
                projected_frame,
                runtime=self.estimator.runtime,
                geometry=self.geometry,
                keypoint_threshold=keypoint_threshold,
                line_threshold=line_threshold,
                pnl_refine=self.estimator.pnl_refine,
                pixels_per_meter=self.estimator.pixels_per_meter,
            )
            diagnostics = self._evaluate_projection_quality(
                raw_estimate,
                frame_shape_hw=projected_frame.shape[:2],
                keypoint_threshold=keypoint_threshold,
                line_threshold=line_threshold,
                attempt_index=attempt_index,
            )
            attempt_records.append((raw_estimate, diagnostics))

        if not attempt_records:
            self.estimator.previous_homography_image_to_field = None
            return None, {
                "accepted": False,
                "quality_status": "rejected",
                "quality_score": None,
                "rejection_type": "no_solution",
                "selected_attempt": None,
                "smoothed_attempt": None,
                "attempts": [],
            }

        selected_estimate, selected_diagnostics = max(
            attempt_records,
            key=lambda item: (
                1 if item[1].get("accepted", False) else 0,
                _safe_metric(item[1], "quality_score"),
                _safe_metric(item[1], "geometry_fit"),
                _safe_metric(item[1], "support_quality"),
                _safe_metric(item[1], "coverage_quality"),
                -int(item[1].get("attempt_index", 0)),
            ),
        )
        attempt_diagnostics = [diagnostics for _, diagnostics in attempt_records]

        if not bool(selected_diagnostics.get("accepted", False)):
            self.estimator.previous_homography_image_to_field = None
            rejected_estimate = replace(
                selected_estimate,
                homography_image_to_field=None,
                homography_image_to_template=None,
                estimation_mode=f"{selected_estimate.estimation_mode}+quality_rejected",
            )
            return rejected_estimate, {
                "accepted": False,
                "quality_status": "rejected",
                "quality_score": selected_diagnostics.get("quality_score"),
                "rejection_type": selected_diagnostics.get("rejection_type"),
                "selected_attempt": selected_diagnostics,
                "smoothed_attempt": None,
                "attempts": attempt_diagnostics,
            }

        smoothed_diagnostics = None
        previous_homography = self.estimator.previous_homography_image_to_field
        if (
            previous_homography is not None
            and self.estimator.temporal_blend > 0.0
            and selected_estimate.homography_image_to_field is not None
        ):
            previous_h = previous_homography.copy()
            current_h = selected_estimate.homography_image_to_field.copy()
            previous_h /= previous_h[2, 2]
            current_h /= current_h[2, 2]
            blended = (
                self.estimator.temporal_blend * previous_h
                + (1.0 - self.estimator.temporal_blend) * current_h
            )
            blended /= blended[2, 2]
            smoothed_estimate = replace(
                selected_estimate,
                homography_image_to_field=blended,
                homography_image_to_template=_to_template_homography(
                    blended,
                    pixels_per_meter=self.estimator.pixels_per_meter,
                ),
                estimation_mode=f"{selected_estimate.estimation_mode}+smoothed",
            )
            smoothed_diagnostics = self._evaluate_projection_quality(
                smoothed_estimate,
                frame_shape_hw=projected_frame.shape[:2],
                keypoint_threshold=selected_diagnostics["keypoint_threshold"],
                line_threshold=selected_diagnostics["line_threshold"],
                attempt_index=selected_diagnostics["attempt_index"],
            )
            if bool(smoothed_diagnostics.get("accepted", False)) and float(
                smoothed_diagnostics.get("quality_score", 0.0) or 0.0
            ) >= float(selected_diagnostics.get("quality_score", 0.0) or 0.0):
                self.estimator.previous_homography_image_to_field = blended.copy()
                return smoothed_estimate, {
                    "accepted": True,
                    "quality_status": "good",
                    "quality_score": smoothed_diagnostics.get("quality_score"),
                    "rejection_type": None,
                    "selected_attempt": selected_diagnostics,
                    "smoothed_attempt": smoothed_diagnostics,
                    "attempts": attempt_diagnostics,
                }

        if selected_estimate.homography_image_to_field is not None:
            self.estimator.previous_homography_image_to_field = (
                selected_estimate.homography_image_to_field.copy()
            )
        return selected_estimate, {
            "accepted": True,
            "quality_status": "good",
            "quality_score": selected_diagnostics.get("quality_score"),
            "rejection_type": None,
            "selected_attempt": selected_diagnostics,
            "smoothed_attempt": smoothed_diagnostics,
            "attempts": attempt_diagnostics,
        }

    def _field_positions_usable_for_tracking(
        self,
        quality_diagnostics: Optional[Dict[str, Any]],
        homography_image_to_field: Optional[np.ndarray],
    ) -> bool:
        if homography_image_to_field is None:
            return False
        if not isinstance(quality_diagnostics, dict):
            return True
        return str(quality_diagnostics.get("quality_status", "rejected")) == "good"

    def _ground_points_from_bboxes(
        self,
        bboxes_xyxy: np.ndarray,
        class_names: Optional[Sequence[Optional[str]]] = None,
    ) -> np.ndarray:
        boxes = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        heights = np.maximum(y2 - y1, 1.0)

        if class_names is None:
            offset_ratios = np.full(len(boxes), self.bottom_offset_ratio, dtype=np.float32)
        else:
            normalized_class_names = [
                str(class_name).strip().lower() if class_name is not None else ""
                for class_name in class_names
            ]
            if len(normalized_class_names) < len(boxes):
                normalized_class_names.extend([""] * (len(boxes) - len(normalized_class_names)))
            offset_ratios = np.asarray(
                [
                    GROUND_POINT_BOTTOM_OFFSET_BY_CLASS.get(
                        class_name,
                        self.bottom_offset_ratio,
                    )
                    for class_name in normalized_class_names[: len(boxes)]
                ],
                dtype=np.float32,
            )

        x_coords = 0.5 * (x1 + x2)
        y_coords = y2 - (offset_ratios * heights)
        return np.column_stack([x_coords, y_coords]).astype(np.float32)

    @staticmethod
    def _scale_points(
        points_xy: np.ndarray,
        source_shape_hw: Tuple[int, int],
        target_shape_hw: Tuple[int, int],
    ) -> np.ndarray:
        source_height, source_width = source_shape_hw
        target_height, target_width = target_shape_hw
        points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        scale_x = target_width / float(source_width)
        scale_y = target_height / float(source_height)
        scaled = points.copy()
        scaled[:, 0] *= scale_x
        scaled[:, 1] *= scale_y
        return scaled

    def project_detections(
        self,
        frame_bgr: np.ndarray,
        bboxes_xyxy: np.ndarray,
        class_names: Optional[Sequence[Optional[str]]] = None,
    ) -> FieldProjectionResult:
        original_frame = np.asarray(frame_bgr)
        projected_frame = resize_frame(original_frame, max_width=self.max_width)

        original_shape_hw = original_frame.shape[:2]
        projected_shape_hw = projected_frame.shape[:2]
        ground_points_original = self._ground_points_from_bboxes(
            bboxes_xyxy,
            class_names=class_names,
        )
        ground_points_projected = self._scale_points(
            ground_points_original,
            source_shape_hw=original_shape_hw,
            target_shape_hw=projected_shape_hw,
        )

        homography_image_to_field: Optional[np.ndarray] = None
        estimation_mode = "error"
        reprojection_error: Optional[float] = None
        visible_keypoints_count = 0
        visible_lines_count = 0
        keypoint_threshold_used: Optional[float] = None
        line_threshold_used: Optional[float] = None
        quality_diagnostics: Optional[Dict[str, Any]] = None
        homography_quality_score: Optional[float] = None
        homography_quality_status = "rejected"
        try:
            estimate, quality_diagnostics = self._estimate_with_adaptive_thresholds(
                projected_frame,
                ground_points_projected,
                class_names,
            )
            if estimate is not None:
                homography_image_to_field = estimate.homography_image_to_field
                estimation_mode = str(getattr(estimate, "estimation_mode", "ok"))
                reprojection_error = getattr(estimate, "reprojection_error", None)
                visible_keypoints_count = int(getattr(estimate, "visible_keypoints_count", 0))
                visible_lines_count = int(getattr(estimate, "visible_lines_count", 0))
            selected_attempt = (
                quality_diagnostics.get("selected_attempt")
                if isinstance(quality_diagnostics, dict)
                else None
            )
            final_attempt = (
                quality_diagnostics.get("smoothed_attempt")
                if isinstance(quality_diagnostics, dict)
                and isinstance(quality_diagnostics.get("smoothed_attempt"), dict)
                and bool(quality_diagnostics["smoothed_attempt"].get("accepted", False))
                else selected_attempt
            )
            if isinstance(final_attempt, dict):
                keypoint_threshold_used = float(final_attempt["keypoint_threshold"])
                line_threshold_used = float(final_attempt["line_threshold"])
                homography_quality_score = (
                    float(final_attempt["quality_score"])
                    if final_attempt.get("quality_score") is not None
                    else None
                )
                homography_quality_status = str(
                    final_attempt.get("quality_status", homography_quality_status)
                )
            elif isinstance(quality_diagnostics, dict):
                homography_quality_score = (
                    float(quality_diagnostics["quality_score"])
                    if quality_diagnostics.get("quality_score") is not None
                    else None
                )
                homography_quality_status = str(
                    quality_diagnostics.get("quality_status", homography_quality_status)
                )
        except np.linalg.LinAlgError as exc:
            logger.warning(
                "PnLCalib devolvió homografía singular; se omite field_position_m en este frame. Error: %s",
                exc,
            )
        except Exception as exc:  # pragma: no cover - fallback defensivo
            logger.warning(
                "PnLCalib falló en este frame; se omite field_position_m. Error: %s",
                exc,
            )

        field_positions_m = np.full((len(ground_points_projected), 2), np.nan, dtype=np.float32)
        if homography_image_to_field is not None and len(ground_points_projected) > 0:
            field_positions_m = project_image_points(
                ground_points_projected,
                homography_image_to_field,
            ).astype(np.float32)

        if class_names is not None:
            class_names = list(class_names)
            for idx, class_name in enumerate(class_names):
                if class_name not in FIELD_POSITION_CLASSES and idx < len(field_positions_m):
                    field_positions_m[idx, :] = np.nan

        field_positions_usable_for_tracking = self._field_positions_usable_for_tracking(
            quality_diagnostics=quality_diagnostics,
            homography_image_to_field=homography_image_to_field,
        )
        if isinstance(quality_diagnostics, dict):
            quality_diagnostics["field_positions_usable_for_tracking"] = bool(
                field_positions_usable_for_tracking
            )
            quality_diagnostics["homography_quality_score"] = homography_quality_score
            quality_diagnostics["homography_quality_status"] = homography_quality_status
            quality_diagnostics["rejection_type"] = quality_diagnostics.get(
                "rejection_type",
                final_attempt.get("rejection_type") if isinstance(final_attempt, dict) else None,
            )
            quality_diagnostics["tracking_quality_status"] = (
                "good_for_tracking"
                if field_positions_usable_for_tracking
                else "not_reliable_for_tracking"
            )

        return FieldProjectionResult(
            frame_shape_original=original_shape_hw,
            frame_shape_projected=projected_shape_hw,
            homography_image_to_field=homography_image_to_field,
            field_positions_m=field_positions_m,
            ground_points_image_original=ground_points_original,
            ground_points_image_projected=ground_points_projected,
            estimation_mode=estimation_mode,
            reprojection_error=reprojection_error,
            visible_keypoints_count=visible_keypoints_count,
            visible_lines_count=visible_lines_count,
            keypoint_threshold_used=keypoint_threshold_used,
            line_threshold_used=line_threshold_used,
            quality_diagnostics=quality_diagnostics,
            homography_quality_score=homography_quality_score,
            homography_quality_status=homography_quality_status,
            field_positions_usable_for_tracking=field_positions_usable_for_tracking,
        )
