from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .common import PitchGeometry
from .geometry import project_image_points
from .pitch_layout import (
    PNLCALIB_AUX_KEYPOINT_WORLD_COORDS_M,
    PNLCALIB_LINE_WORLD_COORDS_M,
    PNLCALIB_MAIN_KEYPOINT_WORLD_COORDS_M,
    PNLCALIB_TEMPLATE_FIELD_LENGTH_M,
    PNLCALIB_TEMPLATE_FIELD_WIDTH_M,
)


class ProjectionQualityAnalyzer:
    def __init__(
        self,
        geometry: PitchGeometry,
        min_visible_keypoints: int = 6,
        min_visible_lines: int = 1,
        
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
    ) -> None:
        self.geometry = geometry
        self.min_visible_keypoints = max(0, int(min_visible_keypoints))
        self.min_visible_lines = max(0, int(min_visible_lines))
        self.validation_score_threshold = float(validation_score_threshold)
        self.validation_geometry_fit_min = float(validation_geometry_fit_min)
        self.validation_support_quality_min = float(validation_support_quality_min)
        self.validation_coverage_quality_min = float(validation_coverage_quality_min)
        self.validation_reprojection_error_threshold_px = (
            float(validation_reprojection_error_threshold_px)
        )
        self.validation_homography_condition_number_threshold = (
            float(validation_homography_condition_number_threshold)
        )
        self.validation_keypoint_world_error_threshold_m = (
            float(validation_keypoint_world_error_threshold_m)
        )
        self.validation_line_world_error_threshold_m = (
            float(validation_line_world_error_threshold_m)
        )
        self.validation_target_visible_keypoints = max(
            self.min_visible_keypoints,
            int(validation_target_visible_keypoints),
        )
        self.validation_target_visible_lines = max(
            self.min_visible_lines,
            int(validation_target_visible_lines),
        )
        self.validation_target_image_point_hull_area_ratio = (
            float(validation_target_image_point_hull_area_ratio)
        )
        self.validation_target_image_x_span_ratio = float(validation_target_image_x_span_ratio)
        self.validation_target_image_y_span_ratio = float(validation_target_image_y_span_ratio)
        self.validation_target_field_coverage_ratio = float(validation_target_field_coverage_ratio)

    @staticmethod
    def _project_or_none(
        points_xy: np.ndarray,
        homography_image_to_field: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if homography_image_to_field is None or len(points_xy) == 0:
            return None
        try:
            return project_image_points(points_xy, homography_image_to_field).astype(np.float32)
        except Exception:  # pragma: no cover - guard defensivo
            return None

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
        return safe_threshold / (safe_threshold + max(float(value), 0.0))

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
        return points[np.all(np.isfinite(points), axis=1)]

    @staticmethod
    def _convex_hull_area(points_xy: Sequence[Sequence[float]]) -> float:
        points = ProjectionQualityAnalyzer._filter_finite_points(points_xy)
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
            projected_point = self._project_or_none(
                np.asarray([[float(point["x"]), float(point["y"])]], dtype=np.float32),
                homography_image_to_field,
            )
            if projected_point is None:
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
            projected_points = self._project_or_none(
                np.asarray(
                    [
                        [float(line["x_1"]), float(line["y_1"])],
                        [float(line["x_2"]), float(line["y_2"])],
                    ],
                    dtype=np.float32,
                ),
                homography_image_to_field,
            )
            if projected_points is None or len(projected_points) != 2:
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
    ) -> tuple[Optional[float], Dict[str, Any], list[str]]:

        reprojection_error = getattr(estimate, "reprojection_error", None)
        reprojection_score = self._inverse_error_score(
            reprojection_error,
            self.validation_reprojection_error_threshold_px,
        )
        
        homography_image_to_field = estimate.homography_image_to_field
        if homography_image_to_field is None:
            return None, {}, ["no_homography"]
        
        homography = np.asarray(homography_image_to_field, dtype=np.float64)
        if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
            return None, {}, ["invalid_homography_matrix"]
        
        normalized_homography = homography.copy()
        if abs(float(normalized_homography[2, 2])) < 1e-9:
            return None, {}, ["homography_scale_zero"]
        normalized_homography /= normalized_homography[2, 2]
        condition_number = float(np.linalg.cond(normalized_homography))

        if not np.isfinite(condition_number):
            return None, {}, ["invalid_homography_condition_number"]

        if condition_number > self.validation_homography_condition_number_threshold:
            return None, {}, ["ill_conditioned_homography"]
    
        log_condition_threshold = max(
            np.log10(self.validation_homography_condition_number_threshold),
            1e-6,
        )
        condition_score = self._clamp01(
            1.0 - (np.log10(max(condition_number, 1.0)) / log_condition_threshold)
        )

        keypoint_world_error_m = self._keypoint_world_alignment_error_m(
            estimate,
            homography_image_to_field,
        )
        
        keypoint_alignment_score = self._inverse_error_score(
            keypoint_world_error_m,
            self.validation_keypoint_world_error_threshold_m,
        )

        line_world_error_m = self._line_world_alignment_error_m(
            estimate,
            homography_image_to_field,
        )
        
        line_alignment_score = self._inverse_error_score(
            line_world_error_m,
            self.validation_line_world_error_threshold_m,
        )

        geometry_fit = self._weighted_average(
            [
                (reprojection_score, 0.35),
                (condition_score, 0.15),
                (keypoint_alignment_score, 0.30),
                (line_alignment_score, 0.20),
            ]
        )
        
        details = {
            "reprojection_error": reprojection_error,
            "reprojection_score": reprojection_score,
            "homography_condition_number": condition_number,
            "condition_score": condition_score,
            "keypoint_world_error_m": keypoint_world_error_m,
            "keypoint_alignment_score": keypoint_alignment_score,
            "line_world_error_m": line_world_error_m,
            "line_alignment_score": line_alignment_score,
            "geometry_fit": geometry_fit,
        }
        
        return geometry_fit, details, []

    def _support_quality_score(
        self,
        estimate,
    ) -> tuple[Optional[float], Dict[str, Any]]:
        keypoint_count = estimate.visible_keypoints_count
        line_count = estimate.visible_lines_count
        keypoint_count_score = self._range_score(
            keypoint_count,
            self.min_visible_keypoints,
            self.validation_target_visible_keypoints,
        )
        line_count_score = self._range_score(
            line_count,
            self.min_visible_lines,
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

    def evaluate_projection_quality(
        self,
        estimate,
        frame_shape_hw: Tuple[int, int],
        keypoint_threshold: float,
        line_threshold: float,
        attempt_index: int,
    ) -> Dict[str, Any]:
        geometry_fit, geometry_details, rejection_reasons = self._geometry_fit_score(estimate)
        support_quality, support_details = self._support_quality_score(estimate)
        coverage_quality, coverage_details = self._coverage_quality_score(
            estimate,
            frame_shape_hw=frame_shape_hw,
        )

        if estimate.homography_image_to_field is None:
            rejection_reasons.append("no_homography")
        if estimate.visible_keypoints_count < self.min_visible_keypoints:
            rejection_reasons.append("insufficient_keypoints")
        if estimate.visible_lines_count < self.min_visible_lines:
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


__all__ = ["ProjectionQualityAnalyzer"]
