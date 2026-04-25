from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
from types import SimpleNamespace

import numpy as np

from .common import PitchGeometry, resize_frame
from .estimation import estimate_pnlcalib_frame
from .geometry import _to_template_homography, project_image_points
from .pitch_layout import FIELD_POSITION_CLASSES, GROUND_POINT_BOTTOM_OFFSET_BY_CLASS
from .quality import ProjectionQualityAnalyzer
from .runtime_loader import (
    ensure_pnlcalib_repo,
    ensure_pnlcalib_weights,
    get_default_pnlcalib_repo_path,
    get_default_pnlcalib_weight_paths,
    load_pnlcalib_runtime,
)


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
    @staticmethod
    def _safe_metric(diagnostics: Dict[str, Any], key: str) -> float:
        value = diagnostics.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("-inf")
        return value if np.isfinite(value) else float("-inf")

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
        del tracking_field_positions_require_attempt0
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

        self.quality_analyzer = ProjectionQualityAnalyzer(
            geometry=self.geometry,
            adaptive_min_visible_keypoints=self.adaptive_min_visible_keypoints,
            adaptive_min_visible_lines=self.adaptive_min_visible_lines,
            validation_score_threshold=self.validation_score_threshold,
            validation_geometry_fit_min=self.validation_geometry_fit_min,
            validation_support_quality_min=self.validation_support_quality_min,
            validation_coverage_quality_min=self.validation_coverage_quality_min,
            validation_reprojection_error_threshold_px=self.validation_reprojection_error_threshold_px,
            validation_homography_condition_number_threshold=self.validation_homography_condition_number_threshold,
            validation_keypoint_world_error_threshold_m=self.validation_keypoint_world_error_threshold_m,
            validation_line_world_error_threshold_m=self.validation_line_world_error_threshold_m,
            validation_target_visible_keypoints=self.validation_target_visible_keypoints,
            validation_target_visible_lines=self.validation_target_visible_lines,
            validation_target_image_point_hull_area_ratio=self.validation_target_image_point_hull_area_ratio,
            validation_target_image_x_span_ratio=self.validation_target_image_x_span_ratio,
            validation_target_image_y_span_ratio=self.validation_target_image_y_span_ratio,
            validation_target_field_coverage_ratio=self.validation_target_field_coverage_ratio,
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
        self.estimator = SimpleNamespace(
            runtime=runtime,
            geometry=self.geometry,
            keypoint_threshold=keypoint_threshold,
            line_threshold=line_threshold,
            pnl_refine=pnl_refine,
            pixels_per_meter=pixels_per_meter,
            temporal_blend=temporal_blend,
            previous_homography_image_to_field=None,
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

    def _estimate_with_adaptive_thresholds(self, projected_frame: np.ndarray):
        attempt_records = []

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
            diagnostics = self.quality_analyzer.evaluate_projection_quality(
                raw_estimate,
                frame_shape_hw=projected_frame.shape[:2],
                keypoint_threshold=keypoint_threshold,
                line_threshold=line_threshold,
                attempt_index=attempt_index,
            )
            attempt_records.append((raw_estimate, diagnostics))

        selected_estimate, selected_diagnostics = max(
            attempt_records,
            key=lambda item: (
                1 if item[1].get("accepted", False) else 0,
                self._safe_metric(item[1], "quality_score"),
                self._safe_metric(item[1], "geometry_fit"),
                self._safe_metric(item[1], "support_quality"),
                self._safe_metric(item[1], "coverage_quality"),
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
            smoothed_diagnostics = self.quality_analyzer.evaluate_projection_quality(
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

    @staticmethod
    def _field_positions_usable_for_tracking(
        quality_diagnostics: Optional[Dict[str, Any]],
        homography_image_to_field: Optional[np.ndarray],
    ) -> bool:
        if homography_image_to_field is None:
            return False
        if not isinstance(quality_diagnostics, dict):
            return True
        return str(quality_diagnostics.get("quality_status", "rejected")) == "good"

    @staticmethod
    def _final_attempt_from_quality(
        quality_diagnostics: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(quality_diagnostics, dict):
            return None
        smoothed_attempt = quality_diagnostics.get("smoothed_attempt")
        if isinstance(smoothed_attempt, dict) and bool(smoothed_attempt.get("accepted", False)):
            return smoothed_attempt
        selected_attempt = quality_diagnostics.get("selected_attempt")
        return selected_attempt if isinstance(selected_attempt, dict) else None

    @classmethod
    def _quality_summary(
        cls,
        quality_diagnostics: Optional[Dict[str, Any]],
    ) -> tuple[Optional[Dict[str, Any]], Optional[float], str]:
        final_attempt = cls._final_attempt_from_quality(quality_diagnostics)
        if isinstance(final_attempt, dict):
            quality_score = final_attempt.get("quality_score")
            return (
                final_attempt,
                float(quality_score) if quality_score is not None else None,
                str(final_attempt.get("quality_status", "rejected")),
            )
        if not isinstance(quality_diagnostics, dict):
            return None, None, "rejected"
        quality_score = quality_diagnostics.get("quality_score")
        return (
            None,
            float(quality_score) if quality_score is not None else None,
            str(quality_diagnostics.get("quality_status", "rejected")),
        )

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
        result, _ = self.project_detections_with_metadata(
            frame_bgr,
            bboxes_xyxy,
            class_names=class_names,
        )
        return result

    def project_detections_with_metadata(
        self,
        frame_bgr: np.ndarray,
        bboxes_xyxy: np.ndarray,
        class_names: Optional[Sequence[Optional[str]]] = None,
    ) -> tuple[FieldProjectionResult, Dict[str, Any]]:
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
        final_attempt = None
        estimate = None
        try:
            estimate, quality_diagnostics = self._estimate_with_adaptive_thresholds(
                projected_frame,
            )
            homography_image_to_field = estimate.homography_image_to_field
            estimation_mode = str(getattr(estimate, "estimation_mode", "ok"))
            reprojection_error = getattr(estimate, "reprojection_error", None)
            visible_keypoints_count = int(getattr(estimate, "visible_keypoints_count", 0))
            visible_lines_count = int(getattr(estimate, "visible_lines_count", 0))
            final_attempt, homography_quality_score, homography_quality_status = (
                self._quality_summary(quality_diagnostics)
            )
            if final_attempt is not None:
                keypoint_threshold_used = float(final_attempt["keypoint_threshold"])
                line_threshold_used = float(final_attempt["line_threshold"])
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

        field_positions_all_m = np.full(
            (len(ground_points_projected), 2),
            np.nan,
            dtype=np.float32,
        )
        if homography_image_to_field is not None and len(ground_points_projected) > 0:
            field_positions_all_m = project_image_points(
                ground_points_projected,
                homography_image_to_field,
            ).astype(np.float32)

        field_positions_m = field_positions_all_m.copy()
        if class_names is not None:
            invalid_indexes = [
                idx
                for idx, class_name in enumerate(class_names)
                if idx < len(field_positions_m) and class_name not in FIELD_POSITION_CLASSES
            ]
            if invalid_indexes:
                field_positions_m[invalid_indexes, :] = np.nan

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

        result = FieldProjectionResult(
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
        return result, {
            "estimate": estimate,
            "projected_frame_shape": projected_shape_hw,
            "quality_diagnostics": quality_diagnostics,
            "field_positions_all_m": field_positions_all_m,
            "final_attempt": final_attempt,
        }


__all__ = ["FieldProjectionResult", "PnLCalibFieldProjector"]
