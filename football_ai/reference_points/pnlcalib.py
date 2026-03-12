from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from experiments.reference_points.pnlcalib_reference_points import (
    PitchGeometry,
    StreamingPnLCalibEstimator,
    ensure_pnlcalib_repo,
    ensure_pnlcalib_weights,
    get_default_pnlcalib_repo_path,
    get_default_pnlcalib_weight_paths,
    load_pnlcalib_runtime,
    project_image_points,
    resize_frame,
)


FIELD_POSITION_CLASSES = frozenset({"player", "goalkeeper"})
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

    def _ground_points_from_bboxes(self, bboxes_xyxy: np.ndarray) -> np.ndarray:
        boxes = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        heights = np.maximum(y2 - y1, 1.0)

        x_coords = 0.5 * (x1 + x2)
        y_coords = y2 - (self.bottom_offset_ratio * heights)
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
        ground_points_original = self._ground_points_from_bboxes(bboxes_xyxy)
        ground_points_projected = self._scale_points(
            ground_points_original,
            source_shape_hw=original_shape_hw,
            target_shape_hw=projected_shape_hw,
        )

        homography_image_to_field: Optional[np.ndarray] = None
        estimation_mode = "error"
        reprojection_error: Optional[float] = None
        try:
            estimate = self.estimator.estimate(projected_frame)
            homography_image_to_field = estimate.homography_image_to_field
            estimation_mode = str(getattr(estimate, "estimation_mode", "ok"))
            reprojection_error = getattr(estimate, "reprojection_error", None)
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

        return FieldProjectionResult(
            frame_shape_original=original_shape_hw,
            frame_shape_projected=projected_shape_hw,
            homography_image_to_field=homography_image_to_field,
            field_positions_m=field_positions_m,
            ground_points_image_original=ground_points_original,
            ground_points_image_projected=ground_points_projected,
            estimation_mode=estimation_mode,
            reprojection_error=reprojection_error,
        )
