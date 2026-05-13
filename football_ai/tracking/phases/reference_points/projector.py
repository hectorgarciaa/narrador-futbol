from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from football_ai.core import (
    IDENTITY_HOMOGRAPHY_3X3,
    PHASE_REFERENCE_POINTS,
    make_phase_packet,
)

from .common import PitchGeometry, resize_frame
from .estimation import prepare_frame_and_forward, estimate_pnlcalib_from_cached_outputs
from .geometry import project_image_points
from .pitch_layout import GROUND_POINT_BOTTOM_OFFSET_BY_CLASS
from .quality import ProjectionQualityAnalyzer
from .runtime_loader import load_pnlcalib_runtime


logger = logging.getLogger(__name__)


def build_reference_points_packet_without_homography(
    detector_packet,
    *,
    execution_mode="runtime",
):
    detector_clean = detector_packet["clean"]
    num_detections = detector_clean["num_detections"]
    zeros_2d = [[0.0, 0.0] for _ in range(num_detections)]
    collect_debug = str(execution_mode).strip().lower() == "debug"
    return make_phase_packet(
        phase_name=PHASE_REFERENCE_POINTS,
        frame_index=detector_packet["frame_index"],
        frame_time_ms=detector_packet["frame_time_ms"],
        image_width=detector_packet["image_width"],
        image_height=detector_packet["image_height"],
        clean={
            "num_detections": num_detections,
            "det_id": list(detector_clean["det_id"]),
            "bbox_xyxy": [list(bbox) for bbox in detector_clean["bbox_xyxy"]],
            "confidence": list(detector_clean["confidence"]),
            "class_name": list(detector_clean["class_name"]),
            "homography_valid": False,
            "homography_image_to_field_3x3": [row[:] for row in IDENTITY_HOMOGRAPHY_3X3],
            "field_positions_m": [point[:] for point in zeros_2d],
            "ground_points_image_original": [point[:] for point in zeros_2d],
            "field_positions_usable_for_tracking": False,
        },
        trace=(
            {
                "keypoints": [],
                "lines": [],
                "attempts": [],
                "diagnostics": {
                    "selected_attempt_index": 0,
                    "rejection_type": "",
                    "rejection_reasons": [],
                    "ground_points_image_projected": [point[:] for point in zeros_2d],
                    "estimation_mode": "",
                    "quality_status": "",
                    "quality_score": 0.0,
                    "reprojection_error_px": 0.0,
                    "visible_keypoints_count": 0,
                    "visible_lines_count": 0,
                    "keypoint_threshold_used": 0.0,
                    "line_threshold_used": 0.0,
                    "attempt_count": 0,
                },
            }
            if collect_debug
            else {}
        ),
    )


class PnLCalibFieldProjector:
    def __init__(
        self,
        project_root: Path | None = None,
        max_width: int = 1280,
        bottom_offset_ratio: float = 0.04,
        keypoint_threshold: float = 0.3434,
        line_threshold: float = 0.7867,
        adaptive_thresholds_enabled: bool = True,
        adaptive_max_attempts: int = 4,
        adaptive_keypoint_threshold_step: float = 0.05,
        adaptive_line_threshold_step: float = 0.10,
        adaptive_keypoint_threshold_min: float = 0.15,
        adaptive_line_threshold_min: float = 0.25,
        pnl_refine: bool = True,
        temporal_blend: float = 0.20,
        device: str | None = None,
        projection_quality_analyzer_conf: dict | None = None,
    ) -> None:
        project_root = Path(project_root).resolve() if project_root is not None else None
        
        self.geometry = PitchGeometry()
        self.quality_analyzer = ProjectionQualityAnalyzer(
            geometry=self.geometry,
            **dict(projection_quality_analyzer_conf or {}),
        )
        
        self.max_width = int(max_width)
        self.bottom_offset_ratio = float(bottom_offset_ratio)
        
        self.keypoint_threshold = float(keypoint_threshold)
        self.line_threshold = float(line_threshold)
        
        self.adaptive_thresholds_enabled = bool(adaptive_thresholds_enabled)
        self.adaptive_max_attempts = max(1, int(adaptive_max_attempts))
        self.adaptive_keypoint_threshold_step = max(0.0, float(adaptive_keypoint_threshold_step))
        self.adaptive_line_threshold_step = max(0.0, float(adaptive_line_threshold_step))
        self.adaptive_keypoint_threshold_min = float(adaptive_keypoint_threshold_min)
        self.adaptive_line_threshold_min = float(adaptive_line_threshold_min)
        
        self.pnl_refine = bool(pnl_refine)
        self.temporal_blend = float(temporal_blend)
        
        self.previous_homography_image_to_field = None

        self.runtime = load_pnlcalib_runtime(device=device)

    def _adaptive_threshold_schedule(self):
        schedule = [(self.keypoint_threshold, self.line_threshold)]
        if not self.adaptive_thresholds_enabled:
            return schedule
        for attempt_index in range(1, self.adaptive_max_attempts):
            thresholds = (
                max(
                    self.adaptive_keypoint_threshold_min,
                    self.keypoint_threshold - attempt_index * self.adaptive_keypoint_threshold_step,
                ),
                max(
                    self.adaptive_line_threshold_min,
                    self.line_threshold - attempt_index * self.adaptive_line_threshold_step,
                ),
            )
            if thresholds not in schedule:
                schedule.append(thresholds)
        return schedule

    def _estimate_with_adaptive_thresholds(self, projected_frame):
        cached_outputs = prepare_frame_and_forward(projected_frame, self.runtime)
        attempts = []
        for attempt_index, (keypoint_threshold, line_threshold) in enumerate(self._adaptive_threshold_schedule()):
            estimate = estimate_pnlcalib_from_cached_outputs(
                cached_outputs,
                runtime=self.runtime,
                geometry=self.geometry,
                keypoint_threshold=keypoint_threshold,
                line_threshold=line_threshold,
                pnl_refine=self.pnl_refine,
            )
            diagnostics = self.quality_analyzer.evaluate_projection_quality(
                estimate,
                frame_shape_hw=projected_frame.shape[:2],
                keypoint_threshold=keypoint_threshold,
                line_threshold=line_threshold,
                attempt_index=attempt_index,
            )
            attempts.append((estimate, diagnostics))

        estimate, diagnostics = max(
            attempts,
            key=lambda item: (
                int(item[1]["accepted"]),
                float(item[1]["quality_score"] or 0.0),
                float(item[1]["geometry_fit"] or 0.0),
                float(item[1]["support_quality"] or 0.0),
                float(item[1]["coverage_quality"] or 0.0),
                -int(item[1]["attempt_index"]),
            ),
        )
        quality = {
            "attempts": [attempt_diagnostics for _, attempt_diagnostics in attempts],
            "selected_attempt": diagnostics,
            "smoothed_attempt": None,
            "quality_score": float(diagnostics["quality_score"] or 0.0),
            "quality_status": diagnostics["quality_status"],
            "rejection_type": diagnostics["rejection_type"] or "",
        }

        if not diagnostics["accepted"]:
            self.previous_homography_image_to_field = None
            estimate.homography_image_to_field = None
            estimate.estimation_mode = f"{estimate.estimation_mode}+quality_rejected"
            return estimate, quality

        if self.previous_homography_image_to_field is not None and self.temporal_blend > 0.0:
            previous = self.previous_homography_image_to_field
            current = estimate.homography_image_to_field
            previous = previous / previous[2, 2]
            current = current / current[2, 2]
            blended = self.temporal_blend * previous + (1.0 - self.temporal_blend) * current
            blended /= blended[2, 2]
            smoothed_mode = f"{estimate.estimation_mode}+smoothed"
            original_homography = estimate.homography_image_to_field
            original_mode = estimate.estimation_mode
            estimate.homography_image_to_field = blended
            estimate.estimation_mode = smoothed_mode
            smoothed = self.quality_analyzer.evaluate_projection_quality(
                estimate,
                frame_shape_hw=projected_frame.shape[:2],
                keypoint_threshold=diagnostics["keypoint_threshold"],
                line_threshold=diagnostics["line_threshold"],
                attempt_index=diagnostics["attempt_index"],
            )
            if smoothed["accepted"] and float(smoothed["quality_score"] or 0.0) >= quality["quality_score"]:
                quality["smoothed_attempt"] = smoothed
                quality["quality_score"] = float(smoothed["quality_score"] or 0.0)
                quality["quality_status"] = smoothed["quality_status"]
                quality["rejection_type"] = smoothed["rejection_type"] or ""
            else:
                estimate.homography_image_to_field = original_homography
                estimate.estimation_mode = original_mode

        self.previous_homography_image_to_field = estimate.homography_image_to_field.copy()
        return estimate, quality

    def _ground_points_from_bboxes(self, bbox_xyxy, class_names):
        boxes = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        offsets = np.asarray(
            [GROUND_POINT_BOTTOM_OFFSET_BY_CLASS.get(name, self.bottom_offset_ratio) for name in class_names],
            dtype=np.float32,
        )
        return np.column_stack(
            [0.5 * (x1 + x2), y2 - offsets * np.maximum(y2 - y1, 1.0)]
        ).astype(np.float32)

    @staticmethod
    def _scale_points(points_xy, source_shape_hw, target_shape_hw):
        points_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        if len(points_xy) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        source_height, source_width = source_shape_hw
        target_height, target_width = target_shape_hw
        points_xy = points_xy.copy()
        points_xy[:, 0] *= target_width / float(source_width)
        points_xy[:, 1] *= target_height / float(source_height)
        return points_xy

    @staticmethod
    def _homography_to_original_frame(homography_projected, original_shape_hw, projected_shape_hw):
        if homography_projected is None:
            return None
        original_height, original_width = original_shape_hw
        projected_height, projected_width = projected_shape_hw
        scale = np.array(
            [
                [projected_width / float(original_width), 0.0, 0.0],
                [0.0, projected_height / float(original_height), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        homography = homography_projected @ scale
        return homography / homography[2, 2]

    @staticmethod
    def _project_trace_point(image_point_xy, homography):
        if homography is None:
            return [0.0, 0.0]
        return project_image_points(np.asarray([image_point_xy], dtype=np.float32), homography)[0].tolist()

    def _keypoints_trace(self, estimate, homography, original_shape_hw, projected_shape_hw):
        items = []
        for keypoint_id, point in sorted(estimate.keypoints_dict.items()):
            image_point = self._scale_points(
                [[point["x"], point["y"]]],
                projected_shape_hw,
                original_shape_hw,
            )[0].tolist()
            items.append(
                {
                    "keypoint_id": int(keypoint_id),
                    "confidence": float(point.get("p", 0.0)),
                    "image_position_px": image_point,
                    "field_position_m": self._project_trace_point(image_point, homography),
                }
            )
        return items

    def _lines_trace(self, estimate, homography, original_shape_hw, projected_shape_hw):
        items = []
        for line_id, line in sorted(estimate.lines_dict.items()):
            image_points = self._scale_points(
                [[line["x_1"], line["y_1"]], [line["x_2"], line["y_2"]]],
                projected_shape_hw,
                original_shape_hw,
            ).tolist()
            items.append(
                {
                    "line_id": int(line_id),
                    "confidence_1": float(line.get("p_1", 0.0)),
                    "confidence_2": float(line.get("p_2", 0.0)),
                    "image_point_1_px": image_points[0],
                    "image_point_2_px": image_points[1],
                    "field_point_1_m": self._project_trace_point(image_points[0], homography),
                    "field_point_2_m": self._project_trace_point(image_points[1], homography),
                }
            )
        return items

    def project_frame(self, frame_bgr, detector_packet, *, execution_mode="runtime"):
        detector_clean = detector_packet["clean"]
        collect_debug = str(execution_mode).strip().lower() == "debug"
        original_frame = np.asarray(frame_bgr)
        projected_frame = resize_frame(original_frame, max_width=self.max_width)
        original_shape_hw = original_frame.shape[:2]
        projected_shape_hw = projected_frame.shape[:2]
        ground_points_original = self._ground_points_from_bboxes(
            detector_clean["bbox_xyxy"],
            detector_clean["class_name"],
        )
        ground_points_projected = self._scale_points(
            ground_points_original,
            original_shape_hw,
            projected_shape_hw,
        )

        estimate = None
        quality = None
        homography_image_to_field = None
        try:
            estimate, quality = self._estimate_with_adaptive_thresholds(projected_frame)
            homography_image_to_field = self._homography_to_original_frame(
                estimate.homography_image_to_field,
                original_shape_hw,
                projected_shape_hw,
            )
        except np.linalg.LinAlgError as exc:
            logger.warning("PnLCalib devolvió homografía singular; se usa packet sin homografía. Error: %s", exc)
        except Exception as exc:  # pragma: no cover
            logger.warning("PnLCalib falló en este frame; se usa packet sin homografía. Error: %s", exc)

        attempts = []
        diagnostics = {}
        quality_status = ""
        keypoints = []
        lines = []

        if quality is not None:
            final_attempt = quality["smoothed_attempt"] or quality["selected_attempt"]
            quality_status = final_attempt["quality_status"]
            if collect_debug:
                attempts = [
                    {
                        "attempt_index": int(item["attempt_index"]),
                        "keypoint_threshold": float(item["keypoint_threshold"]),
                        "line_threshold": float(item["line_threshold"]),
                        "accepted": bool(item["accepted"]),
                        "quality_status": item["quality_status"],
                        "quality_score": float(item["quality_score"] or 0.0),
                        "geometry_fit": float(item["geometry_fit"] or 0.0),
                        "support_quality": float(item["support_quality"] or 0.0),
                        "coverage_quality": float(item["coverage_quality"] or 0.0),
                        "rejection_type": item["rejection_type"] or "",
                        "rejection_reasons": list(item["rejection_reasons"]),
                    }
                    for item in quality["attempts"]
                ]
                diagnostics = {
                    "selected_attempt_index": int(final_attempt["attempt_index"]),
                    "rejection_type": quality["rejection_type"],
                    "rejection_reasons": list(final_attempt["rejection_reasons"]),
                    "ground_points_image_projected": ground_points_projected.tolist(),
                    "estimation_mode": estimate.estimation_mode,
                    "quality_status": final_attempt["quality_status"],
                    "quality_score": float(final_attempt["quality_score"] or 0.0),
                    "reprojection_error_px": float(estimate.reprojection_error or 0.0),
                    "visible_keypoints_count": int(estimate.visible_keypoints_count),
                    "visible_lines_count": int(estimate.visible_lines_count),
                    "keypoint_threshold_used": float(final_attempt["keypoint_threshold"]),
                    "line_threshold_used": float(final_attempt["line_threshold"]),
                    "attempt_count": len(attempts),
                }
                keypoints = self._keypoints_trace(
                    estimate,
                    homography_image_to_field,
                    original_shape_hw,
                    projected_shape_hw,
                )
                lines = self._lines_trace(
                    estimate,
                    homography_image_to_field,
                    original_shape_hw,
                    projected_shape_hw,
                )

        if homography_image_to_field is None:
            field_positions = np.zeros((detector_clean["num_detections"], 2), dtype=np.float32)
        else:
            field_positions = project_image_points(
                ground_points_original,
                homography_image_to_field,
            ).astype(np.float32)

        return make_phase_packet(
            phase_name=PHASE_REFERENCE_POINTS,
            frame_index=detector_packet["frame_index"],
            frame_time_ms=detector_packet["frame_time_ms"],
            image_width=detector_packet["image_width"],
            image_height=detector_packet["image_height"],
            clean={
                "num_detections": detector_clean["num_detections"],
                "det_id": list(detector_clean["det_id"]),
                "bbox_xyxy": [list(bbox) for bbox in detector_clean["bbox_xyxy"]],
                "confidence": list(detector_clean["confidence"]),
                "class_name": list(detector_clean["class_name"]),
                "homography_valid": homography_image_to_field is not None,
                "homography_image_to_field_3x3": (
                    homography_image_to_field.tolist()
                    if homography_image_to_field is not None
                    else [row[:] for row in IDENTITY_HOMOGRAPHY_3X3]
                ),
                "field_positions_m": field_positions.tolist(),
                "ground_points_image_original": ground_points_original.tolist(),
                "field_positions_usable_for_tracking": quality_status == "good",
            },
            trace=(
                {
                    "keypoints": keypoints,
                    "lines": lines,
                    "attempts": attempts,
                    "diagnostics": diagnostics,
                }
                if collect_debug
                else {}
            ),
        )


__all__ = ["PnLCalibFieldProjector", "build_reference_points_packet_without_homography"]
