from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from dataclasses import dataclass
from PIL import Image

from .common import PitchGeometry
from .geometry import (
    _centered_to_field_homography,
    _homography_from_camera_result,
    _projection_from_cam_params,
    _to_template_homography,
)
from .runtime_loader import PnLCalibRuntime


@dataclass
class PnLCalibEstimate:
    keypoints_dict: dict[int, dict[str, float]]
    lines_dict: dict[int, dict[str, float]]
    camera_result: dict | None
    ground_result: dict | None
    homography_image_to_field: np.ndarray | None
    homography_image_to_template: np.ndarray | None
    projection_matrix: np.ndarray | None
    reprojection_error: float | None
    estimation_mode: str

    @property
    def visible_keypoints_count(self) -> int:
        return len(self.keypoints_dict)

    @property
    def visible_lines_count(self) -> int:
        return len(self.lines_dict)


def estimate_pnlcalib_frame(
    frame_bgr: np.ndarray,
    runtime: PnLCalibRuntime,
    geometry: Optional[PitchGeometry] = None,
    keypoint_threshold: float = 0.3434,
    line_threshold: float = 0.7867,
    pnl_refine: bool = True,
    pixels_per_meter: int = 8,
) -> PnLCalibEstimate:
    if geometry is None:
        geometry = PitchGeometry()

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_tensor = TF.to_tensor(Image.fromarray(frame_rgb)).float().unsqueeze(0)
    _, _, original_height, original_width = frame_tensor.size()
    if frame_tensor.size()[-1] != 960:
        frame_tensor = runtime.resize_transform(frame_tensor)
    frame_tensor = frame_tensor.to(runtime.device)
    _, _, resized_height, resized_width = frame_tensor.size()

    with torch.no_grad():
        heatmaps_kp = runtime.model_kp(frame_tensor)
        heatmaps_line = runtime.model_line(frame_tensor)

    keypoint_coords = runtime.get_keypoints_from_heatmap_batch_maxpool(
        heatmaps_kp[:, :-1, :, :]
    )
    line_coords = runtime.get_keypoints_from_heatmap_batch_maxpool_l(
        heatmaps_line[:, :-1, :, :]
    )
    keypoints_batch = runtime.coords_to_dict(keypoint_coords, threshold=keypoint_threshold)
    lines_batch = runtime.coords_to_dict(line_coords, threshold=line_threshold)
    keypoints_dict, lines_dict = runtime.complete_keypoints(
        keypoints_batch[0],
        lines_batch[0],
        w=resized_width,
        h=resized_height,
        normalize=True,
    )

    calibrator = runtime.framebyframe_calib_cls(
        iwidth=original_width,
        iheight=original_height,
        denormalize=True,
    )
    calibrator.update(keypoints_dict, lines_dict)

    ground_result = calibrator.heuristic_voting_ground(refine_lines=pnl_refine)
    camera_result = calibrator.heuristic_voting(refine_lines=pnl_refine)

    homography_image_to_centered = None
    estimation_mode = "no_solution"
    reprojection_error: Optional[float] = None

    if ground_result is not None:
        homography_image_to_centered = np.asarray(ground_result["homography"], dtype=np.float64)
        reprojection_error = float(ground_result["rep_err"])
        estimation_mode = "ground_plane"
    else:
        homography_image_to_centered = _homography_from_camera_result(camera_result)
        if camera_result is not None:
            reprojection_error = float(camera_result["rep_err"])
            estimation_mode = "camera_fallback"

    homography_image_to_field = _centered_to_field_homography(
        homography_image_to_centered,
        geometry,
    )
    homography_image_to_template = None
    if homography_image_to_field is not None:
        homography_image_to_template = _to_template_homography(
            homography_image_to_field,
            pixels_per_meter=pixels_per_meter,
        )

    projection_matrix = None
    if camera_result is not None:
        projection_matrix = _projection_from_cam_params(camera_result["cam_params"])

    return PnLCalibEstimate(
        keypoints_dict=keypoints_dict,
        lines_dict=lines_dict,
        camera_result=camera_result,
        ground_result=ground_result,
        homography_image_to_field=homography_image_to_field,
        homography_image_to_template=homography_image_to_template,
        projection_matrix=projection_matrix,
        reprojection_error=reprojection_error,
        estimation_mode=estimation_mode,
    )


__all__ = ["PnLCalibEstimate", "estimate_pnlcalib_frame"]
