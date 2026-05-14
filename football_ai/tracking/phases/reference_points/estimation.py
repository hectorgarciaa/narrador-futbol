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
    centered_to_field_homography,
    homography_from_camera_result,
)
from .runtime_loader import PnLCalibRuntime


@dataclass
class PnLCalibEstimate:
    keypoints_dict: dict[int, dict[str, float]]
    lines_dict: dict[int, dict[str, float]]
    camera_result: dict | None
    ground_result: dict | None
    homography_image_to_field: np.ndarray | None
    reprojection_error: float | None
    estimation_mode: str

    @property
    def visible_keypoints_count(self) -> int:
        return len(self.keypoints_dict)

    @property
    def visible_lines_count(self) -> int:
        return len(self.lines_dict)


def prepare_frame_and_forward(
    frame_bgr: np.ndarray,
    runtime: PnLCalibRuntime,
):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_tensor = TF.to_tensor(Image.fromarray(frame_rgb)).float().unsqueeze(0)
    _, _, original_height, original_width = frame_tensor.size()
    if frame_tensor.size()[-2:] != (540, 960):
        frame_tensor = runtime.resize_transform(frame_tensor)
    frame_tensor = frame_tensor.to(runtime.device)
    _, _, resized_height, resized_width = frame_tensor.size()
    with torch.no_grad():
        keypoint_heatmaps = runtime.model_kp(frame_tensor)
        line_heatmaps = runtime.model_line(frame_tensor)
    return {
        "original_width": int(original_width),
        "original_height": int(original_height),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "keypoint_coords": runtime.get_keypoints_from_heatmap_batch_maxpool(
            keypoint_heatmaps[:, :-1, :, :]
        ),
        "line_coords": runtime.get_keypoints_from_heatmap_batch_maxpool_l(
            line_heatmaps[:, :-1, :, :]
        ),
    }


def estimate_pnlcalib_from_cached_outputs(
    cached_outputs,
    runtime: PnLCalibRuntime,
    geometry: PitchGeometry,
    keypoint_threshold: float = 0.3434,
    line_threshold: float = 0.7867,
    pnl_refine: bool = True,
) -> PnLCalibEstimate:
    
    keypoints_batch = runtime.coords_to_dict(
        cached_outputs["keypoint_coords"],
        threshold=keypoint_threshold,
    )
    lines_batch = runtime.coords_to_dict(
        cached_outputs["line_coords"],
        threshold=line_threshold,
    )
    keypoints_dict, lines_dict = runtime.complete_keypoints(
        keypoints_batch[0],
        lines_batch[0],
        w=cached_outputs["resized_width"],
        h=cached_outputs["resized_height"],
        normalize=True,
    )

    calibrator = runtime.framebyframe_calib_cls(
        iwidth=cached_outputs["original_width"],
        iheight=cached_outputs["original_height"],
        denormalize=True,
    )
    calibrator.update(keypoints_dict, lines_dict)

    ground_result = calibrator.heuristic_voting_ground(refine_lines=pnl_refine)
    camera_result = None

    homography_image_to_centered = None
    estimation_mode = "no_solution"
    reprojection_error: Optional[float] = None

    if ground_result is not None:
        homography_image_to_centered = np.asarray(ground_result["homography"], dtype=np.float64)
        reprojection_error = float(ground_result["rep_err"])
        estimation_mode = "ground_plane"
    else:
        camera_result = calibrator.heuristic_voting(refine_lines=pnl_refine)
        if camera_result is not None:
            homography_image_to_centered = homography_from_camera_result(camera_result)
            reprojection_error = float(camera_result["rep_err"])
            estimation_mode = "camera_fallback"

    homography_image_to_field = centered_to_field_homography(
        homography_image_to_centered,
        geometry,
    )

    return PnLCalibEstimate(
        keypoints_dict=keypoints_dict,
        lines_dict=lines_dict,
        camera_result=camera_result,
        ground_result=ground_result,
        homography_image_to_field=homography_image_to_field,
        reprojection_error=reprojection_error,
        estimation_mode=estimation_mode,
    )


__all__ = [
    "PnLCalibEstimate",
    "estimate_pnlcalib_from_cached_outputs",
    "prepare_frame_and_forward",
]
