from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from .common import PitchGeometry


def homography_from_camera_result(camera_result: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if camera_result is None:
        return None

    cam_params = camera_result["cam_params"]
    intrinsics = np.array(
        [
            [cam_params["x_focal_length"], 0.0, cam_params["principal_point"][0]],
            [0.0, cam_params["y_focal_length"], cam_params["principal_point"][1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(cam_params["rotation_matrix"], dtype=np.float64)
    position = np.asarray(cam_params["position_meters"], dtype=np.float64)
    translation = -(rotation @ position.reshape(3, 1))
    homography_centered_to_image = intrinsics @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], translation[:, 0]]
    )
    try:
        homography_image_to_centered = np.linalg.inv(homography_centered_to_image)
    except np.linalg.LinAlgError:
        return None
    return homography_image_to_centered / homography_image_to_centered[2, 2]


def centered_to_field_homography(
    homography_image_to_centered: Optional[np.ndarray],
    geometry: PitchGeometry,
) -> Optional[np.ndarray]:
    if homography_image_to_centered is None:
        return None

    centered_to_field = np.array(
        [
            [1.0, 0.0, geometry.field_length_m / 2.0],
            [0.0, 1.0, geometry.field_width_m / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homography_image_to_field = centered_to_field @ homography_image_to_centered
    return homography_image_to_field / homography_image_to_field[2, 2]

def project_image_points(
    points_xy: np.ndarray,
    homography_image_to_field: np.ndarray,
) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points_xy, homography_image_to_field)
    return transformed.reshape(-1, 2)


def points_inside_field_mask(
    field_points_m: np.ndarray,
    geometry: Optional[PitchGeometry] = None,
    margin_m: float = 2.0,
) -> np.ndarray:
    if geometry is None:
        geometry = PitchGeometry()
    points_array = np.asarray(field_points_m, dtype=np.float32)
    x_values = points_array[:, 0]
    y_values = points_array[:, 1]
    return (
        (x_values >= -margin_m)
        & (x_values <= geometry.field_length_m + margin_m)
        & (y_values >= -margin_m)
        & (y_values <= geometry.field_width_m + margin_m)
    )


__all__ = [
    "centered_to_field_homography",
    "homography_from_camera_result",
    "points_inside_field_mask",
    "project_image_points",
]
