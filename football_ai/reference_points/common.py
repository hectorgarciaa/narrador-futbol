from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from football_ai.reference_points.classical_reference_points import (
        PitchGeometry,
        find_project_root,
        resize_frame,
    )
except ModuleNotFoundError:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class PitchGeometry:
        field_length_m: float = 106.0
        field_width_m: float = 68.0
        goal_width_m: float = 7.32
        penalty_area_depth_m: float = 16.5
        penalty_area_width_m: float = 40.32
        goal_area_depth_m: float = 5.5
        goal_area_width_m: float = 18.32
        penalty_mark_distance_m: float = 11.0
        center_circle_radius_m: float = 9.15
        penalty_arc_radius_m: float = 9.15
        corner_arc_radius_m: float = 1.0
        line_width_m: float = 0.12

        @property
        def halfway_x_m(self) -> float:
            return self.field_length_m / 2.0

        @property
        def center_y_m(self) -> float:
            return self.field_width_m / 2.0

        @property
        def penalty_area_top_y_m(self) -> float:
            return (self.field_width_m - self.penalty_area_width_m) / 2.0

        @property
        def penalty_area_bottom_y_m(self) -> float:
            return (self.field_width_m + self.penalty_area_width_m) / 2.0

        @property
        def goal_area_top_y_m(self) -> float:
            return (self.field_width_m - self.goal_area_width_m) / 2.0

        @property
        def goal_area_bottom_y_m(self) -> float:
            return (self.field_width_m + self.goal_area_width_m) / 2.0

    def find_project_root(start_path: Optional[Path] = None) -> Path:
        path = Path(start_path or Path.cwd()).resolve()
        for candidate in (path,) + tuple(path.parents):
            if (candidate / "config.yaml").exists():
                return candidate
        raise FileNotFoundError("No se ha encontrado config.yaml desde la ruta actual.")

    def resize_frame(frame_bgr: np.ndarray, max_width: int = 1280) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        if width <= max_width:
            return frame_bgr.copy()
        scale = max_width / float(width)
        return cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


__all__ = [
    "PitchGeometry",
    "find_project_root",
    "resize_frame",
]
