from __future__ import annotations

from football_ai.core import Phase

from .detector import Detector


class DetectionPhase(Phase):
    def __init__(self, model_path, detector_conf):
        self.detector = Detector(model_path, **dict(detector_conf or {}))

    def execute(self, frame_bgr, *, frame_index=0, frame_time_ms=0.0):
        return self.detector.predict_frame(
            frame_bgr,
            frame_index=frame_index,
            frame_time_ms=frame_time_ms,
        )


__all__ = ["DetectionPhase"]
