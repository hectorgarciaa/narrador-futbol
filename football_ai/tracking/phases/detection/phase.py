from __future__ import annotations

from football_ai.core import Phase

from .detector import Detector


class DetectionPhase(Phase):
    def __init__(self, detector_conf=None):
        runtime_conf = dict(detector_conf or {})
        model_path = runtime_conf.pop("model_path")
        self.detector = Detector(model_path, **runtime_conf)

    def execute(
        self,
        frame_bgr,
        *,
        frame_index=0,
        frame_time_ms=0.0,
        execution_mode="runtime",
    ):
        return self.detector.predict_frame(
            frame_bgr,
            frame_index=frame_index,
            frame_time_ms=frame_time_ms,
            execution_mode=execution_mode,
        )


__all__ = ["DetectionPhase"]
