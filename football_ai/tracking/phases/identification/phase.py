from __future__ import annotations

from football_ai.core import Phase

from .team_detector import TeamDetector


class IdentificationPhase(Phase):
    def __init__(self, identification_conf=None):
        runtime_conf = dict(identification_conf or {})
        self.team_detector = TeamDetector(
            **dict(runtime_conf.get("team_detector_conf") or {})
        )
        self.referee_field_width_m = float(
            runtime_conf.get("referee_field_width_m", 68.0)
        )
        self.referee_sideline_band_distance_m = float(
            runtime_conf.get("referee_sideline_band_distance_m", 3.0)
        )

    def execute(
        self,
        frame_bgr,
        filtering_packet,
        show_kmeans=False,
        *,
        execution_mode="runtime",
    ):
        return self.team_detector.identify_packet(
            frame_bgr,
            filtering_packet,
            self.referee_field_width_m,
            self.referee_sideline_band_distance_m,
            show_kmeans,
            execution_mode=execution_mode,
        )


__all__ = ["IdentificationPhase"]
