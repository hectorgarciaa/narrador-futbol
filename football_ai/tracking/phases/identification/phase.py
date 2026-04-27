from __future__ import annotations

from football_ai.core import Phase

from .team_detector import TeamDetector


class IdentificationPhase(Phase):
    def __init__(
        self,
        team_detector_conf,
        *,
        referee_field_width_m,
        referee_sideline_band_distance_m,
    ):
        self.team_detector = TeamDetector(**dict(team_detector_conf or {}))
        self.referee_field_width_m = float(referee_field_width_m)
        self.referee_sideline_band_distance_m = float(
            referee_sideline_band_distance_m
        )

    def execute(self, frame_bgr, filtering_packet, *, show_kmeans=False):
        return self.team_detector.identify_packet(
            frame_bgr,
            filtering_packet,
            self.referee_field_width_m,
            self.referee_sideline_band_distance_m,
            show_kmeans,
        )


__all__ = ["IdentificationPhase"]
