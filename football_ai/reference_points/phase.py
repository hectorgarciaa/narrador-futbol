from __future__ import annotations

from football_ai.core import Phase

from .projector import (
    PnLCalibFieldProjector,
    build_reference_points_packet_without_homography,
)


class ProjectionPhase(Phase):
    def __init__(self, projector: PnLCalibFieldProjector | None):
        self.projector = projector

    @property
    def geometry(self):
        return getattr(self.projector, "geometry", None)

    def execute(self, frame_bgr, detector_packet):
        if self.projector is None:
            return build_reference_points_packet_without_homography(detector_packet)
        return self.projector.project_frame(frame_bgr, detector_packet)


__all__ = ["ProjectionPhase"]
