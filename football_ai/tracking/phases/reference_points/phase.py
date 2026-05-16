from __future__ import annotations

from football_ai.core import Phase

from .projector import (
    PnLCalibFieldProjector,
    build_reference_points_packet_without_homography,
)


class ProjectionPhase(Phase):
    def __init__(self, projector_conf=None):
        runtime_conf = dict(projector_conf or {})
        constructor_conf = dict(runtime_conf.get("constructor", {}) or {})
        if bool(runtime_conf.get("enabled", False)):
            self.projector = PnLCalibFieldProjector(
                project_root=runtime_conf["project_root"],
                **constructor_conf,
            )
        else:
            self.projector = None

    @property
    def geometry(self):
        return getattr(self.projector, "geometry", None)

    def execute(self, frame_bgr, detector_packet, *, execution_mode="runtime"):
        if self.projector is None:
            return build_reference_points_packet_without_homography(
                detector_packet,
                execution_mode=execution_mode,
            )
        return self.projector.project_frame(
            frame_bgr,
            detector_packet,
            execution_mode=execution_mode,
        )


__all__ = ["ProjectionPhase"]
