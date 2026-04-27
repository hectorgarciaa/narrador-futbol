from __future__ import annotations

from football_ai.core import Phase

from .post_projection import filter_reference_points


class FilteringPhase(Phase):
    def execute(
        self,
        reference_packet,
        *,
        active_track_boxes_xyxy=None,
        sideline_margin_m=0.75,
        geometry=None,
    ):
        return filter_reference_points(
            reference_packet,
            active_track_boxes_xyxy=active_track_boxes_xyxy,
            sideline_margin_m=sideline_margin_m,
            geometry=geometry,
        )


__all__ = ["FilteringPhase"]
