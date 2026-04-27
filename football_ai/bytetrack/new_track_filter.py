from __future__ import annotations

from typing import Optional

from supervision.tracker.byte_tracker.single_object_track import STrack, TrackState

from .utils import tlbr_iou, track_tlbr


class ByteTrackNewTrackFilter:
    def _filter_new_track_candidate_indices(
        self,
        detections: list[STrack],
        candidate_indices: list[int],
        active_tracked_pool: list[STrack],
        reference_unconfirmed_pool: Optional[list[STrack]] = None,
        return_reasons: bool = False,
    ):
        active_overlap_iou = max(1e-6, float(self.new_track_active_overlap_iou))
        unconfirmed_overlap_iou = max(1e-6, float(self.new_track_unconfirmed_overlap_iou))
        candidate_overlap_iou = max(1e-6, float(self.new_track_candidate_overlap_iou))

        valid_indices = []
        reason_by_index = {}
        for idx in candidate_indices:
            candidate_box = track_tlbr(detections[idx])
            overlaps_active = any(
                tlbr_iou(candidate_box, track_tlbr(active_track)) > active_overlap_iou
                for active_track in active_tracked_pool
            )
            if overlaps_active:
                reason_by_index[idx] = "suppressed_by_active_overlap"
            else:
                valid_indices.append(idx)

        valid_after_unconfirmed = []
        for idx in valid_indices:
            candidate_box = track_tlbr(detections[idx])
            overlaps_unconfirmed = any(
                tlbr_iou(candidate_box, track_tlbr(unconfirmed_track)) > unconfirmed_overlap_iou
                for unconfirmed_track in (reference_unconfirmed_pool or [])
            )
            if overlaps_unconfirmed:
                reason_by_index[idx] = "suppressed_by_unconfirmed_overlap"
            else:
                valid_after_unconfirmed.append(idx)

        valid_after_unconfirmed.sort(
            key=lambda i: float(getattr(detections[i], "score", 0.0)),
            reverse=True,
        )
        selected_indices = []
        selected_boxes = []
        for idx in valid_after_unconfirmed:
            candidate_box = track_tlbr(detections[idx])
            overlaps_selected = any(
                tlbr_iou(candidate_box, selected_box) > candidate_overlap_iou
                for selected_box in selected_boxes
            )
            if overlaps_selected:
                reason_by_index[idx] = "suppressed_by_candidate_overlap"
                continue
            selected_indices.append(idx)
            selected_boxes.append(candidate_box)
            reason_by_index[idx] = "passed_new_track_filter"
        return (selected_indices, reason_by_index) if return_reasons else selected_indices

    @staticmethod
    def _active_tracked_pool(tracked_tracks, activated_tracks, refound_tracks):
        active_tracked_pool = []
        seen_internal_ids = set()
        for candidate_track in list(tracked_tracks) + list(activated_tracks) + list(refound_tracks):
            internal_track_id = getattr(candidate_track, "internal_track_id", None)
            if internal_track_id is not None:
                if internal_track_id in seen_internal_ids:
                    continue
                seen_internal_ids.add(internal_track_id)
            if getattr(candidate_track, "state", None) != TrackState.Tracked:
                continue
            if not bool(getattr(candidate_track, "is_activated", False)):
                continue
            active_tracked_pool.append(candidate_track)
        return active_tracked_pool
