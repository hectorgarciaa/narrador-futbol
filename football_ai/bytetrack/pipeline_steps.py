from __future__ import annotations

import numpy as np
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.single_object_track import STrack, TrackState

from .utils import joint_tracks, remove_duplicate_tracks, sub_tracks


class ByteTrackPipelineSteps:
    def _partition_existing_tracks(self):
        unconfirmed = []
        tracked_tracks = []
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracks.append(track)
        return tracked_tracks, unconfirmed

    def _associate_high_confidence(
        self,
        strack_pool,
        detections,
        *,
        activated_tracks,
        refound_tracks,
    ):
        first_costs = self._build_association_costs(
            strack_pool,
            detections,
            include_team_penalty=True,
            include_fuse_score=True,
            use_field_positions=True,
        )
        matches, u_track, u_detection = matching.linear_assignment(
            first_costs["final_costs"],
            thresh=self.minimum_matching_threshold,
        )
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            self._set_detection_debug_reason(
                getattr(det, "raw_det_idx", None),
                "matched_existing_track_high",
                class_name=getattr(det, "class_name", None),
                confidence=getattr(det, "score", None),
                stage="first_association",
                tracker_id=getattr(track, "external_track_id", None),
            )
            self._apply_match(
                track,
                det,
                activated_tracks=activated_tracks,
                refound_tracks=refound_tracks,
            )
        return u_track, u_detection

    def _associate_low_confidence(
        self,
        strack_pool,
        u_track,
        detections_second,
        *,
        activated_tracks,
        refound_tracks,
        lost_tracks,
    ):
        remaining_tracked_tracks = [
            strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked
        ]
        second_costs = self._build_association_costs(
            remaining_tracked_tracks,
            detections_second,
            include_team_penalty=False,
            include_fuse_score=False,
            use_field_positions=True,
        )
        matches, remaining_u_track, u_detection_second = matching.linear_assignment(
            second_costs["final_costs"],
            thresh=self.second_match_threshold,
        )
        for itracked, idet in matches:
            track = remaining_tracked_tracks[itracked]
            det = detections_second[idet]
            self._set_detection_debug_reason(
                getattr(det, "raw_det_idx", None),
                "matched_existing_track_low",
                class_name=getattr(det, "class_name", None),
                confidence=getattr(det, "score", None),
                stage="second_association",
                tracker_id=getattr(track, "external_track_id", None),
            )
            self._apply_match(
                track,
                det,
                activated_tracks=activated_tracks,
                refound_tracks=refound_tracks,
            )
        for it in remaining_u_track:
            track = remaining_tracked_tracks[it]
            if track.state != TrackState.Lost:
                track.state = TrackState.Lost
                lost_tracks.append(track)
        return u_detection_second

    def _associate_unconfirmed(
        self,
        unconfirmed,
        detections,
        *,
        activated_tracks,
        removed_tracks,
    ):
        unconfirmed_costs = self._build_association_costs(
            unconfirmed,
            detections,
            include_team_penalty=False,
            include_fuse_score=True,
            use_field_positions=self.use_field_positions_for_unconfirmed,
        )
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            unconfirmed_costs["final_costs"],
            thresh=self.unconfirmed_match_threshold,
        )
        self._collect_unconfirmed_association_debug(
            unconfirmed=unconfirmed,
            detections=detections,
            iou_costs=unconfirmed_costs["iou_costs"],
            class_penalties=unconfirmed_costs["class_penalties"],
            bbox_size_penalties=unconfirmed_costs["bbox_size_penalties"],
            after_bbox_costs=unconfirmed_costs["after_bbox_costs"],
            after_field_costs=unconfirmed_costs["after_field_costs"],
            after_fuse_costs=unconfirmed_costs["after_fuse_costs"],
            final_costs=unconfirmed_costs["final_costs"],
            matches=matches,
            threshold=self.unconfirmed_match_threshold,
        )

        accepted_unconfirmed_tracks = []
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            det = detections[idet]
            self._set_detection_debug_reason(
                getattr(det, "raw_det_idx", None),
                "matched_unconfirmed_track",
                class_name=getattr(det, "class_name", None),
                confidence=getattr(det, "score", None),
                stage="unconfirmed_association",
                tracker_id=getattr(track, "external_track_id", None),
            )
            track.update(det, self.frame_id)
            self._apply_track_metadata(track, det)
            activated_tracks.append(track)
            accepted_unconfirmed_tracks.append(track)

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = TrackState.Removed
            removed_tracks.append(track)

        surviving_unconfirmed_tracks = [
            track
            for track in accepted_unconfirmed_tracks
            if getattr(track, "state", None) == TrackState.Tracked
            and not bool(getattr(track, "is_activated", False))
        ]
        return u_detection, surviving_unconfirmed_tracks

    def _activate_new_tracks(
        self,
        detections,
        u_detection,
        surviving_unconfirmed_tracks,
        *,
        activated_tracks,
        refound_tracks,
    ):
        active_tracked_pool = self._active_tracked_pool(
            self.tracked_tracks,
            activated_tracks,
            refound_tracks,
        )
        candidate_indices = [inew for inew in u_detection if detections[inew].score >= self.det_thresh]
        filtered_candidate_indices, filter_reasons_by_index = self._filter_new_track_candidate_indices(
            detections=detections,
            candidate_indices=candidate_indices,
            active_tracked_pool=active_tracked_pool,
            reference_unconfirmed_pool=surviving_unconfirmed_tracks,
            return_reasons=True,
        )

        for inew in u_detection:
            if inew in candidate_indices:
                continue
            det = detections[inew]
            self._set_detection_debug_reason(
                getattr(det, "raw_det_idx", None),
                "below_new_track_init_threshold",
                class_name=getattr(det, "class_name", None),
                confidence=getattr(det, "score", None),
                stage="new_track_init",
            )

        for idx, reason in filter_reasons_by_index.items():
            if idx in filtered_candidate_indices:
                continue
            det = detections[idx]
            self._set_detection_debug_reason(
                getattr(det, "raw_det_idx", None),
                reason,
                class_name=getattr(det, "class_name", None),
                confidence=getattr(det, "score", None),
                stage="new_track_filter",
            )

        for inew in filtered_candidate_indices:
            track = detections[inew]
            track.activate(self.kalman_filter, self.frame_id)
            self._apply_track_metadata(track, track)
            self._set_detection_debug_reason(
                getattr(track, "raw_det_idx", None),
                "activated_new_track",
                class_name=getattr(track, "class_name", None),
                confidence=getattr(track, "score", None),
                stage="new_track_activation",
                tracker_id=getattr(track, "external_track_id", None),
            )
            activated_tracks.append(track)

    def _finalize_frame_state(
        self,
        *,
        activated_tracks,
        refound_tracks,
        lost_tracks,
        removed_tracks,
    ):
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = TrackState.Removed
                removed_tracks.append(track)

        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_tracks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refound_tracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_tracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks = removed_tracks
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks,
            self.lost_tracks,
        )
