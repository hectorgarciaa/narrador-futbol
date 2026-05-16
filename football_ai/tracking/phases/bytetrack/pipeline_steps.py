from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from supervision.tracker.byte_tracker.single_object_track import STrack, TrackState

from .utils import joint_tracks, remove_duplicate_tracks, sub_tracks

try:
    import lap
except Exception:  # pragma: no cover - fallback runtime path
    lap = None


class ByteTrackPipelineSteps:
    def _run_assignment_solver(self, sub_cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if lap is not None:
            _cost, row_assignments, _ = lap.lapjv(
                sub_cost.astype(np.float64, copy=False),
                extend_cost=True,
            )
            row_indices = np.arange(len(row_assignments), dtype=np.int32)
            valid_mask = row_assignments >= 0
            return row_indices[valid_mask], row_assignments[valid_mask].astype(np.int32)
        return linear_sum_assignment(sub_cost)

    def _partition_existing_tracks(self):
        unconfirmed = []
        tracked_tracks = []
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracks.append(track)
        return tracked_tracks, unconfirmed

    def _solve_phase_matches(
        self,
        tracks: list[STrack],
        detections: list[STrack],
        *,
        phase_name: str,
        cost_mode: str,
        threshold: float | None,
    ) -> tuple[list[tuple[int, int]], list[int], list[int], dict[str, object]]:
        shape = (len(tracks), len(detections))
        if not tracks or not detections:
            match_data = {
                "phase_name": phase_name,
                "cost_mode": cost_mode,
                "base_cost": np.full(shape, np.inf, dtype=np.float32),
                "feasible_mask": np.zeros(shape, dtype=bool),
                "pair_metrics": (
                    self._empty_pair_metrics(shape, np.zeros(shape, dtype=np.float32))
                    if self.collect_internal_matching_debug
                    else None
                ),
            }
            if self.collect_internal_matching_debug:
                self._collect_phase_matching_debug(
                    phase_name=phase_name,
                    tracks=tracks,
                    detections=detections,
                    match_data=match_data,
                    matches=[],
                    threshold=threshold,
                )
            return [], list(range(len(tracks))), list(range(len(detections))), match_data

        match_data = self._build_phase_match_data(
            tracks,
            detections,
            phase_name=phase_name,
            cost_mode=cost_mode,
        )
        feasible_mask = match_data["feasible_mask"]
        base_cost = match_data["base_cost"]
        matches: list[tuple[int, int]] = []

        row_indices = [i for i in range(len(tracks)) if bool(feasible_mask[i].any())]
        col_indices = [j for j in range(len(detections)) if bool(feasible_mask[:, j].any())]

        if row_indices and col_indices:
            row_lookup = {track_idx: offset for offset, track_idx in enumerate(row_indices)}
            col_lookup = {det_idx: offset for offset, det_idx in enumerate(col_indices)}
            sub_cost = np.full(
                (len(row_indices), len(col_indices)),
                self.large_match_cost,
                dtype=np.float32,
            )
            for track_idx in row_indices:
                feasible_det_indices = np.where(feasible_mask[track_idx])[0].tolist()
                for det_idx in feasible_det_indices:
                    sub_cost[row_lookup[track_idx], col_lookup[det_idx]] = base_cost[
                        track_idx, det_idx
                    ]

            row_ind, col_ind = self._run_assignment_solver(sub_cost)
            for row_offset, col_offset in zip(row_ind.tolist(), col_ind.tolist()):
                track_idx = row_indices[row_offset]
                det_idx = col_indices[col_offset]
                if not bool(feasible_mask[track_idx, det_idx]):
                    continue
                cost = float(base_cost[track_idx, det_idx])
                if cost >= self.large_match_cost:
                    continue
                if threshold is not None and cost > float(threshold):
                    continue
                matches.append((track_idx, det_idx))

        matched_track_indices = {track_idx for track_idx, _ in matches}
        matched_det_indices = {det_idx for _, det_idx in matches}
        unmatched_track_indices = [
            index for index in range(len(tracks)) if index not in matched_track_indices
        ]
        unmatched_det_indices = [
            index for index in range(len(detections)) if index not in matched_det_indices
        ]

        if self.collect_internal_matching_debug:
            self._collect_phase_matching_debug(
                phase_name=phase_name,
                tracks=tracks,
                detections=detections,
                match_data=match_data,
                matches=matches,
                threshold=threshold,
            )
        return matches, unmatched_track_indices, unmatched_det_indices, match_data

    def _associate_confirmed_phase(
        self,
        tracks: list[STrack],
        detections: list[STrack],
        *,
        phase_name: str,
        cost_mode: str,
        threshold: float | None,
        activated_tracks,
        refound_tracks,
        reason: str,
    ) -> tuple[list[STrack], list[STrack]]:
        matches, unmatched_track_indices, unmatched_det_indices, _ = self._solve_phase_matches(
            tracks,
            detections,
            phase_name=phase_name,
            cost_mode=cost_mode,
            threshold=threshold,
        )
        for track_idx, det_idx in matches:
            track = tracks[track_idx]
            det = detections[det_idx]
            self._set_detection_debug_reason(
                getattr(det, "raw_det_idx", None),
                reason,
                class_name=getattr(det, "class_name", None),
                confidence=getattr(det, "score", None),
                stage=phase_name,
                tracker_id=getattr(track, "external_track_id", None),
            )
            self._apply_match(
                track,
                det,
                activated_tracks=activated_tracks,
                refound_tracks=refound_tracks,
            )
        return (
            [tracks[index] for index in unmatched_track_indices],
            [detections[index] for index in unmatched_det_indices],
        )

    def _associate_unconfirmed_phase(
        self,
        tracks: list[STrack],
        detections: list[STrack],
        *,
        phase_name: str,
        cost_mode: str,
        threshold: float | None,
        activated_tracks,
        reason: str,
    ) -> tuple[list[STrack], list[STrack], list[STrack]]:
        matches, unmatched_track_indices, unmatched_det_indices, _ = self._solve_phase_matches(
            tracks,
            detections,
            phase_name=phase_name,
            cost_mode=cost_mode,
            threshold=threshold,
        )
        accepted_tracks = []
        for track_idx, det_idx in matches:
            track = tracks[track_idx]
            det = detections[det_idx]
            self._set_detection_debug_reason(
                getattr(det, "raw_det_idx", None),
                reason,
                class_name=getattr(det, "class_name", None),
                confidence=getattr(det, "score", None),
                stage=phase_name,
                tracker_id=getattr(track, "external_track_id", None),
            )
            track.update(det, self.frame_id)
            self._apply_track_metadata(track, det)
            activated_tracks.append(track)
            accepted_tracks.append(track)
        return (
            [tracks[index] for index in unmatched_track_indices],
            [detections[index] for index in unmatched_det_indices],
            accepted_tracks,
        )

    def _activate_new_tracks(
        self,
        detections: list[STrack],
        surviving_unconfirmed_tracks: list[STrack],
        *,
        activated_tracks,
        refound_tracks,
    ):
        active_tracked_pool = self._active_tracked_pool(
            self.tracked_tracks,
            activated_tracks,
            refound_tracks,
        )
        candidate_indices = [
            index
            for index, det in enumerate(detections)
            if det.score >= self.track_activation_threshold
        ]
        candidate_index_set = set(candidate_indices)
        filtered_candidate_indices, filter_reasons_by_index = (
            self._filter_new_track_candidate_indices(
                detections=detections,
                candidate_indices=candidate_indices,
                active_tracked_pool=active_tracked_pool,
                reference_unconfirmed_pool=surviving_unconfirmed_tracks,
                return_reasons=True,
            )
        )

        for index, det in enumerate(detections):
            if index in candidate_index_set:
                continue
            self._set_detection_debug_reason(
                getattr(det, "raw_det_idx", None),
                "below_new_track_init_threshold",
                class_name=getattr(det, "class_name", None),
                confidence=getattr(det, "score", None),
                stage="new_track_init",
            )

        for index, reason in filter_reasons_by_index.items():
            if index in filtered_candidate_indices:
                continue
            det = detections[index]
            self._set_detection_debug_reason(
                getattr(det, "raw_det_idx", None),
                reason,
                class_name=getattr(det, "class_name", None),
                confidence=getattr(det, "score", None),
                stage="new_track_filter",
            )

        for index in filtered_candidate_indices:
            track = detections[index]
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

        all_removed_tracks = joint_tracks(self.removed_tracks, removed_tracks)

        tracked_tracks = [track for track in self.tracked_tracks if track.state == TrackState.Tracked]
        tracked_tracks = joint_tracks(tracked_tracks, activated_tracks)
        tracked_tracks = joint_tracks(tracked_tracks, refound_tracks)
        tracked_tracks = sub_tracks(tracked_tracks, all_removed_tracks)

        lost_tracks_next = sub_tracks(self.lost_tracks, tracked_tracks)
        lost_tracks_next = joint_tracks(lost_tracks_next, lost_tracks)
        lost_tracks_next = sub_tracks(lost_tracks_next, all_removed_tracks)

        self.tracked_tracks = tracked_tracks
        self.lost_tracks = lost_tracks_next
        self.removed_tracks = all_removed_tracks
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks,
            self.lost_tracks,
        )
