from __future__ import annotations

from typing import Optional

import numpy as np
from supervision.tracker.byte_tracker.single_object_track import STrack, TrackState

from .utils import field_position_to_array, shirt_color_to_array


class ByteTrackDetectionMetadata:
    def _resolve_detection_class(
        self,
        class_name_relabel: Optional[str],
        class_name_yolo: Optional[str],
    ) -> Optional[str]:
        return class_name_relabel if class_name_relabel is not None else class_name_yolo

    def _update_track_class_consensus(self, track: STrack, det: STrack) -> None:
        votes = getattr(track, "class_votes", None)
        if not isinstance(votes, dict):
            votes = {}

        class_relabel = getattr(det, "class_name_team", None)
        class_yolo = getattr(det, "class_name_yolo", None)
        if class_relabel is not None and self.class_vote_weight_relabel > 0.0:
            votes[class_relabel] = float(votes.get(class_relabel, 0.0)) + self.class_vote_weight_relabel
        if class_yolo is not None and self.class_vote_weight_yolo > 0.0:
            votes[class_yolo] = float(votes.get(class_yolo, 0.0)) + self.class_vote_weight_yolo
        track.class_votes = votes
        if not votes:
            return

        best_class = max(votes.items(), key=lambda item: item[1])[0]
        current_class = getattr(track, "class_name", None)
        if current_class is None:
            track.class_name = best_class
            return
        if best_class != current_class and votes[best_class] >= (float(votes.get(current_class, 0.0)) + self.class_consensus_switch_margin):
            track.class_name = best_class

    def _update_track_team_consensus(self, track: STrack, det: STrack) -> None:
        new_team = getattr(det, "equipo", None)
        if new_team is None:
            return

        votes = getattr(track, "team_votes", None)
        if not isinstance(votes, dict):
            votes = {}
        if self.team_vote_weight > 0.0:
            votes[new_team] = float(votes.get(new_team, 0.0)) + self.team_vote_weight
        track.team_votes = votes

        current_team = getattr(track, "equipo", None)
        if current_team is None:
            track.equipo = new_team
            return
        if votes and new_team != current_team:
            best_team = max(votes.items(), key=lambda item: item[1])[0]
            if votes[best_team] >= (float(votes.get(current_team, 0.0)) + self.team_consensus_switch_margin):
                track.equipo = best_team

    def _apply_track_metadata(self, track: STrack, det: STrack) -> None:
        if getattr(det, "shirt_color", None) is not None:
            track.shirt_color = det.shirt_color
        if getattr(det, "class_name_team", None) is not None:
            track.class_name_team = det.class_name_team
        if getattr(det, "class_name_yolo", None) is not None:
            track.class_name_yolo = det.class_name_yolo
        if getattr(track, "class_name", None) is None and getattr(det, "class_name", None) is not None:
            track.class_name = det.class_name
        self._update_track_class_consensus(track, det)
        self._update_track_team_consensus(track, det)
        det_field_position = field_position_to_array(getattr(det, "field_position", None))
        if det_field_position is not None:
            track.field_position = det_field_position.copy()
        raw_det_idx = getattr(det, "raw_det_idx", None)
        if raw_det_idx is not None:
            track.raw_det_idx = int(raw_det_idx)

    def _build_detection(
        self,
        source_index: int,
        tlbr: np.ndarray,
        score: float,
        *,
        team_labels,
        class_labels,
        yolo_class_labels,
        field_positions,
        shirt_colors,
        raw_det_indices,
    ) -> STrack:
        det = STrack(
            STrack.tlbr_to_tlwh(tlbr),
            score,
            self.minimum_consecutive_frames,
            self.shared_kalman,
            self.internal_id_counter,
            self.external_id_counter,
        )
        det.equipo = team_labels[source_index] if team_labels is not None and source_index < len(team_labels) else None
        det.class_name_team = class_labels[source_index] if class_labels is not None and source_index < len(class_labels) else None
        det.class_name_yolo = yolo_class_labels[source_index] if yolo_class_labels is not None and source_index < len(yolo_class_labels) else None
        det.class_name = self._resolve_detection_class(det.class_name_team, det.class_name_yolo)
        det.field_position = (
            field_position_to_array(field_positions[source_index])
            if field_positions is not None and source_index < len(field_positions)
            else None
        )
        det.shirt_color = (
            shirt_color_to_array(shirt_colors[source_index])
            if shirt_colors is not None and source_index < len(shirt_colors)
            else None
        )
        det.raw_det_idx = int(raw_det_indices[source_index]) if source_index < len(raw_det_indices) else int(source_index)
        return det

    def _build_detection_batch(
        self,
        source_indices,
        boxes,
        scores,
        *,
        team_labels,
        class_labels,
        yolo_class_labels,
        field_positions,
        shirt_colors,
        raw_det_indices,
    ) -> list[STrack]:
        return [
            self._build_detection(
                source_index,
                tlbr,
                score,
                team_labels=team_labels,
                class_labels=class_labels,
                yolo_class_labels=yolo_class_labels,
                field_positions=field_positions,
                shirt_colors=shirt_colors,
                raw_det_indices=raw_det_indices,
            )
            for source_index, tlbr, score in zip(source_indices, boxes, scores)
        ]

    def _apply_match(
        self,
        track: STrack,
        det: STrack,
        *,
        activated_tracks: list[STrack],
        refound_tracks: list[STrack],
    ) -> None:
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id)
            activated_tracks.append(track)
        else:
            track.re_activate(det, self.frame_id)
            refound_tracks.append(track)
        self._apply_track_metadata(track, det)
