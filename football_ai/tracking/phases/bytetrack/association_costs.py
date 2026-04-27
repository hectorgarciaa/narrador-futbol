from __future__ import annotations

from typing import Optional

import numpy as np
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.single_object_track import STrack

from .utils import bbox_center_from_tlbr, bbox_size_from_tlbr, field_position_to_array


class ByteTrackAssociationCosts:
    def _class_mismatch_pair_penalty(self, track: STrack, det: STrack) -> float:
        track_class = getattr(track, "class_name", None)
        if track_class is None:
            return 0.0

        det_class = getattr(det, "class_name", None)
        if det_class is None or det_class == track_class:
            return 0.0

        det_class_team = getattr(det, "class_name_team", None)
        det_class_yolo = getattr(det, "class_name_yolo", None)
        if det_class_team is None and det_class_yolo is None:
            return self.class_mismatch_penalty
        if det_class_team == track_class:
            return 0.0
        if (
            det_class_team is not None
            and det_class_team != track_class
            and det_class_yolo == track_class
            and self.allow_class_remap_when_signals_disagree
        ):
            return self.class_mismatch_relaxed_penalty
        return self.class_mismatch_penalty

    def _bbox_size_pair_penalty(self, track: STrack, det: STrack) -> float:
        if self.bbox_size_mismatch_penalty <= 0.0:
            return 0.0

        track_width, track_height = bbox_size_from_tlbr(getattr(track, "tlbr", None))
        det_width, det_height = bbox_size_from_tlbr(getattr(det, "tlbr", None))
        if None in (track_width, track_height, det_width, det_height):
            return 0.0

        penalty = 0.0
        if self.bbox_height_ratio_threshold > 0.0:
            min_height = track_height * (1.0 - self.bbox_height_ratio_threshold)
            max_height = track_height * (1.0 + self.bbox_height_ratio_threshold)
            if det_height < min_height or det_height > max_height:
                penalty += self.bbox_size_mismatch_penalty

        if self.bbox_width_ratio_threshold > 0.0:
            min_width = track_width * (1.0 - self.bbox_width_ratio_threshold)
            max_width = track_width * (1.0 + self.bbox_width_ratio_threshold)
            if det_width < min_width or det_width > max_width:
                penalty += self.bbox_size_mismatch_penalty
        return penalty

    def _uses_field_positions_for_class(self, class_name: Optional[str]) -> bool:
        return self.use_field_positions and class_name in self.field_position_classes

    def _track_distance_gate(self, track: STrack) -> float:
        lost_frames = max(1, self.frame_id - int(getattr(track, "frame_id", self.frame_id)))
        if self.field_distance_gate_max_lost_frames is not None:
            lost_frames = min(lost_frames, self.field_distance_gate_max_lost_frames)
        if self.field_distance_growth_mode == "linear_decay":
            gate = 0.0
            for step_idx in range(lost_frames):
                step = self.field_distance_gate_m - (self.field_distance_decay_per_frame * step_idx)
                if step <= 0.0:
                    break
                gate += step
            return min(
                max(self.field_distance_gate_m, gate),
                self.field_distance_gate_cap_m or np.inf,
            )

        gate = self.field_distance_gate_m * (float(lost_frames) ** self.field_distance_lost_exponent)
        if self.field_distance_gate_cap_m is not None:
            gate = min(gate, self.field_distance_gate_cap_m)
        return gate

    def _track_image_distance_gate(self, track: STrack) -> float:
        lost_frames = max(1, self.frame_id - int(getattr(track, "frame_id", self.frame_id)))
        return self.bbox_center_distance_gate_px * lost_frames

    def _track_missed_frames(self, track: STrack) -> int:
        return max(0, self.frame_id - int(getattr(track, "frame_id", self.frame_id)) - 1)

    def _apply_lost_time_penalty(
        self,
        dists: np.ndarray,
        tracks: list[STrack],
    ) -> np.ndarray:
        if self.lost_time_penalty_weight <= 0.0 or dists.size == 0 or not tracks:
            return dists

        for i, track in enumerate(tracks):
            missed_frames = self._track_missed_frames(track)
            if missed_frames <= 0:
                continue
            penalty_factor = min(
                float(missed_frames) / float(self.lost_time_penalty_max_frames),
                1.0,
            )
            dists[i, :] += self.lost_time_penalty_weight * penalty_factor
        return dists

    def _apply_bbox_center_costs(
        self,
        dists: np.ndarray,
        tracks: list[STrack],
        detections: list[STrack],
    ) -> np.ndarray:
        if (
            not self.use_bbox_center_for_matching
            or self.bbox_center_distance_weight <= 0.0
            or dists.size == 0
            or not tracks
            or not detections
        ):
            return dists

        bbox_weight = self.bbox_center_distance_weight
        iou_weight = 1.0 - bbox_weight

        for i, track in enumerate(tracks):
            track_center = bbox_center_from_tlbr(getattr(track, "tlbr", None))
            if not np.all(np.isfinite(track_center)):
                continue
            max_distance = max(self._track_image_distance_gate(track), 1e-6)
            track_class = getattr(track, "class_name", None)
            for j, det in enumerate(detections):
                det_class = getattr(det, "class_name", None)
                if track_class is not None and det_class is not None and track_class != det_class:
                    continue
                det_center = bbox_center_from_tlbr(getattr(det, "tlbr", None))
                if not np.all(np.isfinite(det_center)):
                    continue
                normalized_distance = min(
                    float(np.linalg.norm(track_center - det_center)) / max_distance,
                    1.0,
                )
                dists[i, j] = (iou_weight * dists[i, j]) + (bbox_weight * normalized_distance)
        return dists

    def _apply_field_position_costs(
        self,
        dists: np.ndarray,
        tracks: list[STrack],
        detections: list[STrack],
    ) -> np.ndarray:
        if not self.use_field_positions or dists.size == 0 or not tracks or not detections:
            return dists

        for i, track in enumerate(tracks):
            track_class = getattr(track, "class_name", None)
            if not self._uses_field_positions_for_class(track_class):
                continue

            track_position = field_position_to_array(getattr(track, "field_position", None))
            max_distance = max(self._track_distance_gate(track), 1e-6)
            for j, det in enumerate(detections):
                if getattr(det, "class_name", None) != track_class:
                    continue
                det_position = field_position_to_array(getattr(det, "field_position", None))
                if track_position is None or det_position is None:
                    if self.use_field_position_as_primary_cost:
                        dists[i, j] += 1000.0
                    continue
                field_distance = float(np.linalg.norm(track_position - det_position))
                if field_distance > max_distance:
                    dists[i, j] += 1000.0
                    continue
                normalized_distance = min(field_distance / max_distance, 1.0)
                if self.use_field_position_as_primary_cost:
                    dists[i, j] = normalized_distance
                else:
                    dists[i, j] += self.field_distance_weight * normalized_distance
        return dists

    def _team_mismatch_pair_penalty(self, track: STrack, det: STrack) -> float:
        if self.team_mismatch_penalty <= 0.0:
            return 0.0
        if getattr(track, "equipo", None) == getattr(det, "equipo", None):
            return 0.0
        return self.team_mismatch_penalty

    def _build_association_costs(
        self,
        tracks: list[STrack],
        detections: list[STrack],
        *,
        include_team_penalty: bool,
        include_fuse_score: bool,
        use_field_positions: bool,
    ) -> dict[str, np.ndarray]:
        iou_costs = matching.iou_distance(tracks, detections)
        after_bbox_costs = self._apply_bbox_center_costs(iou_costs.copy(), tracks, detections)
        team_penalties = np.zeros_like(after_bbox_costs, dtype=np.float32)
        class_penalties = np.zeros_like(after_bbox_costs, dtype=np.float32)
        bbox_size_penalties = np.zeros_like(after_bbox_costs, dtype=np.float32)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                if include_team_penalty:
                    team_penalties[i, j] = self._team_mismatch_pair_penalty(track, det)
                class_penalties[i, j] = self._class_mismatch_pair_penalty(track, det)
                bbox_size_penalties[i, j] = self._bbox_size_pair_penalty(track, det)

        after_penalty_costs = after_bbox_costs + team_penalties + class_penalties + bbox_size_penalties
        after_field_costs = (
            self._apply_field_position_costs(after_penalty_costs.copy(), tracks, detections)
            if use_field_positions
            else after_penalty_costs.copy()
        )
        after_fuse_costs = (
            matching.fuse_score(after_field_costs.copy(), detections)
            if include_fuse_score
            else after_field_costs.copy()
        )
        return {
            "iou_costs": iou_costs,
            "after_bbox_costs": after_bbox_costs,
            "team_penalties": team_penalties,
            "class_penalties": class_penalties,
            "bbox_size_penalties": bbox_size_penalties,
            "after_field_costs": after_field_costs,
            "after_fuse_costs": after_fuse_costs,
            "final_costs": self._apply_lost_time_penalty(after_fuse_costs.copy(), tracks),
        }
