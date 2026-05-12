from __future__ import annotations

from typing import Optional

import numpy as np
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.single_object_track import STrack

from .utils import bbox_center_from_tlbr, bbox_size_from_tlbr, field_position_to_array


class ByteTrackAssociationCosts:
    def _uses_field_positions_for_class(self, class_name: Optional[str]) -> bool:
        return class_name in self.field_position_classes

    def _track_distance_gate(self, track: STrack) -> float:
        lost_frames = max(1, self.frame_id - int(getattr(track, "frame_id", self.frame_id)))
        if self.field_distance_gate_max_lost_frames is not None:
            lost_frames = min(lost_frames, self.field_distance_gate_max_lost_frames)
        if self.field_distance_growth_mode == "linear_decay":
            gate = 0.0
            for step_idx in range(lost_frames):
                step = self.field_distance_gate_m - (
                    self.field_distance_decay_per_frame * step_idx
                )
                if step <= 0.0:
                    break
                gate += step
            return min(
                max(self.field_distance_gate_m, gate),
                self.field_distance_gate_cap_m or np.inf,
            )

        gate = self.field_distance_gate_m * (
            float(lost_frames) ** self.field_distance_lost_exponent
        )
        if self.field_distance_gate_cap_m is not None:
            gate = min(gate, self.field_distance_gate_cap_m)
        return gate

    def _track_image_distance_gate(self, track: STrack) -> float:
        lost_frames = max(1, self.frame_id - int(getattr(track, "frame_id", self.frame_id)))
        return self.bbox_center_distance_gate_px * lost_frames

    def _track_missed_frames(self, track: STrack) -> int:
        return max(0, self.frame_id - int(getattr(track, "frame_id", self.frame_id)) - 1)

    def _lost_time_penalty_by_track(self, tracks: list[STrack]) -> np.ndarray:
        penalties = np.zeros(len(tracks), dtype=np.float32)
        if self.lost_time_penalty_weight <= 0.0 or not tracks:
            return penalties

        for index, track in enumerate(tracks):
            missed_frames = self._track_missed_frames(track)
            if missed_frames <= 0:
                continue
            penalty_factor = min(
                float(missed_frames) / float(self.lost_time_penalty_max_frames),
                1.0,
            )
            penalties[index] = self.lost_time_penalty_weight * penalty_factor
        return penalties

    def _class_gate_pass(self, track: STrack, det: STrack) -> bool:
        return getattr(track, "class_name", None) == getattr(det, "class_name", None)

    def _team_gate_pass(self, track: STrack, det: STrack) -> bool:
        if getattr(track, "class_name", None) != "player":
            return True
        track_team = getattr(track, "equipo", None)
        det_team = getattr(det, "equipo", None)
        if track_team is None or det_team is None:
            return True
        return track_team == det_team

    def _bbox_size_gate_pass(self, track: STrack, det: STrack) -> bool:
        track_width, track_height = bbox_size_from_tlbr(getattr(track, "tlbr", None))
        det_width, det_height = bbox_size_from_tlbr(getattr(det, "tlbr", None))
        if None in (track_width, track_height, det_width, det_height):
            return True

        if self.bbox_height_ratio_threshold > 0.0:
            min_height = track_height * (1.0 - self.bbox_height_ratio_threshold)
            max_height = track_height * (1.0 + self.bbox_height_ratio_threshold)
            if det_height < min_height or det_height > max_height:
                return False

        if self.bbox_width_ratio_threshold > 0.0:
            min_width = track_width * (1.0 - self.bbox_width_ratio_threshold)
            max_width = track_width * (1.0 + self.bbox_width_ratio_threshold)
            if det_width < min_width or det_width > max_width:
                return False

        return True

    def _field_gate_metrics(self, track: STrack, det: STrack) -> tuple[bool, float, float]:
        track_class = getattr(track, "class_name", None)
        if not self._uses_field_positions_for_class(track_class):
            return True, np.nan, np.nan

        track_position = field_position_to_array(getattr(track, "field_position", None))
        det_position = field_position_to_array(getattr(det, "field_position", None))
        if track_position is None or det_position is None:
            return True, np.nan, float(self._track_distance_gate(track))

        field_distance = float(np.linalg.norm(track_position - det_position))
        field_gate = float(self._track_distance_gate(track))
        return field_distance <= field_gate, field_distance, field_gate

    def _bbox_distance_metrics(self, track: STrack, det: STrack) -> tuple[bool, float, float]:
        track_center = bbox_center_from_tlbr(getattr(track, "tlbr", None))
        det_center = bbox_center_from_tlbr(getattr(det, "tlbr", None))
        bbox_gate = float(self._track_image_distance_gate(track))
        if not np.all(np.isfinite(track_center)) or not np.all(np.isfinite(det_center)):
            return False, np.nan, bbox_gate
        bbox_distance = float(np.linalg.norm(track_center - det_center))
        return bbox_distance <= bbox_gate, bbox_distance, bbox_gate

    def _build_phase_match_data(
        self,
        tracks: list[STrack],
        detections: list[STrack],
        *,
        phase_name: str,
        cost_mode: str,
    ) -> dict[str, object]:
        shape = (len(tracks), len(detections))
        iou_cost = matching.iou_distance(tracks, detections).astype(np.float32)
        iou = (1.0 - iou_cost).astype(np.float32)

        bbox_center_distance_px = np.full(shape, np.nan, dtype=np.float32)
        bbox_center_cost = np.full(shape, np.inf, dtype=np.float32)
        bbox_center_gate_px = np.full(shape, np.nan, dtype=np.float32)
        field_distance_m = np.full(shape, np.nan, dtype=np.float32)
        field_gate_m = np.full(shape, np.nan, dtype=np.float32)

        class_gate_pass = np.zeros(shape, dtype=bool)
        team_gate_pass = np.ones(shape, dtype=bool)
        bbox_size_gate_pass = np.ones(shape, dtype=bool)
        field_gate_pass = np.ones(shape, dtype=bool)
        bbox_distance_gate_pass = np.ones(shape, dtype=bool)

        lost_time_penalty_by_track = self._lost_time_penalty_by_track(tracks)
        lost_time_penalty = np.repeat(
            lost_time_penalty_by_track[:, np.newaxis],
            len(detections),
            axis=1,
        ).astype(np.float32)

        base_cost = np.full(shape, np.inf, dtype=np.float32)
        feasible_mask = np.zeros(shape, dtype=bool)

        for i, track in enumerate(tracks):
            bbox_gate_track = float(self._track_image_distance_gate(track))
            for j, det in enumerate(detections):
                class_ok = self._class_gate_pass(track, det)
                team_ok = self._team_gate_pass(track, det)
                bbox_size_ok = self._bbox_size_gate_pass(track, det)
                field_ok, field_distance, field_gate = self._field_gate_metrics(track, det)
                bbox_distance_ok, bbox_distance, bbox_gate = self._bbox_distance_metrics(
                    track, det
                )

                class_gate_pass[i, j] = class_ok
                team_gate_pass[i, j] = team_ok
                bbox_size_gate_pass[i, j] = bbox_size_ok
                field_gate_pass[i, j] = field_ok
                bbox_distance_gate_pass[i, j] = bbox_distance_ok
                field_distance_m[i, j] = np.float32(field_distance)
                field_gate_m[i, j] = np.float32(field_gate)
                bbox_center_distance_px[i, j] = np.float32(bbox_distance)
                bbox_center_gate_px[i, j] = np.float32(bbox_gate)

                if np.isfinite(bbox_distance):
                    bbox_center_cost[i, j] = min(bbox_distance / max(bbox_gate_track, 1e-6), 1.0)

                phase_gate_ok = False
                if cost_mode == "iou":
                    phase_gate_ok = float(iou[i, j]) > 0.0
                    base_cost[i, j] = iou_cost[i, j] + lost_time_penalty[i, j]
                    bbox_distance_gate_pass[i, j] = True
                elif cost_mode == "bbox_center":
                    phase_gate_ok = float(iou[i, j]) <= 0.0 and bbox_distance_ok
                    base_cost[i, j] = bbox_center_cost[i, j] + lost_time_penalty[i, j]
                else:
                    raise ValueError(f"Unsupported cost_mode={cost_mode!r} for phase {phase_name}")

                feasible_mask[i, j] = (
                    class_ok
                    and team_ok
                    and bbox_size_ok
                    and field_ok
                    and phase_gate_ok
                )

        return {
            "phase_name": phase_name,
            "cost_mode": cost_mode,
            "base_cost": base_cost,
            "feasible_mask": feasible_mask,
            "pair_metrics": {
                "iou": iou,
                "iou_cost": iou_cost,
                "bbox_center_distance_px": bbox_center_distance_px,
                "bbox_center_gate_px": bbox_center_gate_px,
                "bbox_center_cost": bbox_center_cost,
                "field_distance_m": field_distance_m,
                "field_gate_m": field_gate_m,
                "class_gate_pass": class_gate_pass,
                "team_gate_pass": team_gate_pass,
                "bbox_size_gate_pass": bbox_size_gate_pass,
                "field_gate_pass": field_gate_pass,
                "bbox_distance_gate_pass": bbox_distance_gate_pass,
                "lost_time_penalty": lost_time_penalty,
            },
        }
