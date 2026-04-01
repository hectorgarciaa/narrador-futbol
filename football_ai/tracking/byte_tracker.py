import logging

import numpy as np
from types import SimpleNamespace
from typing import Dict, Optional

from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import box_iou_batch
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter
from supervision.tracker.byte_tracker.single_object_track import STrack, TrackState
from supervision.tracker.byte_tracker.utils import IdCounter

logger = logging.getLogger(__name__)


class ByteTrack:
    """
    Initialize the ByteTrack object.

    <video controls>
        <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/annotate-video-with-traces.mp4" type="video/mp4">
    </video>

    Parameters:
        track_activation_threshold (float): Detection confidence threshold
            for track activation. Increasing track_activation_threshold improves accuracy
            and stability but might miss true detections. Decreasing it increases
            completeness but risks introducing noise and instability.
        lost_track_buffer (int): Number of frames to buffer when a track is lost.
            Increasing lost_track_buffer enhances occlusion handling, significantly
            reducing the likelihood of track fragmentation or disappearance caused
            by brief detection gaps.
        minimum_matching_threshold (float): Threshold for matching tracks with detections.
            Increasing minimum_matching_threshold improves accuracy but risks fragmentation.
            Decreasing it improves completeness but risks false positives and drift.
        frame_rate (int): The frame rate of the video.
        minimum_consecutive_frames (int): Number of consecutive frames that an object must
            be tracked before it is considered a 'valid' track.
            Increasing minimum_consecutive_frames prevents the creation of accidental tracks from
            false detection or double detection, but risks missing shorter tracks.
    """  # noqa: E501 // docs

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
        max_tracks_per_class: Optional[Dict[str, int]] = None,
        team_mismatch_penalty: float = 1000.0,
        second_match_threshold: float = 0.7,
        unconfirmed_match_threshold: float = 0.8,
        use_field_positions: bool = False,
        field_position_classes: Optional[list[str]] = None,
        field_distance_gate_m: float = 8.0,
        field_distance_weight: float = 0.25,
        field_distance_gate_max_lost_frames: Optional[int] = None,
        field_distance_gate_cap_m: Optional[float] = None,
        field_distance_growth_mode: str = "power",
        field_distance_lost_exponent: float = 1.0,
        field_distance_decay_per_frame: float = 0.0,
        lost_time_penalty_weight: float = 0.0,
        lost_time_penalty_max_frames: int = 10,
        use_field_position_as_primary_cost: bool = False,
        use_bbox_center_for_matching: bool = True,
        bbox_center_distance_weight: float = 0.5,
        bbox_center_distance_gate_px: float = 120.0,
        class_mismatch_penalty: float = 1000.0,
        class_mismatch_relaxed_penalty: float = 180.0,
        allow_class_remap_when_signals_disagree: bool = True,
        class_vote_weight_relabel: float = 1.0,
        class_vote_weight_yolo: float = 1.0,
        class_consensus_switch_margin: float = 2.0,
        shirt_color_distance_weight: float = 0.0,
        shirt_color_distance_gate: float = 45.0,
        team_vote_weight: float = 1.0,
        team_consensus_switch_margin: float = 2.0,
        new_track_active_overlap_iou: float = 0.0,
    ):
        self.track_activation_threshold = track_activation_threshold
        self.minimum_matching_threshold = minimum_matching_threshold
        self.max_tracks_per_class = max_tracks_per_class or {}
        self.team_mismatch_penalty = team_mismatch_penalty
        self.second_match_threshold = second_match_threshold
        self.unconfirmed_match_threshold = unconfirmed_match_threshold
        self.use_field_positions = bool(use_field_positions)
        self.field_position_classes = frozenset(
            field_position_classes or ["player", "goalkeeper"]
        )
        self.field_distance_gate_m = float(field_distance_gate_m)
        self.field_distance_weight = float(field_distance_weight)
        if field_distance_gate_max_lost_frames is None:
            self.field_distance_gate_max_lost_frames = None
        else:
            self.field_distance_gate_max_lost_frames = max(
                1, int(field_distance_gate_max_lost_frames)
            )
        if field_distance_gate_cap_m is None:
            self.field_distance_gate_cap_m = None
        else:
            gate_cap = float(field_distance_gate_cap_m)
            self.field_distance_gate_cap_m = (
                gate_cap if np.isfinite(gate_cap) and gate_cap > 0.0 else None
            )
        self.field_distance_growth_mode = str(
            field_distance_growth_mode or "power"
        ).strip().lower()
        if self.field_distance_growth_mode not in {"power", "linear_decay"}:
            self.field_distance_growth_mode = "power"
        self.field_distance_lost_exponent = float(field_distance_lost_exponent)
        if (
            not np.isfinite(self.field_distance_lost_exponent)
            or self.field_distance_lost_exponent <= 0.0
        ):
            self.field_distance_lost_exponent = 1.0
        self.field_distance_decay_per_frame = float(field_distance_decay_per_frame)
        if (
            not np.isfinite(self.field_distance_decay_per_frame)
            or self.field_distance_decay_per_frame < 0.0
        ):
            self.field_distance_decay_per_frame = 0.0
        self.lost_time_penalty_weight = float(max(0.0, lost_time_penalty_weight))
        self.lost_time_penalty_max_frames = max(1, int(lost_time_penalty_max_frames))
        self.use_field_position_as_primary_cost = bool(
            use_field_position_as_primary_cost
        )
        self.use_bbox_center_for_matching = bool(use_bbox_center_for_matching)
        self.bbox_center_distance_weight = float(
            min(1.0, max(0.0, bbox_center_distance_weight))
        )
        self.bbox_center_distance_gate_px = float(max(1.0, bbox_center_distance_gate_px))
        self.class_mismatch_penalty = float(max(0.0, class_mismatch_penalty))
        relaxed_penalty = float(max(0.0, class_mismatch_relaxed_penalty))
        self.class_mismatch_relaxed_penalty = min(
            self.class_mismatch_penalty,
            relaxed_penalty,
        )
        self.allow_class_remap_when_signals_disagree = bool(
            allow_class_remap_when_signals_disagree
        )
        self.class_vote_weight_relabel = float(max(0.0, class_vote_weight_relabel))
        self.class_vote_weight_yolo = float(max(0.0, class_vote_weight_yolo))
        self.class_consensus_switch_margin = float(max(0.0, class_consensus_switch_margin))
        self.shirt_color_distance_weight = float(max(0.0, shirt_color_distance_weight))
        self.shirt_color_distance_gate = float(max(1.0, shirt_color_distance_gate))
        self.team_vote_weight = float(max(0.0, team_vote_weight))
        self.team_consensus_switch_margin = float(max(0.0, team_consensus_switch_margin))
        self.new_track_active_overlap_iou = float(
            min(1.0, max(0.0, new_track_active_overlap_iou))
        )
        self.assigned_track_ids_by_class = {
            class_name: set() for class_name in self.max_tracks_per_class
        }

        self.frame_id = 0
        self.det_thresh = self.track_activation_threshold + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.kalman_filter = KalmanFilter()
        self.shared_kalman = KalmanFilter()

        self.tracked_tracks: list[STrack] = []
        self.lost_tracks: list[STrack] = []
        self.removed_tracks: list[STrack] = []

        # Warning, possible bug: If you also set internal_id to start at 1,
        # all traces will be connected across objects.
        self.internal_id_counter = IdCounter()
        self.external_id_counter = IdCounter(start_id=1)

    def _can_activate_track(self, class_name: Optional[str]) -> bool:
        # Internal class limits are disabled: track creation is never blocked by class count.
        return True

    def _register_track(self, track: STrack) -> None:
        # Kept as a no-op for compatibility with existing call sites.
        return

    @staticmethod
    def _track_tlbr(track: STrack) -> Optional[np.ndarray]:
        tlbr = getattr(track, "tlbr", None)
        if tlbr is None:
            return None
        tlbr = np.asarray(tlbr, dtype=np.float32).reshape(-1)
        if tlbr.size < 4 or not np.all(np.isfinite(tlbr[:4])):
            return None
        return tlbr[:4]

    @staticmethod
    def _tlbr_iou(box_a: Optional[np.ndarray], box_b: Optional[np.ndarray]) -> float:
        if box_a is None or box_b is None:
            return 0.0
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
            0.0, float(box_a[3] - box_a[1])
        )
        area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
            0.0, float(box_b[3] - box_b[1])
        )
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)

    def _filter_new_track_candidate_indices(
        self,
        detections: list[STrack],
        candidate_indices: list[int],
        active_tracked_pool: list[STrack],
        reference_unconfirmed_pool: Optional[list[STrack]] = None,
    ) -> list[int]:
        # Rule 1: if candidate overlaps an already-active track above the configured
        # threshold, reject it.
        min_overlap_iou = 1e-6
        active_overlap_iou = max(min_overlap_iou, float(self.new_track_active_overlap_iou))
        valid_indices = []
        for idx in candidate_indices:
            candidate_box = self._track_tlbr(detections[idx])
            overlaps_active = False
            for active_track in active_tracked_pool:
                if self._tlbr_iou(candidate_box, self._track_tlbr(active_track)) > active_overlap_iou:
                    overlaps_active = True
                    break
            if not overlaps_active:
                valid_indices.append(idx)

        # Rule 2: if candidate overlaps any previous unconfirmed track even minimally,
        # reject it.
        unconfirmed_pool = reference_unconfirmed_pool or []
        valid_after_unconfirmed = []
        for idx in valid_indices:
            candidate_box = self._track_tlbr(detections[idx])
            overlaps_unconfirmed = any(
                self._tlbr_iou(candidate_box, self._track_tlbr(unconfirmed_track)) > min_overlap_iou
                for unconfirmed_track in unconfirmed_pool
            )
            if overlaps_unconfirmed:
                continue
            valid_after_unconfirmed.append(idx)

        # Rule 3: among remaining candidates, suppress any overlap, keeping only
        # the highest-confidence one.
        # keeping only the highest-confidence one.
        valid_after_unconfirmed.sort(
            key=lambda i: float(getattr(detections[i], "score", 0.0)),
            reverse=True,
        )
        selected_indices = []
        selected_boxes = []
        for idx in valid_after_unconfirmed:
            candidate_box = self._track_tlbr(detections[idx])
            overlaps_selected = any(
                self._tlbr_iou(candidate_box, selected_box) > min_overlap_iou
                for selected_box in selected_boxes
            )
            if overlaps_selected:
                continue
            selected_indices.append(idx)
            selected_boxes.append(candidate_box)
        return selected_indices

    @staticmethod
    def _track_priority_for_unconfirmed(track: STrack) -> tuple[float, float]:
        """Higher priority wins when suppressing duplicated unconfirmed tracks."""
        age = float(int(getattr(track, "frame_id", 0)) - int(getattr(track, "start_frame", 0)))
        score = float(getattr(track, "score", 0.0) or 0.0)
        return age, score

    def _filter_unconfirmed_tracks(
        self,
        unconfirmed_tracks: list[STrack],
        active_tracked_pool: list[STrack],
    ) -> tuple[list[STrack], list[STrack]]:
        """
        1) Drop any unconfirmed track with any overlap against active tracks (IoU > 0).
        2) Suppress duplicate unconfirmed tracks among themselves with any overlap.
        """
        if not unconfirmed_tracks:
            return [], []

        removed_tracks = []
        min_overlap_iou_active = 1e-6

        valid_unconfirmed = []
        for track in unconfirmed_tracks:
            track_box = self._track_tlbr(track)
            overlaps_active = any(
                self._tlbr_iou(track_box, self._track_tlbr(active_track)) > min_overlap_iou_active
                for active_track in active_tracked_pool
            )
            if overlaps_active:
                track.state = TrackState.Removed
                removed_tracks.append(track)
                continue
            valid_unconfirmed.append(track)

        valid_unconfirmed.sort(
            key=self._track_priority_for_unconfirmed,
            reverse=True,
        )
        deduped_unconfirmed = []
        kept_boxes = []
        for track in valid_unconfirmed:
            track_box = self._track_tlbr(track)
            overlaps_kept = any(
                self._tlbr_iou(track_box, kept_box) > min_overlap_iou_active
                for kept_box in kept_boxes
            )
            if overlaps_kept:
                track.state = TrackState.Removed
                removed_tracks.append(track)
                continue
            deduped_unconfirmed.append(track)
            kept_boxes.append(track_box)

        return deduped_unconfirmed, removed_tracks

    @staticmethod
    def _field_position_to_array(field_position) -> Optional[np.ndarray]:
        if field_position is None:
            return None
        field_position = np.asarray(field_position, dtype=np.float32).reshape(-1)
        if field_position.size < 2 or not np.all(np.isfinite(field_position[:2])):
            return None
        return field_position[:2]

    @staticmethod
    def _normalize_class_label(class_name) -> Optional[str]:
        token = str(class_name or "").strip().lower()
        if not token:
            return None
        aliases = {
            "player": "player",
            "players": "player",
            "goalkeeper": "goalkeeper",
            "gk": "goalkeeper",
            "keeper": "goalkeeper",
            "referee": "referee",
            "ref": "referee",
            "refs": "referee",
            "ball": "ball",
            "balls": "ball",
        }
        return aliases.get(token, token)

    def _normalize_class_labels(self, labels):
        if labels is None:
            return None
        return [self._normalize_class_label(label) for label in labels]

    @staticmethod
    def _shirt_color_to_array(shirt_color) -> Optional[np.ndarray]:
        if shirt_color is None:
            return None
        array = np.asarray(shirt_color, dtype=np.float32).reshape(-1)
        if array.size < 3 or not np.all(np.isfinite(array[:3])):
            return None
        return array[:3]

    @staticmethod
    def _person_like_class(class_name: Optional[str]) -> bool:
        return class_name in {"player", "goalkeeper", "referee"}

    def _resolve_detection_class(
        self,
        class_name_relabel: Optional[str],
        class_name_yolo: Optional[str],
    ) -> Optional[str]:
        if class_name_relabel is not None:
            return class_name_relabel
        return class_name_yolo

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

        # Regla pedida:
        # 1) team detector == track -> 0
        # 2) team detector != track y yolo == track -> relajada
        # 3) team detector != track y yolo != track -> fuerte
        if det_class_team is not None and det_class_team == track_class:
            return 0.0

        if (
            det_class_team is not None
            and det_class_team != track_class
            and det_class_yolo is not None
            and det_class_yolo == track_class
            and self.allow_class_remap_when_signals_disagree
        ):
            return self.class_mismatch_relaxed_penalty

        return self.class_mismatch_penalty

    def _shirt_color_pair_penalty(self, track: STrack, det: STrack) -> float:
        if self.shirt_color_distance_weight <= 0.0:
            return 0.0
        track_class = getattr(track, "class_name", None)
        det_class = getattr(det, "class_name", None)
        if not self._person_like_class(track_class) or not self._person_like_class(det_class):
            return 0.0

        track_color = self._shirt_color_to_array(getattr(track, "shirt_color", None))
        det_color = self._shirt_color_to_array(getattr(det, "shirt_color", None))
        if track_color is None or det_color is None:
            return 0.0

        gate = max(self.shirt_color_distance_gate, 1e-6)
        color_distance = float(np.linalg.norm(track_color - det_color))
        normalized = min(color_distance / gate, 1.0)
        return self.shirt_color_distance_weight * normalized

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
        best_score = float(votes.get(best_class, 0.0))
        current_class = getattr(track, "class_name", None)
        current_score = float(votes.get(current_class, 0.0))

        if current_class is None:
            track.class_name = best_class
            return

        if (
            best_class != current_class
            and best_score >= (current_score + self.class_consensus_switch_margin)
        ):
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
        if not votes:
            return

        best_team = max(votes.items(), key=lambda item: item[1])[0]
        best_score = float(votes.get(best_team, 0.0))
        current_score = float(votes.get(current_team, 0.0))
        if (
            best_team != current_team
            and best_score >= (current_score + self.team_consensus_switch_margin)
        ):
            track.equipo = best_team

    def _uses_field_positions_for_class(self, class_name: Optional[str]) -> bool:
        if not self.use_field_positions:
            return False
        return class_name in self.field_position_classes

    def _track_distance_gate(self, track: STrack) -> float:
        lost_frames = max(1, self.frame_id - int(getattr(track, "frame_id", self.frame_id)))
        if self.field_distance_gate_max_lost_frames is not None:
            lost_frames = min(lost_frames, self.field_distance_gate_max_lost_frames)
        if self.field_distance_growth_mode == "linear_decay":
            step_base = self.field_distance_gate_m
            decay = self.field_distance_decay_per_frame
            gate = 0.0
            for step_idx in range(lost_frames):
                step = step_base - (decay * step_idx)
                if step <= 0.0:
                    break
                gate += step
            # Keep at least one-step gate for numerical safety.
            gate = max(step_base, gate)
        else:
            growth = float(lost_frames) ** self.field_distance_lost_exponent
            gate = self.field_distance_gate_m * growth

        if self.field_distance_gate_cap_m is not None:
            gate = min(gate, self.field_distance_gate_cap_m)
        return gate

    def _track_image_distance_gate(self, track: STrack) -> float:
        lost_frames = max(1, self.frame_id - int(getattr(track, "frame_id", self.frame_id)))
        return self.bbox_center_distance_gate_px * lost_frames

    def _track_missed_frames(self, track: STrack) -> int:
        # tracked in previous frame => 0 missed frames penalty
        return max(0, self.frame_id - int(getattr(track, "frame_id", self.frame_id)) - 1)

    def _apply_lost_time_penalty(
        self,
        dists: np.ndarray,
        tracks: list[STrack],
    ) -> np.ndarray:
        if (
            self.lost_time_penalty_weight <= 0.0
            or dists.size == 0
            or len(tracks) == 0
        ):
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

    @staticmethod
    def _bbox_center_from_tlbr(tlbr: np.ndarray) -> np.ndarray:
        tlbr = np.asarray(tlbr, dtype=np.float32).reshape(-1)
        if tlbr.size < 4 or not np.all(np.isfinite(tlbr[:4])):
            return np.array([np.nan, np.nan], dtype=np.float32)
        x1, y1, x2, y2 = tlbr[:4]
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

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
            or len(tracks) == 0
            or len(detections) == 0
        ):
            return dists

        bbox_weight = self.bbox_center_distance_weight
        iou_weight = 1.0 - bbox_weight

        for i, track in enumerate(tracks):
            track_center = self._bbox_center_from_tlbr(getattr(track, "tlbr", None))
            if not np.all(np.isfinite(track_center)):
                continue

            max_distance = max(self._track_image_distance_gate(track), 1e-6)
            track_class = getattr(track, "class_name", None)
            for j, det in enumerate(detections):
                det_class = getattr(det, "class_name", None)
                if (
                    track_class is not None
                    and det_class is not None
                    and track_class != det_class
                ):
                    continue
                det_center = self._bbox_center_from_tlbr(getattr(det, "tlbr", None))
                if not np.all(np.isfinite(det_center)):
                    continue

                center_distance = float(np.linalg.norm(track_center - det_center))
                normalized_distance = min(center_distance / max_distance, 1.0)
                dists[i, j] = (iou_weight * dists[i, j]) + (
                    bbox_weight * normalized_distance
                )

        return dists

    def _apply_field_position_costs(
        self,
        dists: np.ndarray,
        tracks: list[STrack],
        detections: list[STrack],
    ) -> np.ndarray:
        if (
            not self.use_field_positions
            or dists.size == 0
            or len(tracks) == 0
            or len(detections) == 0
        ):
            return dists

        for i, track in enumerate(tracks):
            track_class = getattr(track, "class_name", None)
            uses_field_for_track = self._uses_field_positions_for_class(track_class)
            track_position = self._field_position_to_array(
                getattr(track, "field_position", None)
            )
            if not uses_field_for_track:
                continue

            max_distance = max(self._track_distance_gate(track), 1e-6)
            for j, det in enumerate(detections):
                det_class = getattr(det, "class_name", None)
                if det_class != track_class:
                    continue
                det_position = self._field_position_to_array(
                    getattr(det, "field_position", None)
                )
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
        det_field_position = ByteTrack._field_position_to_array(
            getattr(det, "field_position", None)
        )
        if det_field_position is not None:
            track.field_position = det_field_position.copy()

    def update_with_detections(
        self,
        detections: Detections,
        team_labels,
        class_labels=None,
        field_positions=None,
        yolo_class_labels=None,
        shirt_colors=None,
    ) -> Detections:
        """
        Updates the tracker with the provided detections and returns the updated
        detection results.

        Args:
            detections (Detections): The detections to pass through the tracker.

        Example:
            ```python
            import supervision as sv
            from ultralytics import YOLO

            model = YOLO(<MODEL_PATH>)
            tracker = sv.ByteTrack()

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            def callback(frame: np.ndarray, index: int) -> np.ndarray:
                results = model(frame)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = tracker.update_with_detections(detections)

                labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

                annotated_frame = box_annotator.annotate(
                    scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels)
                return annotated_frame

            sv.process_video(
                source_path=<SOURCE_VIDEO_PATH>,
                target_path=<TARGET_VIDEO_PATH>,
                callback=callback
            )
            ```
        """
        tensors = np.hstack(
            (
                detections.xyxy,
                detections.confidence[:, np.newaxis],
            )
        )
        if class_labels is None and detections.data is not None:
            class_labels = detections.data.get("class")
        if yolo_class_labels is None and detections.data is not None:
            yolo_class_labels = detections.data.get("class_yolo")
        if field_positions is None and detections.data is not None:
            field_positions = detections.data.get("field_position")
        if shirt_colors is None and detections.data is not None:
            shirt_colors = detections.data.get("shirt_color")

        class_labels = self._normalize_class_labels(class_labels)
        yolo_class_labels = self._normalize_class_labels(yolo_class_labels)

        tracks = self.update_with_tensors(
            tensors=tensors,
            team_labels=team_labels,
            class_labels=class_labels,
            field_positions=field_positions,
            yolo_class_labels=yolo_class_labels,
            shirt_colors=shirt_colors,
        )

        if len(tracks) > 0:
            detection_bounding_boxes = np.asarray([det[:4] for det in tensors])
            track_bounding_boxes = np.asarray([track.tlbr for track in tracks])

            ious = box_iou_batch(detection_bounding_boxes, track_bounding_boxes)

            iou_costs = 1 - ious
            class_tracker_by_detection = np.array(
                [None] * len(detections),
                dtype=object,
            )
            if class_labels is not None:
                for i_detection, det_class in enumerate(class_labels):
                    det_yolo_class = (
                        yolo_class_labels[i_detection]
                        if yolo_class_labels is not None
                        and i_detection < len(yolo_class_labels)
                        else None
                    )
                    for i_track, track in enumerate(tracks):
                        track_like = SimpleNamespace(
                            class_name=getattr(track, "class_name", None),
                            class_name_team=getattr(track, "class_name_team", None),
                            class_name_yolo=getattr(track, "class_name_yolo", None),
                        )
                        det_like = SimpleNamespace(
                            class_name=det_class,
                            class_name_team=det_class,
                            class_name_yolo=det_yolo_class,
                        )
                        penalty = self._class_mismatch_pair_penalty(track_like, det_like)
                        if penalty > 0.0:
                            iou_costs[i_detection, i_track] += penalty
            if field_positions is not None:
                detection_field_positions = np.asarray(field_positions, dtype=np.float32)
                for i_detection, det_class in enumerate(class_labels if class_labels is not None else []):
                    if det_class not in self.field_position_classes:
                        continue
                    det_position = self._field_position_to_array(
                        detection_field_positions[i_detection]
                    )
                    if det_position is None:
                        continue
                    for i_track, track in enumerate(tracks):
                        track_class = getattr(track, "class_name", None)
                        if track_class != det_class:
                            continue
                        track_position = self._field_position_to_array(
                            getattr(track, "field_position", None)
                        )
                        if track_position is None:
                            continue
                        max_distance = max(self._track_distance_gate(track), 1e-6)
                        field_distance = float(np.linalg.norm(track_position - det_position))
                        if field_distance > max_distance:
                            iou_costs[i_detection, i_track] += 1000.0
                        else:
                            iou_costs[i_detection, i_track] += (
                                self.field_distance_weight
                                * min(field_distance / max_distance, 1.0)
                            )

            matches, _, _ = matching.linear_assignment(iou_costs, 0.5)
            detections.tracker_id = np.full(len(detections), -1, dtype=int)
            for i_detection, i_track in matches:
                detections.tracker_id[i_detection] = int(
                    tracks[i_track].external_track_id
                )
                class_tracker_by_detection[i_detection] = getattr(
                    tracks[i_track],
                    "class_name",
                    None,
                )
            if detections.data is None:
                detections.data = {}
            detections.data["class_tracker"] = class_tracker_by_detection

            return detections[detections.tracker_id != -1]

        else:
            detections = Detections.empty()
            detections.tracker_id = np.array([], dtype=int)

            return detections

    def reset(self) -> None:
        """
        Resets the internal state of the ByteTrack tracker.

        This method clears the tracking data, including tracked, lost,
        and removed tracks, as well as resetting the frame counter. It's
        particularly useful when processing multiple videos sequentially,
        ensuring the tracker starts with a clean state for each new video.
        """
        self.frame_id = 0
        self.internal_id_counter.reset()
        self.external_id_counter.reset()
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.assigned_track_ids_by_class = {
            class_name: set() for class_name in self.max_tracks_per_class
        }

    def update_with_tensors(
        self,
        tensors: np.ndarray,
        team_labels,
        class_labels=None,
        field_positions=None,
        yolo_class_labels=None,
        shirt_colors=None,
    ) -> list[STrack]:
        """
        Updates the tracker with the provided tensors and returns the updated tracks.

        Parameters:
            tensors: The new tensors to update with.

        Returns:
            List[STrack]: Updated tracks.
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = tensors[:, 4]
        bboxes = tensors[:, :4]

        remain_inds = scores > self.track_activation_threshold
        inds_low = scores > 0.1
        inds_high = scores < self.track_activation_threshold

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        keep_indices = np.where(remain_inds)[0]
        second_indices = np.where(inds_second)[0]
        normalized_class_labels = self._normalize_class_labels(class_labels)
        normalized_yolo_class_labels = self._normalize_class_labels(yolo_class_labels)

        # Filter team_labels with the same indices as high-confidence detections
        if team_labels is not None:
            remain_indices = np.where(remain_inds)[0]
            team_labels_keep = [team_labels[j] for j in remain_indices]
        else:
            team_labels_keep = None

        detections = []
        for keep_idx, tlbr, s in zip(keep_indices, dets, scores_keep):
            det = STrack(
                STrack.tlbr_to_tlwh(tlbr),
                s,
                self.minimum_consecutive_frames,
                self.shared_kalman,
                self.internal_id_counter,
                self.external_id_counter,
            )
            det.equipo = (
                team_labels[keep_idx]
                if team_labels is not None and keep_idx < len(team_labels)
                else None
            )
            det.class_name_team = (
                normalized_class_labels[keep_idx]
                if normalized_class_labels is not None and keep_idx < len(normalized_class_labels)
                else None
            )
            det.class_name_yolo = (
                normalized_yolo_class_labels[keep_idx]
                if normalized_yolo_class_labels is not None and keep_idx < len(normalized_yolo_class_labels)
                else None
            )
            det.class_name = self._resolve_detection_class(
                det.class_name_team,
                det.class_name_yolo,
            )
            det.field_position = (
                self._field_position_to_array(field_positions[keep_idx])
                if field_positions is not None and keep_idx < len(field_positions)
                else None
            )
            det.shirt_color = (
                self._shirt_color_to_array(shirt_colors[keep_idx])
                if shirt_colors is not None and keep_idx < len(shirt_colors)
                else None
            )
            
            detections.append(det)

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]

        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_tracks(tracked_stracks, self.lost_tracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.shared_kalman)
        dists = matching.iou_distance(strack_pool, detections)
        dists = self._apply_bbox_center_costs(dists, strack_pool, detections)

        # Penalizaciones en matching.
        for i, track in enumerate(strack_pool):
            for j, det in enumerate(detections):
                if hasattr(track, "equipo") and hasattr(det, "equipo"):
                    if track.equipo != det.equipo:
                        dists[i, j] += self.team_mismatch_penalty
                dists[i, j] += self._class_mismatch_pair_penalty(track, det)

        dists = self._apply_field_position_costs(dists, strack_pool, detections)
        dists = matching.fuse_score(dists, detections)
        dists = self._apply_lost_time_penalty(dists, strack_pool)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.minimum_matching_threshold
        )
        # Team update is handled via vote consensus in `_apply_track_metadata`.
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            self._apply_track_metadata(track, det)

            # --- Normal track update ---
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                self._apply_track_metadata(track, det)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                self._apply_track_metadata(track, det)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = []
            for second_idx, tlbr, score_second in zip(
                second_indices,
                dets_second,
                scores_second,
            ):
                det = STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    score_second,
                    self.minimum_consecutive_frames,
                    self.shared_kalman,
                    self.internal_id_counter,
                    self.external_id_counter,
                )
                det.equipo = (
                    team_labels[second_idx]
                    if team_labels is not None and second_idx < len(team_labels)
                    else None
                )
                det.class_name_team = (
                    normalized_class_labels[second_idx]
                    if normalized_class_labels is not None and second_idx < len(normalized_class_labels)
                    else None
                )
                det.class_name_yolo = (
                    normalized_yolo_class_labels[second_idx]
                    if normalized_yolo_class_labels is not None and second_idx < len(normalized_yolo_class_labels)
                    else None
                )
                det.class_name = self._resolve_detection_class(
                    det.class_name_team,
                    det.class_name_yolo,
                )
                det.field_position = (
                    self._field_position_to_array(field_positions[second_idx])
                    if field_positions is not None and second_idx < len(field_positions)
                    else None
                )
                det.shirt_color = (
                    self._shirt_color_to_array(shirt_colors[second_idx])
                    if shirt_colors is not None and second_idx < len(shirt_colors)
                    else None
                )
                detections_second.append(det)
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        dists = self._apply_bbox_center_costs(
            dists,
            r_tracked_stracks,
            detections_second,
        )
        for i, track in enumerate(r_tracked_stracks):
            for j, det in enumerate(detections_second):
                dists[i, j] += self._class_mismatch_pair_penalty(track, det)
        dists = self._apply_field_position_costs(
            dists,
            r_tracked_stracks,
            detections_second,
        )
        dists = self._apply_lost_time_penalty(dists, r_tracked_stracks)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=self.second_match_threshold
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                self._apply_track_metadata(track, det)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                self._apply_track_metadata(track, det)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.state = TrackState.Lost
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        active_for_unconfirmed_filter = []
        seen_active_ids_for_unconfirmed = set()
        for candidate_track in tracked_stracks + activated_starcks + refind_stracks:
            internal_track_id = getattr(candidate_track, "internal_track_id", None)
            if internal_track_id is not None:
                if internal_track_id in seen_active_ids_for_unconfirmed:
                    continue
                seen_active_ids_for_unconfirmed.add(internal_track_id)
            if getattr(candidate_track, "state", None) != TrackState.Tracked:
                continue
            if not bool(getattr(candidate_track, "is_activated", False)):
                continue
            active_for_unconfirmed_filter.append(candidate_track)

        unconfirmed, removed_unconfirmed_overlaps = self._filter_unconfirmed_tracks(
            unconfirmed_tracks=unconfirmed,
            active_tracked_pool=active_for_unconfirmed_filter,
        )
        removed_stracks.extend(removed_unconfirmed_overlaps)

        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        dists = self._apply_bbox_center_costs(dists, unconfirmed, detections)
        for i, track in enumerate(unconfirmed):
            for j, det in enumerate(detections):
                dists[i, j] += self._class_mismatch_pair_penalty(track, det)

        dists = self._apply_field_position_costs(dists, unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)
        dists = self._apply_lost_time_penalty(dists, unconfirmed)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=self.unconfirmed_match_threshold
        )
        matched_pairs = sorted(
            list(matches),
            key=lambda pair: float(getattr(detections[pair[1]], "score", 0.0)),
            reverse=True,
        )
        kept_unconfirmed_boxes = []
        accepted_unconfirmed_tracks = []
        min_overlap_iou_active = 1e-6
        for itracked, idet in matched_pairs:
            track = unconfirmed[itracked]
            det = detections[idet]
            candidate_box = self._track_tlbr(det)

            overlaps_active = any(
                self._tlbr_iou(candidate_box, self._track_tlbr(active_track)) > min_overlap_iou_active
                for active_track in active_for_unconfirmed_filter
            )
            if overlaps_active:
                track.state = TrackState.Removed
                removed_stracks.append(track)
                continue

            overlaps_kept_unconfirmed = any(
                self._tlbr_iou(candidate_box, kept_box) > min_overlap_iou_active
                for kept_box in kept_unconfirmed_boxes
            )
            if overlaps_kept_unconfirmed:
                track.state = TrackState.Removed
                removed_stracks.append(track)
                continue

            track.update(det, self.frame_id)
            self._apply_track_metadata(track, det)
            activated_starcks.append(track)
            accepted_unconfirmed_tracks.append(track)
            kept_unconfirmed_boxes.append(self._track_tlbr(track))

        surviving_unconfirmed_tracks = [
            track
            for track in accepted_unconfirmed_tracks
            if (
                getattr(track, "state", None) == TrackState.Tracked
                and not bool(getattr(track, "is_activated", False))
            )
        ]
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = TrackState.Removed
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        active_tracked_pool = []
        seen_internal_ids = set()
        for candidate_track in (
            list(self.tracked_tracks) + list(activated_starcks) + list(refind_stracks)
        ):
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

        candidate_indices = [
            inew for inew in u_detection if detections[inew].score >= self.det_thresh
        ]
        filtered_candidate_indices = self._filter_new_track_candidate_indices(
            detections=detections,
            candidate_indices=candidate_indices,
            active_tracked_pool=active_tracked_pool,
            reference_unconfirmed_pool=surviving_unconfirmed_tracks,
        )

        for inew in filtered_candidate_indices:
            track = detections[inew]
            if not self._can_activate_track(getattr(track, "class_name", None)):
                continue
            track.activate(self.kalman_filter, self.frame_id)
            self._apply_track_metadata(track, track)
            self._register_track(track)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = TrackState.Removed
                removed_stracks.append(track)

        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.Tracked
        ]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_starcks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks = removed_stracks
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        output_stracks = [track for track in self.tracked_tracks if track.is_activated]

        return output_stracks


def joint_tracks(
    track_list_a: list[STrack], track_list_b: list[STrack]
) -> list[STrack]:
    """
    Joins two lists of tracks, ensuring that the resulting list does not
    contain tracks with duplicate internal_track_id values.

    Parameters:
        track_list_a: First list of tracks (with internal_track_id attribute).
        track_list_b: Second list of tracks (with internal_track_id attribute).

    Returns:
        Combined list of tracks from track_list_a and track_list_b
            without duplicate internal_track_id values.
    """
    seen_track_ids = set()
    result = []

    for track in track_list_a + track_list_b:
        if track.internal_track_id not in seen_track_ids:
            seen_track_ids.add(track.internal_track_id)
            result.append(track)

    return result


def sub_tracks(track_list_a: list[STrack], track_list_b: list[STrack]) -> list[STrack]:
    """
    Returns a list of tracks from track_list_a after removing any tracks
    that share the same internal_track_id with tracks in track_list_b.

    Parameters:
        track_list_a: List of tracks (with internal_track_id attribute).
        track_list_b: List of tracks (with internal_track_id attribute) to
            be subtracted from track_list_a.
    Returns:
        List of remaining tracks from track_list_a after subtraction.
    """
    tracks = {track.internal_track_id: track for track in track_list_a}
    track_ids_b = {track.internal_track_id for track in track_list_b}

    for track_id in track_ids_b:
        tracks.pop(track_id, None)

    return list(tracks.values())


def remove_duplicate_tracks(
    tracks_a: list[STrack], tracks_b: list[STrack]
) -> tuple[list[STrack], list[STrack]]:
    pairwise_distance = matching.iou_distance(tracks_a, tracks_b)
    matching_pairs = np.where(pairwise_distance < 0.15)

    duplicates_a, duplicates_b = set(), set()
    for track_index_a, track_index_b in zip(*matching_pairs):
        time_a = tracks_a[track_index_a].frame_id - tracks_a[track_index_a].start_frame
        time_b = tracks_b[track_index_b].frame_id - tracks_b[track_index_b].start_frame
        if time_a > time_b:
            duplicates_b.add(track_index_b)
        else:
            duplicates_a.add(track_index_a)

    result_a = [
        track for index, track in enumerate(tracks_a) if index not in duplicates_a
    ]
    result_b = [
        track for index, track in enumerate(tracks_b) if index not in duplicates_b
    ]

    return result_a, result_b
