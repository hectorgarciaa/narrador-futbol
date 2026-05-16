import numpy as np
from typing import Optional

from supervision.detection.core import Detections
from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter
from supervision.tracker.byte_tracker.single_object_track import STrack, TrackState
from supervision.tracker.byte_tracker.utils import IdCounter

from .debug_tools import ByteTrackDebugTools
from .association_costs import ByteTrackAssociationCosts
from .detection_metadata import ByteTrackDetectionMetadata
from .new_track_filter import ByteTrackNewTrackFilter
from .pipeline_steps import ByteTrackPipelineSteps
from .utils import joint_tracks


class ByteTrack(
    ByteTrackDebugTools,
    ByteTrackAssociationCosts,
    ByteTrackDetectionMetadata,
    ByteTrackNewTrackFilter,
    ByteTrackPipelineSteps,
):
    def __init__(
        self,
        track_activation_threshold: float = 0.10,
        low_conf_threshold: float = 0.01,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
        second_match_threshold: float = 0.7,
        unconfirmed_match_threshold: float = 0.8,
        field_position_classes: Optional[list[str]] = None,
        field_distance_gate_m: float = 8.0,
        field_distance_gate_max_lost_frames: Optional[int] = None,
        field_distance_gate_cap_m: Optional[float] = None,
        field_distance_decay_per_frame: float = 0.0,
        lost_time_penalty_weight: float = 0.0,
        lost_time_penalty_max_frames: int = 10,
        bbox_center_distance_gate_px: float = 120.0,
        bbox_center_distance_gate_max_lost_frames: Optional[int] = None,
        bbox_center_distance_gate_cap_px: Optional[float] = None,
        class_vote_weight_relabel: float = 1.0,
        class_vote_weight_yolo: float = 1.0,
        class_consensus_switch_margin: float = 2.0,
        bbox_height_ratio_threshold: float = 0.20,
        bbox_width_ratio_threshold: float = 0.30,
        team_vote_weight: float = 1.0,
        team_consensus_switch_margin: float = 2.0,
        new_track_active_overlap_iou: float = 0.0,
        new_track_unconfirmed_overlap_iou: float = 0.0,
        new_track_candidate_overlap_iou: float = 0.0,
    ):
        self.track_activation_threshold = float(track_activation_threshold)
        self.low_conf_threshold = float(max(0.0, low_conf_threshold))
        self.minimum_matching_threshold = float(minimum_matching_threshold)
        self.second_match_threshold = float(second_match_threshold)
        self.unconfirmed_match_threshold = float(unconfirmed_match_threshold)
        self.field_position_classes = frozenset(
            field_position_classes or ["player", "goalkeeper"]
        )
        self.field_distance_gate_m = float(field_distance_gate_m)
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
        self.field_distance_decay_per_frame = float(field_distance_decay_per_frame)
        if (
            not np.isfinite(self.field_distance_decay_per_frame)
            or self.field_distance_decay_per_frame < 0.0
        ):
            self.field_distance_decay_per_frame = 0.0
        self.lost_time_penalty_weight = float(max(0.0, lost_time_penalty_weight))
        self.lost_time_penalty_max_frames = max(1, int(lost_time_penalty_max_frames))
        self.bbox_center_distance_gate_px = float(max(1.0, bbox_center_distance_gate_px))
        if bbox_center_distance_gate_max_lost_frames is None:
            self.bbox_center_distance_gate_max_lost_frames = None
        else:
            self.bbox_center_distance_gate_max_lost_frames = max(
                1, int(bbox_center_distance_gate_max_lost_frames)
            )
        if bbox_center_distance_gate_cap_px is None:
            self.bbox_center_distance_gate_cap_px = None
        else:
            gate_cap_px = float(bbox_center_distance_gate_cap_px)
            self.bbox_center_distance_gate_cap_px = (
                gate_cap_px if np.isfinite(gate_cap_px) and gate_cap_px > 0.0 else None
            )
        self.class_vote_weight_relabel = float(max(0.0, class_vote_weight_relabel))
        self.class_vote_weight_yolo = float(max(0.0, class_vote_weight_yolo))
        self.class_consensus_switch_margin = float(max(0.0, class_consensus_switch_margin))
        self.bbox_height_ratio_threshold = float(max(0.0, bbox_height_ratio_threshold))
        self.bbox_width_ratio_threshold = float(max(0.0, bbox_width_ratio_threshold))
        self.team_vote_weight = float(max(0.0, team_vote_weight))
        self.team_consensus_switch_margin = float(max(0.0, team_consensus_switch_margin))
        self.new_track_active_overlap_iou = float(
            min(1.0, max(0.0, new_track_active_overlap_iou))
        )
        self.new_track_unconfirmed_overlap_iou = float(
            min(1.0, max(0.0, new_track_unconfirmed_overlap_iou))
        )
        self.new_track_candidate_overlap_iou = float(
            min(1.0, max(0.0, new_track_candidate_overlap_iou))
        )
        self.frame_id = 0
        self.max_time_lost = max(1, int(lost_track_buffer))
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.large_match_cost = 1e6
        self.kalman_filter = KalmanFilter()
        self.shared_kalman = KalmanFilter()

        self.tracked_tracks: list[STrack] = []
        self.lost_tracks: list[STrack] = []
        self.removed_tracks: list[STrack] = []

        self.internal_id_counter = IdCounter()
        self.external_id_counter = IdCounter(start_id=1)
        self.collect_internal_matching_debug = False
        self.last_detection_debug_by_raw_idx = {}
        self.last_matching_debug = {}

    def update_with_detections(self, detections: Detections) -> Detections:
        data = detections.data

        bboxes = detections.xyxy
        scores = detections.confidence
        team_labels = data["team"]
        class_td_labels = data["class_td"]
        class_yolo_labels = data["class_yolo"]
        field_positions = data["field_position"]
        shirt_colors = data["shirt_color"]
        bbox_sizes = data["bbox_size"]
        raw_det_indices = np.asarray(data["raw_det_idx"], dtype=np.int32).reshape(-1)

        tracks = self.update_with_tensors(
            bboxes=bboxes,
            scores=scores,
            team_labels=team_labels,
            class_labels=class_td_labels,
            field_positions=field_positions,
            yolo_class_labels=class_yolo_labels,
            shirt_colors=shirt_colors,
            bbox_sizes=bbox_sizes,
            raw_det_indices=raw_det_indices,
        )
        track_by_raw_idx = {}
        for track in tracks:
            raw_det_idx = getattr(track, "raw_det_idx", None)
            if raw_det_idx is not None:
                track_by_raw_idx[int(raw_det_idx)] = track

        tracker_ids = np.full(len(detections), -1, dtype=int)
        class_tracker_by_detection = np.array([None] * len(detections), dtype=object)
        for i_detection, raw_det_idx in enumerate(raw_det_indices.tolist()):
            track = track_by_raw_idx.get(int(raw_det_idx))
            if track is None:
                continue
            tracker_ids[i_detection] = int(track.external_track_id)
            class_tracker_by_detection[i_detection] = getattr(track, "class_name", None)

        detections.tracker_id = tracker_ids
        if detections.data is None:
            detections.data = {}
        detections.data["class_tracker"] = class_tracker_by_detection
        return detections[detections.tracker_id != -1]

    def reset(self) -> None:
        self.frame_id = 0
        self.internal_id_counter.reset()
        self.external_id_counter.reset()
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.last_detection_debug_by_raw_idx = {}
        self.last_matching_debug = {}

    def update_with_tensors(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        team_labels,
        class_labels=None,
        field_positions=None,
        yolo_class_labels=None,
        shirt_colors=None,
        bbox_sizes=None,
        raw_det_indices=None,
    ) -> list[STrack]:
        self.frame_id += 1
        self.last_detection_debug_by_raw_idx = {}
        self.last_matching_debug = {}
        activated_tracks = []
        refound_tracks = []
        lost_tracks = []
        removed_tracks = []

        high_conf_mask = scores >= self.track_activation_threshold
        low_conf_mask = np.logical_and(
            scores >= self.low_conf_threshold,
            scores < self.track_activation_threshold,
        )
        keep_indices = np.where(high_conf_mask)[0]
        second_indices = np.where(low_conf_mask)[0]
        raw_det_indices = (
            np.asarray(raw_det_indices, dtype=np.int32).reshape(-1)
            if raw_det_indices is not None
            else np.arange(len(bboxes), dtype=np.int32)
        )

        high_conf_detections = self._build_detection_batch(
            keep_indices,
            bboxes[high_conf_mask],
            scores[high_conf_mask],
            team_labels=team_labels,
            class_labels=class_labels,
            yolo_class_labels=yolo_class_labels,
            field_positions=field_positions,
            shirt_colors=shirt_colors,
            bbox_sizes=bbox_sizes,
            raw_det_indices=raw_det_indices,
        )
        low_conf_detections = self._build_detection_batch(
            second_indices,
            bboxes[low_conf_mask],
            scores[low_conf_mask],
            team_labels=team_labels,
            class_labels=class_labels,
            yolo_class_labels=yolo_class_labels,
            field_positions=field_positions,
            shirt_colors=shirt_colors,
            bbox_sizes=bbox_sizes,
            raw_det_indices=raw_det_indices,
        )

        tracked_tracks, unconfirmed = self._partition_existing_tracks()
        strack_pool = joint_tracks(tracked_tracks, self.lost_tracks)
        STrack.multi_predict(strack_pool, self.shared_kalman)

        remaining_pool, remaining_high = self._associate_confirmed_phase(
            strack_pool,
            high_conf_detections,
            phase_name="high_iou",
            cost_mode="iou",
            threshold=self.minimum_matching_threshold,
            activated_tracks=activated_tracks,
            refound_tracks=refound_tracks,
            reason="matched_existing_track_high_iou",
        )
        remaining_pool, remaining_low = self._associate_confirmed_phase(
            remaining_pool,
            low_conf_detections,
            phase_name="low_iou",
            cost_mode="iou",
            threshold=self.second_match_threshold,
            activated_tracks=activated_tracks,
            refound_tracks=refound_tracks,
            reason="matched_existing_track_low_iou",
        )

        remaining_unconfirmed, remaining_high, accepted_unconfirmed_iou = (
            self._associate_unconfirmed_phase(
                unconfirmed,
                remaining_high,
                phase_name="unconfirmed_iou",
                cost_mode="iou",
                threshold=self.unconfirmed_match_threshold,
                activated_tracks=activated_tracks,
                reason="matched_unconfirmed_track_iou",
            )
        )

        remaining_pool, remaining_high = self._associate_confirmed_phase(
            remaining_pool,
            remaining_high,
            phase_name="high_bbox",
            cost_mode="bbox_center",
            threshold=None,
            activated_tracks=activated_tracks,
            refound_tracks=refound_tracks,
            reason="matched_existing_track_high_bbox",
        )
        remaining_pool, remaining_low = self._associate_confirmed_phase(
            remaining_pool,
            remaining_low,
            phase_name="low_bbox",
            cost_mode="bbox_center",
            threshold=None,
            activated_tracks=activated_tracks,
            refound_tracks=refound_tracks,
            reason="matched_existing_track_low_bbox",
        )

        remaining_unconfirmed, remaining_high, accepted_unconfirmed_bbox = (
            self._associate_unconfirmed_phase(
                remaining_unconfirmed,
                remaining_high,
                phase_name="unconfirmed_bbox",
                cost_mode="bbox_center",
                threshold=None,
                activated_tracks=activated_tracks,
                reason="matched_unconfirmed_track_bbox",
            )
        )

        for track in remaining_unconfirmed:
            track.state = TrackState.Removed
            removed_tracks.append(track)

        for track in remaining_pool:
            if track.state != TrackState.Lost:
                track.state = TrackState.Lost
                lost_tracks.append(track)

        surviving_unconfirmed_tracks = [
            track
            for track in (accepted_unconfirmed_iou + accepted_unconfirmed_bbox)
            if getattr(track, "state", None) == TrackState.Tracked
            and not bool(getattr(track, "is_activated", False))
        ]

        self._activate_new_tracks(
            remaining_high,
            surviving_unconfirmed_tracks,
            activated_tracks=activated_tracks,
            refound_tracks=refound_tracks,
        )
        self._finalize_frame_state(
            activated_tracks=activated_tracks,
            refound_tracks=refound_tracks,
            lost_tracks=lost_tracks,
            removed_tracks=removed_tracks,
        )
        return [track for track in self.tracked_tracks if track.is_activated]
