import logging

import numpy as np
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
        field_distance_growth_mode: str = "power",
        field_distance_lost_exponent: float = 1.0,
        field_distance_decay_per_frame: float = 0.0,
        lost_time_penalty_weight: float = 0.0,
        lost_time_penalty_max_frames: int = 10,
        use_field_position_as_primary_cost: bool = False,
        use_bbox_center_for_matching: bool = True,
        bbox_center_distance_weight: float = 0.5,
        bbox_center_distance_gate_px: float = 120.0,
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
        if class_name is None:
            return True
        class_limit = self.max_tracks_per_class.get(class_name)
        if class_limit is None:
            return True
        assigned_ids = self.assigned_track_ids_by_class.setdefault(class_name, set())
        return len(assigned_ids) < class_limit

    def _register_track(self, track: STrack) -> None:
        class_name = getattr(track, "class_name", None)
        if class_name is None or class_name not in self.max_tracks_per_class:
            return
        self.assigned_track_ids_by_class.setdefault(class_name, set()).add(
            int(track.external_track_id)
        )

    @staticmethod
    def _field_position_to_array(field_position) -> Optional[np.ndarray]:
        if field_position is None:
            return None
        field_position = np.asarray(field_position, dtype=np.float32).reshape(-1)
        if field_position.size < 2 or not np.all(np.isfinite(field_position[:2])):
            return None
        return field_position[:2]

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
            return max(step_base, gate)

        growth = float(lost_frames) ** self.field_distance_lost_exponent
        return self.field_distance_gate_m * growth

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

    @staticmethod
    def _apply_track_metadata(track: STrack, det: STrack) -> None:
        if getattr(det, "equipo", None) is not None:
            track.equipo = det.equipo
        if getattr(track, "class_name", None) is None and getattr(
            det, "class_name", None
        ) is not None:
            track.class_name = det.class_name
        det_field_position = ByteTrack._field_position_to_array(
            getattr(det, "field_position", None)
        )
        if det_field_position is not None:
            track.field_position = det_field_position.copy()

    def update_with_detections(
        self, detections: Detections, team_labels, class_labels=None, field_positions=None
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
        if field_positions is None and detections.data is not None:
            field_positions = detections.data.get("field_position")

        tracks = self.update_with_tensors(
            tensors=tensors,
            team_labels=team_labels,
            class_labels=class_labels,
            field_positions=field_positions,
        )

        if len(tracks) > 0:
            detection_bounding_boxes = np.asarray([det[:4] for det in tensors])
            track_bounding_boxes = np.asarray([track.tlbr for track in tracks])

            ious = box_iou_batch(detection_bounding_boxes, track_bounding_boxes)

            iou_costs = 1 - ious
            if class_labels is not None:
                for i_detection, det_class in enumerate(class_labels):
                    for i_track, track in enumerate(tracks):
                        track_class = getattr(track, "class_name", None)
                        if track_class is not None and det_class != track_class:
                            iou_costs[i_detection, i_track] += 1000
            if field_positions is not None:
                detection_field_positions = np.asarray(field_positions, dtype=np.float32)
                for i_detection, det_class in enumerate(class_labels or []):
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
            det.class_name = (
                class_labels[keep_idx]
                if class_labels is not None and keep_idx < len(class_labels)
                else None
            )
            det.field_position = (
                self._field_position_to_array(field_positions[keep_idx])
                if field_positions is not None and keep_idx < len(field_positions)
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
                if hasattr(track, "class_name") and hasattr(det, "class_name"):
                    if track.class_name != det.class_name:
                        dists[i, j] += 1000  # evita cruces de IDs entre clases

        dists = self._apply_field_position_costs(dists, strack_pool, detections)
        dists = matching.fuse_score(dists, detections)
        dists = self._apply_lost_time_penalty(dists, strack_pool)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.minimum_matching_threshold
        )
        # Handle team switches in tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]

            old_team = getattr(track, "team", None)
            new_team = getattr(det, "team", None)

            if old_team is not None and new_team is not None and old_team != new_team:
                if not hasattr(track, "team_switch_frames"):
                    track.team_switch_frames = 1
                else:
                    track.team_switch_frames += 1

                # Si se mantiene el error muchos frames, asumimos que sí cambió realmente
                if track.team_switch_frames >= 5:
                    track.equipo = new_team
                    track.team_switch_frames = 0  # reset

            else:
                if hasattr(track, "team_switch_frames"):
                    track.team_switch_frames = 0

                if old_team is None and new_team is not None:
                    track.equipo = new_team
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
                det.class_name = (
                    class_labels[second_idx]
                    if class_labels is not None and second_idx < len(class_labels)
                    else None
                )
                det.field_position = (
                    self._field_position_to_array(field_positions[second_idx])
                    if field_positions is not None and second_idx < len(field_positions)
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
                if hasattr(track, "class_name") and hasattr(det, "class_name"):
                    if track.class_name != det.class_name:
                        dists[i, j] += 1000
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
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        dists = self._apply_bbox_center_costs(dists, unconfirmed, detections)
        for i, track in enumerate(unconfirmed):
            for j, det in enumerate(detections):
                if hasattr(track, "class_name") and hasattr(det, "class_name"):
                    if track.class_name != det.class_name:
                        dists[i, j] += 1000

        dists = self._apply_field_position_costs(dists, unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)
        dists = self._apply_lost_time_penalty(dists, unconfirmed)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=self.unconfirmed_match_threshold
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            self._apply_track_metadata(unconfirmed[itracked], detections[idet])
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = TrackState.Removed
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
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
