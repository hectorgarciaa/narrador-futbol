from __future__ import annotations

from typing import Optional

import numpy as np
from supervision.tracker.byte_tracker.single_object_track import STrack

from .utils import bbox_center_from_tlbr, field_position_to_array


class ByteTrackDebugTools:
    def _set_detection_debug_reason(
        self,
        raw_det_idx,
        reason,
        *,
        class_name=None,
        confidence=None,
        stage=None,
        tracker_id=None,
    ) -> None:
        if raw_det_idx is None:
            return
        try:
            raw_det_idx = int(raw_det_idx)
        except (TypeError, ValueError):
            return
        payload = self.last_detection_debug_by_raw_idx.setdefault(raw_det_idx, {})
        payload["reason"] = str(reason)
        if class_name is not None:
            payload["class_name"] = class_name
        if confidence is not None:
            try:
                payload["confidence"] = float(confidence)
            except (TypeError, ValueError):
                pass
        if stage is not None:
            payload["stage"] = str(stage)
        if tracker_id is not None:
            try:
                payload["tracker_id"] = int(tracker_id)
            except (TypeError, ValueError):
                pass

    @staticmethod
    def _pair_bbox_center_distance_px(
        track: STrack,
        det: STrack,
    ) -> Optional[float]:
        track_center = bbox_center_from_tlbr(getattr(track, "tlbr", None))
        det_center = bbox_center_from_tlbr(getattr(det, "tlbr", None))
        if not np.all(np.isfinite(track_center)) or not np.all(np.isfinite(det_center)):
            return None
        return float(np.linalg.norm(track_center - det_center))

    def _pair_field_distance_metrics(
        self,
        track: STrack,
        det: STrack,
    ) -> dict:
        track_class = getattr(track, "class_name", None)
        det_class = getattr(det, "class_name", None)
        uses_field = self._uses_field_positions_for_class(track_class)
        track_position = field_position_to_array(getattr(track, "field_position", None))
        det_position = field_position_to_array(getattr(det, "field_position", None))
        metrics = {
            "uses_field": bool(uses_field),
            "track_has_field_position": track_position is not None,
            "det_has_field_position": det_position is not None,
            "same_class": track_class == det_class,
            "field_distance_m": None,
            "field_gate_m": None,
        }
        if not uses_field or track_class != det_class:
            return metrics
        metrics["field_gate_m"] = float(self._track_distance_gate(track))
        if track_position is None or det_position is None:
            return metrics
        metrics["field_distance_m"] = float(np.linalg.norm(track_position - det_position))
        return metrics

    def _summarize_unconfirmed_candidate(
        self,
        track: STrack,
        det: STrack,
        *,
        i_track: int,
        i_det: int,
        iou_costs: np.ndarray,
        class_penalties: np.ndarray,
        bbox_size_penalties: np.ndarray,
        after_bbox_costs: np.ndarray,
        after_field_costs: np.ndarray,
        after_fuse_costs: np.ndarray,
        final_costs: np.ndarray,
    ) -> dict:
        bbox_center_distance_px = self._pair_bbox_center_distance_px(track, det)
        field_metrics = self._pair_field_distance_metrics(track, det)
        base_iou_cost = float(iou_costs[i_track, i_det])
        after_bbox_cost = float(after_bbox_costs[i_track, i_det])
        after_field_cost = float(after_field_costs[i_track, i_det])
        after_fuse_cost = float(after_fuse_costs[i_track, i_det])
        final_cost = float(final_costs[i_track, i_det])
        candidate = {
            "raw_det_idx": getattr(det, "raw_det_idx", None),
            "tracker_id": getattr(track, "external_track_id", None),
            "det_score": float(getattr(det, "score", 0.0) or 0.0),
            "det_class": getattr(det, "class_name", None),
            "det_class_team": getattr(det, "class_name_team", None),
            "det_class_yolo": getattr(det, "class_name_yolo", None),
            "det_team": getattr(det, "equipo", None),
            "track_class": getattr(track, "class_name", None),
            "track_team": getattr(track, "equipo", None),
            "iou": float(max(0.0, 1.0 - base_iou_cost)),
            "iou_cost": base_iou_cost,
            "bbox_center_distance_px": bbox_center_distance_px,
            "bbox_center_gate_px": float(self._track_image_distance_gate(track)),
            "bbox_center_cost_delta": float(after_bbox_cost - base_iou_cost),
            "class_penalty": float(class_penalties[i_track, i_det]),
            "bbox_size_penalty": float(bbox_size_penalties[i_track, i_det]),
            "field_cost_delta": float(
                after_field_cost
                - after_bbox_cost
                - class_penalties[i_track, i_det]
                - bbox_size_penalties[i_track, i_det]
            ),
            "fuse_score_delta": float(after_fuse_cost - after_field_cost),
            "lost_time_penalty": float(final_cost - after_fuse_cost),
            "final_cost": final_cost,
            "uses_field_positions": field_metrics["uses_field"],
            "track_has_field_position": field_metrics["track_has_field_position"],
            "det_has_field_position": field_metrics["det_has_field_position"],
            "field_distance_m": field_metrics["field_distance_m"],
            "field_gate_m": field_metrics["field_gate_m"],
        }
        if candidate["field_cost_delta"] >= 999.0:
            candidate["hard_blocker"] = "field_gate"
        elif candidate["class_penalty"] >= self.class_mismatch_relaxed_penalty:
            candidate["hard_blocker"] = "class_penalty"
        elif candidate["bbox_size_penalty"] >= self.bbox_size_mismatch_penalty:
            candidate["hard_blocker"] = "bbox_size_penalty"
        else:
            candidate["hard_blocker"] = None
        return candidate

    def _collect_unconfirmed_association_debug(
        self,
        *,
        unconfirmed: list[STrack],
        detections: list[STrack],
        iou_costs: np.ndarray,
        class_penalties: np.ndarray,
        bbox_size_penalties: np.ndarray,
        after_bbox_costs: np.ndarray,
        after_field_costs: np.ndarray,
        after_fuse_costs: np.ndarray,
        final_costs: np.ndarray,
        matches,
        threshold: float,
    ) -> None:
        if not self.collect_internal_matching_debug:
            self.last_unconfirmed_association_debug = []
            return

        diagnostics = []
        matched_track_to_det = {int(i_track): int(i_det) for i_track, i_det in matches}
        matched_det_to_track = {int(i_det): int(i_track) for i_track, i_det in matches}

        for i_track, track in enumerate(unconfirmed):
            payload = {
                "tracker_id": getattr(track, "external_track_id", None),
                "internal_track_id": getattr(track, "internal_track_id", None),
                "track_class": getattr(track, "class_name", None),
                "track_team": getattr(track, "equipo", None),
                "tracklet_len": int(getattr(track, "tracklet_len", 0) or 0),
                "is_activated": bool(getattr(track, "is_activated", False)),
                "track_frame_id": int(getattr(track, "frame_id", self.frame_id) or self.frame_id),
                "threshold": float(threshold),
                "candidate_count": int(len(detections)),
                "selected_candidate": None,
                "best_candidate": None,
                "top_candidates": [],
                "outcome": None,
            }
            if len(detections) <= 0:
                payload["outcome"] = "no_candidate_detections"
                diagnostics.append(payload)
                continue

            row = final_costs[i_track]
            finite_indices = [
                idx for idx, value in enumerate(row.tolist()) if np.isfinite(value)
            ]
            if not finite_indices:
                payload["outcome"] = "no_finite_candidate"
                diagnostics.append(payload)
                continue

            sorted_indices = sorted(finite_indices, key=lambda idx: float(row[idx]))
            top_indices = sorted_indices[:3]
            top_candidates = [
                self._summarize_unconfirmed_candidate(
                    track,
                    detections[i_det],
                    i_track=i_track,
                    i_det=i_det,
                    iou_costs=iou_costs,
                    class_penalties=class_penalties,
                    bbox_size_penalties=bbox_size_penalties,
                    after_bbox_costs=after_bbox_costs,
                    after_field_costs=after_field_costs,
                    after_fuse_costs=after_fuse_costs,
                    final_costs=final_costs,
                )
                for i_det in top_indices
            ]
            payload["top_candidates"] = top_candidates
            payload["best_candidate"] = top_candidates[0] if top_candidates else None

            matched_det_idx = matched_track_to_det.get(i_track)
            if matched_det_idx is not None:
                payload["outcome"] = "matched"
                payload["selected_candidate"] = self._summarize_unconfirmed_candidate(
                    track,
                    detections[matched_det_idx],
                    i_track=i_track,
                    i_det=matched_det_idx,
                    iou_costs=iou_costs,
                    class_penalties=class_penalties,
                    bbox_size_penalties=bbox_size_penalties,
                    after_bbox_costs=after_bbox_costs,
                    after_field_costs=after_field_costs,
                    after_fuse_costs=after_fuse_costs,
                    final_costs=final_costs,
                )
                diagnostics.append(payload)
                continue

            best_det_idx = top_indices[0]
            best_candidate = payload["best_candidate"]
            if float(row[best_det_idx]) > float(threshold):
                payload["outcome"] = "best_cost_above_threshold"
            elif best_det_idx in matched_det_to_track:
                payload["outcome"] = "assignment_conflict"
                payload["conflict_with_tracker_id"] = getattr(
                    unconfirmed[matched_det_to_track[best_det_idx]],
                    "external_track_id",
                    None,
                )
            else:
                payload["outcome"] = "unmatched_without_assignment"
            if best_candidate is not None:
                payload["best_candidate_hard_blocker"] = best_candidate.get(
                    "hard_blocker"
                )
            diagnostics.append(payload)

        self.last_unconfirmed_association_debug = diagnostics
