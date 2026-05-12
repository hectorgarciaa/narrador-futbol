from __future__ import annotations

import numpy as np
from supervision.tracker.byte_tracker.single_object_track import STrack


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
    def _optional_float(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def _candidate_infeasible_reasons(
        self,
        match_data: dict[str, object],
        *,
        i_track: int,
        i_det: int,
    ) -> list[str]:
        pair_metrics = match_data["pair_metrics"]
        reasons: list[str] = []
        if not bool(pair_metrics["class_gate_pass"][i_track, i_det]):
            reasons.append("class_gate")
        if not bool(pair_metrics["team_gate_pass"][i_track, i_det]):
            reasons.append("team_gate")
        if not bool(pair_metrics["bbox_size_gate_pass"][i_track, i_det]):
            reasons.append("bbox_size_gate")
        if not bool(pair_metrics["field_gate_pass"][i_track, i_det]):
            reasons.append("field_gate")
        cost_mode = str(match_data["cost_mode"])
        iou = float(pair_metrics["iou"][i_track, i_det])
        if cost_mode == "iou" and iou <= 0.0:
            reasons.append("iou_gate")
        if cost_mode == "bbox_center":
            if iou > 0.0:
                reasons.append("iou_nonzero_for_bbox_phase")
            if not bool(pair_metrics["bbox_distance_gate_pass"][i_track, i_det]):
                reasons.append("bbox_distance_gate")
        return reasons

    def _summarize_phase_candidate(
        self,
        match_data: dict[str, object],
        track: STrack,
        det: STrack,
        *,
        i_track: int,
        i_det: int,
    ) -> dict:
        pair_metrics = match_data["pair_metrics"]
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
            "phase": match_data["phase_name"],
            "cost_mode": match_data["cost_mode"],
            "iou": self._optional_float(pair_metrics["iou"][i_track, i_det]),
            "iou_cost": self._optional_float(pair_metrics["iou_cost"][i_track, i_det]),
            "bbox_center_distance_px": self._optional_float(
                pair_metrics["bbox_center_distance_px"][i_track, i_det]
            ),
            "bbox_center_gate_px": self._optional_float(
                pair_metrics["bbox_center_gate_px"][i_track, i_det]
            ),
            "bbox_center_cost": self._optional_float(
                pair_metrics["bbox_center_cost"][i_track, i_det]
            ),
            "field_distance_m": self._optional_float(
                pair_metrics["field_distance_m"][i_track, i_det]
            ),
            "field_gate_m": self._optional_float(pair_metrics["field_gate_m"][i_track, i_det]),
            "class_gate_pass": bool(pair_metrics["class_gate_pass"][i_track, i_det]),
            "team_gate_pass": bool(pair_metrics["team_gate_pass"][i_track, i_det]),
            "bbox_size_gate_pass": bool(pair_metrics["bbox_size_gate_pass"][i_track, i_det]),
            "field_gate_pass": bool(pair_metrics["field_gate_pass"][i_track, i_det]),
            "bbox_distance_gate_pass": bool(
                pair_metrics["bbox_distance_gate_pass"][i_track, i_det]
            ),
            "lost_time_penalty": self._optional_float(
                pair_metrics["lost_time_penalty"][i_track, i_det]
            ),
            "base_cost": self._optional_float(match_data["base_cost"][i_track, i_det]),
        }
        infeasible_reasons = self._candidate_infeasible_reasons(
            match_data,
            i_track=i_track,
            i_det=i_det,
        )
        candidate["is_feasible"] = len(infeasible_reasons) == 0
        candidate["infeasible_reasons"] = infeasible_reasons
        return candidate

    def _collect_phase_matching_debug(
        self,
        *,
        phase_name: str,
        tracks: list[STrack],
        detections: list[STrack],
        match_data: dict[str, object],
        matches: list[tuple[int, int]],
        threshold: float | None,
    ) -> None:
        if not self.collect_internal_matching_debug:
            self.last_matching_debug[phase_name] = []
            return

        diagnostics = []
        matched_track_to_det = {int(i_track): int(i_det) for i_track, i_det in matches}
        matched_det_to_track = {int(i_det): int(i_track) for i_track, i_det in matches}
        feasible_mask = match_data["feasible_mask"]
        base_cost = match_data["base_cost"]

        for i_track, track in enumerate(tracks):
            payload = {
                "tracker_id": getattr(track, "external_track_id", None),
                "internal_track_id": getattr(track, "internal_track_id", None),
                "track_class": getattr(track, "class_name", None),
                "track_team": getattr(track, "equipo", None),
                "tracklet_len": int(getattr(track, "tracklet_len", 0) or 0),
                "is_activated": bool(getattr(track, "is_activated", False)),
                "track_frame_id": int(getattr(track, "frame_id", self.frame_id) or self.frame_id),
                "phase": phase_name,
                "cost_mode": match_data["cost_mode"],
                "threshold": self._optional_float(threshold),
                "candidate_count": int(len(detections)),
                "feasible_candidate_count": 0,
                "selected_candidate": None,
                "best_feasible_candidate": None,
                "top_candidates": [],
                "outcome": None,
            }
            if len(detections) <= 0:
                payload["outcome"] = "no_feasible_candidate"
                diagnostics.append(payload)
                continue

            feasible_indices = [
                idx
                for idx, is_feasible in enumerate(feasible_mask[i_track].tolist())
                if bool(is_feasible)
            ]
            payload["feasible_candidate_count"] = len(feasible_indices)
            if not feasible_indices:
                payload["outcome"] = "no_feasible_candidate"
                diagnostics.append(payload)
                continue

            sorted_indices = sorted(
                feasible_indices,
                key=lambda idx: float(base_cost[i_track, idx]),
            )
            top_indices = sorted_indices[:3]
            top_candidates = [
                self._summarize_phase_candidate(
                    match_data,
                    track,
                    detections[i_det],
                    i_track=i_track,
                    i_det=i_det,
                )
                for i_det in top_indices
            ]
            payload["top_candidates"] = top_candidates
            payload["best_feasible_candidate"] = top_candidates[0] if top_candidates else None

            matched_det_idx = matched_track_to_det.get(i_track)
            if matched_det_idx is not None:
                payload["outcome"] = "matched"
                payload["selected_candidate"] = self._summarize_phase_candidate(
                    match_data,
                    track,
                    detections[matched_det_idx],
                    i_track=i_track,
                    i_det=matched_det_idx,
                )
                diagnostics.append(payload)
                continue

            best_det_idx = sorted_indices[0]
            if threshold is not None and float(base_cost[i_track, best_det_idx]) > float(threshold):
                payload["outcome"] = "best_cost_above_threshold"
            elif best_det_idx in matched_det_to_track:
                payload["outcome"] = "assignment_conflict"
                payload["conflict_with_tracker_id"] = getattr(
                    tracks[matched_det_to_track[best_det_idx]],
                    "external_track_id",
                    None,
                )
            else:
                payload["outcome"] = "unmatched_without_assignment"
            diagnostics.append(payload)

        self.last_matching_debug[phase_name] = diagnostics
