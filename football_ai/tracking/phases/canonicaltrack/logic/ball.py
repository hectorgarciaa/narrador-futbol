from __future__ import annotations

import numpy as np


class CanonicalBallMixin:
    def _ball_state_is_active(self, ball_state, current_frame):
        if ball_state is None:
            return False
        if self.ball_max_reassign_lost_frames is None:
            return True
        lost_frames = current_frame - int(ball_state.get("last_frame", current_frame))
        return lost_frames <= self.ball_max_reassign_lost_frames

    def _predict_ball_center(self, ball_state, current_frame):
        if ball_state is None:
            return None
        previous_bbox = ball_state.get("bbox")
        if previous_bbox is None:
            return None
        last_center = self._bbox_center(previous_bbox)
        prev_center = ball_state.get("prev_center")
        prev_frame = ball_state.get("prev_frame")
        last_frame = int(ball_state.get("last_frame", current_frame))
        gap_frames = max(1, current_frame - last_frame)
        if prev_center is None or prev_frame is None:
            return last_center
        frame_delta = max(1, last_frame - int(prev_frame))
        vx = (last_center[0] - float(prev_center[0])) / frame_delta
        vy = (last_center[1] - float(prev_center[1])) / frame_delta
        return (
            last_center[0] + (vx * gap_frames),
            last_center[1] + (vy * gap_frames),
        )

    def _ball_prediction_gate_px(self, ball_state, current_frame):
        base_gate = float(self.ball_expected_position_gate_px)
        if ball_state is None:
            return base_gate
        last_frame = int(ball_state.get("last_frame", current_frame))
        gap_frames = max(1, current_frame - last_frame)
        mean_step = float(ball_state.get("step_per_frame_mean", 0.0))
        dynamic_gate = base_gate + (
            self.ball_expected_position_gate_growth_per_frame * max(0, gap_frames - 1)
        )
        if mean_step > 0.0:
            dynamic_gate = max(dynamic_gate, mean_step * 2.5 * gap_frames)
        return float(dynamic_gate)

    @staticmethod
    def _clamp_coordinate(value, lower_bound, upper_bound):
        return max(float(lower_bound), min(float(value), float(upper_bound)))

    def _ball_prediction_reference(self, ball_state, current_frame, frame_size=None):
        predicted_center = self._predict_ball_center(ball_state, current_frame)
        if predicted_center is None:
            return None, {}
        if frame_size is None:
            return predicted_center, {}

        frame_width = max(1.0, float(frame_size[0]))
        frame_height = max(1.0, float(frame_size[1]))
        max_x = frame_width - 1.0
        max_y = frame_height - 1.0
        px, py = predicted_center
        edge_constraints = {}

        if px < 0.0:
            edge_constraints["left"] = True
        elif px > max_x:
            edge_constraints["right"] = True

        if py < 0.0:
            edge_constraints["top"] = True
        elif py > max_y:
            edge_constraints["bottom"] = True

        if not edge_constraints:
            return predicted_center, {}

        return (
            self._clamp_coordinate(px, 0.0, max_x),
            self._clamp_coordinate(py, 0.0, max_y),
        ), edge_constraints

    def _matches_ball_edge_constraints(
        self,
        candidate_center,
        edge_constraints,
        frame_size,
        prediction_gate,
    ):
        if not edge_constraints or frame_size is None:
            return True

        frame_width = max(1.0, float(frame_size[0]))
        frame_height = max(1.0, float(frame_size[1]))
        edge_band = max(40.0, min(float(prediction_gate) * 0.5, 140.0))
        cx, cy = candidate_center

        if edge_constraints.get("left") and cx > edge_band:
            return False
        if edge_constraints.get("right") and cx < (frame_width - edge_band):
            return False
        if edge_constraints.get("top") and cy > edge_band:
            return False
        if edge_constraints.get("bottom") and cy < (frame_height - edge_band):
            return False

        return True

    def _is_ball_size_compatible(self, previous_state, new_bbox, confidence):
        if previous_state is None:
            return True

        prev_bbox = previous_state.get("bbox")
        if prev_bbox is None:
            return True

        new_size = self._ball_size_measure(new_bbox)
        prev_size = self._ball_size_measure(prev_bbox)
        if prev_size <= 0.0 or new_size <= 0.0:
            return True

        ratio_limit = float(self.ball_size_ratio_per_frame)
        ratio_limit = max(1.0, ratio_limit)
        size_ratio = max(new_size, prev_size) / max(min(new_size, prev_size), 1e-6)

        if size_ratio <= ratio_limit:
            return True

        size_count = int(previous_state.get("ball_size_count", 0))
        if size_count >= self.ball_size_min_samples:
            size_mean = float(previous_state.get("ball_size_mean", prev_size))
            size_m2 = float(previous_state.get("ball_size_m2", 0.0))
            size_variance = size_m2 / max(1, size_count - 1)
            size_std = max(
                float(np.sqrt(max(size_variance, 0.0))),
                self.ball_size_std_floor,
            )
            lower_bound = max(0.0, size_mean - (self.ball_size_std_factor * size_std))
            upper_bound = size_mean + (self.ball_size_std_factor * size_std)
            if lower_bound <= new_size <= upper_bound:
                return True

        if confidence >= self.ball_high_conf_override:
            relaxed_ratio_limit = ratio_limit * self.ball_expected_position_confidence_relax
            if size_ratio <= relaxed_ratio_limit:
                return True

        return False

    def _build_ball_track_payload(self, bbox, confidence, metadata):
        field_position = self._field_position_to_tuple(metadata.get("field_position"))
        return {
            "bbox": bbox,
            "confidence": float(confidence),
            "team": metadata.get("team"),
            "distances": metadata.get("distances"),
            "shirt_color": metadata.get("shirt_color"),
            "class_tracker": "ball",
            "class_td": "ball",
            "class_yolo": "ball",
            "bbox_size": metadata.get("bbox_size"),
            "source_raw_tracker_id": None,
            "canonical_assignment_mode": "ball_selection",
            "canonical_relinked": False,
            "forced_absorption": False,
            "field_position_m": list(field_position) if field_position is not None else None,
            "ground_point_image": (
                metadata.get("ground_point_image").tolist()
                if hasattr(metadata.get("ground_point_image"), "tolist")
                else metadata.get("ground_point_image")
            ),
        }

    def _update_ball_state(self, previous_state, bbox, current_frame):
        previous_bbox = previous_state.get("bbox") if previous_state is not None else None
        prev_last_frame = (
            int(previous_state.get("last_frame", current_frame))
            if previous_state is not None
            else current_frame
        )
        frame_gap = max(1, current_frame - prev_last_frame)

        prev_samples = int(previous_state.get("movement_samples", 0)) if previous_state else 0
        prev_mean = float(previous_state.get("mean_step_distance", 0.0)) if previous_state else 0.0
        prev_step_pf_count = (
            int(previous_state.get("step_per_frame_count", 0)) if previous_state else 0
        )
        prev_step_pf_mean = (
            float(previous_state.get("step_per_frame_mean", 0.0)) if previous_state else 0.0
        )
        prev_step_pf_m2 = (
            float(previous_state.get("step_per_frame_m2", 0.0)) if previous_state else 0.0
        )

        step_distance = self._step_distance(
            previous_bbox,
            bbox,
            class_name="ball",
            previous_field_position=None,
            new_field_position=None,
        )
        if step_distance is not None:
            movement_samples = prev_samples + 1
            if prev_samples <= 0:
                mean_step_distance = step_distance
            else:
                mean_step_distance = (
                    (prev_mean * prev_samples) + step_distance
                ) / movement_samples
            step_per_frame = step_distance / frame_gap
            (
                step_per_frame_count,
                step_per_frame_mean,
                step_per_frame_m2,
            ) = self._update_running_stats(
                prev_step_pf_count,
                prev_step_pf_mean,
                prev_step_pf_m2,
                step_per_frame,
            )
        else:
            movement_samples = prev_samples
            mean_step_distance = prev_mean
            step_per_frame_count = prev_step_pf_count
            step_per_frame_mean = prev_step_pf_mean
            step_per_frame_m2 = prev_step_pf_m2

        new_size = self._ball_size_measure(bbox)
        prev_size_count = int(previous_state.get("ball_size_count", 0)) if previous_state else 0
        prev_size_mean = float(previous_state.get("ball_size_mean", 0.0)) if previous_state else 0.0
        prev_size_m2 = float(previous_state.get("ball_size_m2", 0.0)) if previous_state else 0.0
        (
            ball_size_count,
            ball_size_mean,
            ball_size_m2,
        ) = self._update_running_stats(
            prev_size_count,
            prev_size_mean,
            prev_size_m2,
            new_size,
        )

        return {
            "bbox": bbox,
            "class_name": "ball",
            "last_frame": current_frame,
            "field_position": None,
            "team": None,
            "movement_samples": movement_samples,
            "mean_step_distance": mean_step_distance,
            "step_per_frame_count": step_per_frame_count,
            "step_per_frame_mean": step_per_frame_mean,
            "step_per_frame_m2": step_per_frame_m2,
            "prev_center": (
                self._bbox_center(previous_bbox) if previous_bbox is not None else None
            ),
            "prev_frame": prev_last_frame if previous_bbox is not None else None,
            "ball_size_count": ball_size_count,
            "ball_size_mean": ball_size_mean,
            "ball_size_m2": ball_size_m2,
        }

    def _select_ball_candidate(
        self,
        candidates,
        ball_state,
        current_frame,
        frame_size=None,
    ):
        reference_state = (
            ball_state if self._ball_state_is_active(ball_state, current_frame) else None
        )
        evaluated_candidates = []

        for candidate in candidates:
            bbox = candidate["bbox"]
            confidence = float(candidate["confidence"])
            if not self._is_ball_size_compatible(reference_state, bbox, confidence):
                continue

            expected_error = 0.0
            if reference_state is not None:
                predicted_center, edge_constraints = self._ball_prediction_reference(
                    reference_state,
                    current_frame,
                    frame_size=frame_size,
                )
                if predicted_center is not None:
                    candidate_center = self._bbox_center(bbox)
                    prediction_gate = self._ball_prediction_gate_px(
                        reference_state,
                        current_frame,
                    )
                    if confidence >= self.ball_high_conf_override:
                        prediction_gate *= self.ball_expected_position_confidence_relax
                    if not self._matches_ball_edge_constraints(
                        candidate_center,
                        edge_constraints,
                        frame_size,
                        prediction_gate,
                    ):
                        continue
                    expected_error = float(
                        ((candidate_center[0] - predicted_center[0]) ** 2 + (candidate_center[1] - predicted_center[1]) ** 2) ** 0.5
                    )
                    if expected_error > prediction_gate:
                        continue

            evaluated_candidates.append(
                (
                    expected_error,
                    -confidence,
                    0 if candidate["source"] == "tracked" else 1,
                    candidate,
                )
            )

        if evaluated_candidates:
            evaluated_candidates.sort(key=lambda item: item[:3])
            return evaluated_candidates[0][3]

        if reference_state is None and candidates:
            bootstrap_candidates = sorted(
                candidates,
                key=lambda candidate: (
                    -float(candidate["confidence"]),
                    0 if candidate["source"] == "tracked" else 1,
                ),
            )
            return bootstrap_candidates[0]

        return None
