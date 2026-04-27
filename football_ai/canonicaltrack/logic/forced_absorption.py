from __future__ import annotations

import numpy as np


class CanonicalForcedAbsorptionMixin:
    def _is_forced_absorption_player_detection_consistent(
        self,
        pending_detection,
    ):
        if not self.forced_absorption_enabled:
            return False
        detection_class = pending_detection.get("preferred_class_name")
        if detection_class != "player":
            return False
        detected_team = pending_detection.get("detected_team")
        if detected_team is None:
            return False
        detection_class_candidates = self._ordered_detection_class_candidates(
            detection_class,
            pending_detection.get("detection_class_candidates"),
        )
        if not detection_class_candidates:
            return False
        return all(candidate_class == "player" for candidate_class in detection_class_candidates)

    def _update_forced_absorption_candidate_state(
        self,
        forced_absorption_state,
        pending_detection,
        current_frame,
    ):
        raw_tracker_id = int(pending_detection["raw_tracker_id"])
        if not self._is_forced_absorption_player_detection_consistent(pending_detection):
            forced_absorption_state.pop(raw_tracker_id, None)
            return None

        detected_team = pending_detection.get("detected_team")
        previous_state = forced_absorption_state.get(raw_tracker_id)
        if (
            previous_state is not None
            and int(previous_state.get("last_frame", -10_000)) == int(current_frame) - 1
            and previous_state.get("class_name") == "player"
            and previous_state.get("team") == detected_team
        ):
            consecutive_frames = int(previous_state.get("consecutive_frames", 0)) + 1
            first_frame = int(previous_state.get("first_frame", current_frame))
            samples = list(previous_state.get("samples", []))
        else:
            consecutive_frames = 1
            first_frame = int(current_frame)
            samples = []

        samples.append(
            {
                "frame": int(current_frame),
                "bbox": self._bbox_to_list(pending_detection.get("bbox")),
                "field_position": self._field_position_to_tuple(
                    pending_detection.get("field_position")
                ),
            }
        )

        updated_state = {
            "raw_tracker_id": raw_tracker_id,
            "class_name": "player",
            "team": detected_team,
            "first_frame": first_frame,
            "last_frame": int(current_frame),
            "consecutive_frames": consecutive_frames,
            "samples": samples,
        }
        forced_absorption_state[raw_tracker_id] = updated_state
        return updated_state

    @staticmethod
    def _finalize_forced_absorption_state(
        forced_absorption_state,
        active_raw_tracker_ids,
    ):
        active_raw_tracker_ids = {int(raw_id) for raw_id in active_raw_tracker_ids}
        for raw_tracker_id in list(forced_absorption_state.keys()):
            if int(raw_tracker_id) not in active_raw_tracker_ids:
                forced_absorption_state.pop(raw_tracker_id, None)

    def _eligible_forced_absorption_player_ids(
        self,
        canonical_state,
        used_canonical_ids_in_frame,
        current_frame,
        team_name,
    ):
        candidate_ids = []
        min_lost_frames = int(self.forced_absorption_player_min_lost_frames)
        for canonical_id, state in canonical_state.items():
            if canonical_id in used_canonical_ids_in_frame:
                continue
            if state.get("class_name") != "player":
                continue
            if state.get("team") != team_name:
                continue
            if state.get("reserved_seed", False):
                continue
            if state.get("special_penalty_seed", False):
                continue
            lost_frames = max(
                0,
                int(current_frame) - int(state.get("last_frame", current_frame)),
            )
            if lost_frames < min_lost_frames:
                continue
            candidate_ids.append((canonical_id, lost_frames))
        return candidate_ids

    def _forced_absorption_sample_nearest_to_frame(
        self,
        orphan_candidate_state,
        reference_frame,
    ):
        samples = list(orphan_candidate_state.get("samples", []))
        if not samples:
            return None
        return min(
            samples,
            key=lambda sample: (
                abs(int(sample.get("frame", reference_frame)) - int(reference_frame)),
                abs(int(sample.get("frame", reference_frame)) - int(reference_frame)),
                int(sample.get("frame", reference_frame)),
            ),
        )

    def _forced_absorption_pair_priority(
        self,
        canonical_id,
        canonical_candidate_state,
        orphan_pending_index,
        orphan_candidate_state,
        current_frame,
    ):
        canonical_lost_frame = int(
            canonical_candidate_state.get("last_frame", current_frame)
        )
        canonical_lost_frames = max(0, int(current_frame) - canonical_lost_frame)
        orphan_reference_sample = self._forced_absorption_sample_nearest_to_frame(
            orphan_candidate_state,
            canonical_lost_frame,
        )
        if orphan_reference_sample is None:
            return None

        orphan_reference_frame = int(orphan_reference_sample.get("frame", current_frame))
        orphan_reference_bbox = self._bbox_to_list(orphan_reference_sample.get("bbox"))
        orphan_reference_field_position = self._field_position_to_tuple(
            orphan_reference_sample.get("field_position")
        )

        distance_sq = self._bbox_distance_sq(
            canonical_candidate_state.get("bbox"),
            orphan_reference_bbox,
            class_name="player",
            field_position_a=canonical_candidate_state.get("field_position"),
            field_position_b=orphan_reference_field_position,
        )
        distance_bucket = 0 if distance_sq is not None else 1
        if distance_sq is None:
            distance_sq = float("inf")
        time_delta = abs(orphan_reference_frame - canonical_lost_frame)
        orphan_streak = int(orphan_candidate_state.get("consecutive_frames", 0))

        return {
            "canonical_id": int(canonical_id),
            "pending_idx": int(orphan_pending_index),
            "priority": (
                distance_bucket,
                float(distance_sq),
                int(time_delta),
                -int(orphan_streak),
                -int(canonical_lost_frames),
                int(canonical_id),
                int(orphan_pending_index),
            ),
            "info": {
                "raw_tracker_id": int(orphan_candidate_state["raw_tracker_id"]),
                "raw_tracker_streak_frames": int(orphan_streak),
                "canonical_lost_frames": int(canonical_lost_frames),
                "team": orphan_candidate_state.get("team"),
                "mode": "player_same_team_same_class_without_position_gate",
                "reference_frame": int(orphan_reference_frame),
                "distance_sq": (
                    None
                    if not np.isfinite(float(distance_sq))
                    else float(distance_sq)
                ),
            },
        }

    def _build_forced_absorption_assignments(
        self,
        pending_detections,
        canonical_state,
        used_canonical_ids_in_frame,
        current_frame,
        forced_absorption_state,
        excluded_pending_indexes=None,
        excluded_canonical_ids=None,
    ):
        forced_assignments = {}
        active_raw_tracker_ids = set()
        excluded_pending_indexes = {
            int(pending_idx) for pending_idx in (excluded_pending_indexes or set())
        }
        excluded_canonical_ids = {
            int(canonical_id) for canonical_id in (excluded_canonical_ids or set())
        }

        eligible_orphans_by_team = {}
        min_consistent_frames = int(self.forced_absorption_player_min_consistent_frames)
        for pending_idx, pending_detection in enumerate(pending_detections):
            orphan_candidate_state = self._update_forced_absorption_candidate_state(
                forced_absorption_state,
                pending_detection,
                current_frame,
            )
            if orphan_candidate_state is None:
                continue

            raw_tracker_id = int(orphan_candidate_state["raw_tracker_id"])
            active_raw_tracker_ids.add(raw_tracker_id)
            if pending_idx in excluded_pending_indexes:
                continue
            if int(orphan_candidate_state.get("consecutive_frames", 0)) < min_consistent_frames:
                continue

            team_name = orphan_candidate_state.get("team")
            if team_name is None:
                continue
            eligible_orphans_by_team.setdefault(str(team_name), []).append(
                (int(pending_idx), pending_detection, orphan_candidate_state)
            )

        pair_candidates = []
        for team_name, orphan_candidates in eligible_orphans_by_team.items():
            eligible_canonical_ids = [
                (canonical_id, lost_frames)
                for canonical_id, lost_frames in self._eligible_forced_absorption_player_ids(
                    canonical_state,
                    used_canonical_ids_in_frame,
                    current_frame,
                    team_name,
                )
                if int(canonical_id) not in excluded_canonical_ids
            ]
            if not eligible_canonical_ids:
                continue

            for canonical_id, _ in eligible_canonical_ids:
                canonical_candidate_state = canonical_state.get(canonical_id)
                if not isinstance(canonical_candidate_state, dict):
                    continue
                for (
                    orphan_pending_index,
                    _orphan_pending_detection,
                    orphan_candidate_state,
                ) in orphan_candidates:
                    pair_priority = self._forced_absorption_pair_priority(
                        canonical_id,
                        canonical_candidate_state,
                        orphan_pending_index,
                        orphan_candidate_state,
                        current_frame,
                    )
                    if pair_priority is None:
                        continue
                    pair_candidates.append(pair_priority)

        used_pending_indexes = set()
        used_canonical_ids = set()
        for pair_priority in sorted(pair_candidates, key=lambda item: item["priority"]):
            pending_idx = int(pair_priority["pending_idx"])
            canonical_id = int(pair_priority["canonical_id"])
            if pending_idx in used_pending_indexes:
                continue
            if canonical_id in used_canonical_ids:
                continue
            forced_assignments[pending_idx] = (
                canonical_id,
                "player",
                dict(pair_priority["info"]),
            )
            used_pending_indexes.add(pending_idx)
            used_canonical_ids.add(canonical_id)

        return forced_assignments, active_raw_tracker_ids
