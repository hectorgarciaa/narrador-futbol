from __future__ import annotations


class CanonicalPendingAssignmentMixin:
    def _assign_pending_by_lost_order(
        self,
        pending_detections,
        available_ids,
        canonical_state,
        current_frame,
    ):
        assignments = {}
        assigned_pending_indexes = set()
        sorted_candidate_ids = sorted(
            available_ids,
            key=lambda canonical_id: (
                0 if canonical_state[canonical_id].get("reserved_seed") else 1,
                max(
                    0,
                    current_frame
                    - int(canonical_state[canonical_id].get("last_frame", current_frame)),
                ),
                canonical_id,
            ),
        )

        for canonical_id in sorted_candidate_ids:
            candidate_state = canonical_state[canonical_id]
            best_pending_idx = None
            best_distance_sq = None
            best_output_class = None

            for pending_idx, pending in enumerate(pending_detections):
                if pending_idx in assigned_pending_indexes:
                    continue

                output_class_name = self._resolve_candidate_class_for_detection(
                    canonical_id,
                    candidate_state,
                    pending["preferred_class_name"],
                    pending["detected_team"],
                    pending["bbox"],
                    pending.get("field_position"),
                    current_frame,
                    detection_class_candidates=pending.get("detection_class_candidates"),
                    apply_statistical_gate=True,
                )
                if output_class_name is None:
                    continue

                distance_sq = self._bbox_distance_sq(
                    candidate_state.get("bbox"),
                    pending["bbox"],
                    class_name=output_class_name,
                    field_position_a=candidate_state.get("field_position"),
                    field_position_b=pending.get("field_position"),
                )
                if distance_sq is None:
                    continue
                if best_distance_sq is None or distance_sq < best_distance_sq:
                    best_distance_sq = distance_sq
                    best_pending_idx = pending_idx
                    best_output_class = output_class_name

            if best_pending_idx is not None:
                assignments[best_pending_idx] = (canonical_id, best_output_class)
                assigned_pending_indexes.add(best_pending_idx)

        return assignments

    def _count_ids_for_class(self, canonical_state, class_name, current_frame=None):
        max_lost_frames = self._max_lost_frames_for_class(class_name)
        return sum(
            1
            for state in canonical_state.values()
            if state.get("class_name") == class_name
            and not state.get("reserved_seed", False)
            and (
                current_frame is None
                or (
                    max_lost_frames is None
                    or (
                        max(
                            0,
                            int(current_frame)
                            - int(state.get("last_frame", current_frame)),
                        )
                        <= max_lost_frames
                    )
                )
            )
        )

    def _best_special_penalty_seed_id(
        self,
        bbox,
        candidate_ids,
        canonical_state,
        detection_class,
        detection_team,
        current_frame,
        field_position=None,
        detection_class_candidates=None,
    ):
        best_id = None
        best_output_class = None
        best_distance_sq = None
        best_priority = None

        for canonical_id in candidate_ids:
            candidate_state = canonical_state.get(canonical_id)
            if not candidate_state.get("special_penalty_seed", False):
                continue

            output_class_name = self._resolve_candidate_class_for_detection(
                canonical_id,
                candidate_state,
                detection_class,
                detection_team,
                bbox,
                field_position,
                current_frame,
                detection_class_candidates=detection_class_candidates,
                apply_statistical_gate=True,
            )
            if output_class_name is None:
                continue

            distance_sq = self._bbox_distance_sq(
                candidate_state.get("bbox"),
                bbox,
                class_name=output_class_name,
                field_position_a=candidate_state.get("field_position"),
                field_position_b=field_position,
            )
            if distance_sq is None:
                continue

            lost_frames = max(
                0,
                current_frame - int(candidate_state.get("last_frame", current_frame)),
            )
            priority = (distance_sq, lost_frames, canonical_id)
            if best_priority is None or priority < best_priority:
                best_priority = priority
                best_id = canonical_id
                best_output_class = output_class_name
                best_distance_sq = distance_sq

        return best_id, best_output_class, best_distance_sq
