from __future__ import annotations


class CanonicalAssignmentProcessMixin:
    def _phase_process_tracked_detections(
        self,
        sorted_tracked_detections,
        canonical_state,
        raw_to_canonical_id,
        canonical_to_raw_id,
        used_canonical_ids_in_frame,
        tracks,
        accepted_raw_detection_indexes,
        collect_visual_debug,
        n_frame,
        pending_detections,
        ball_candidates,
        bytetrack_discard_reason_by_raw_idx,
    ):
        for object_detected in sorted_tracked_detections:
            bbox, _, confidence, _, tracker_id, metadata = object_detected
            bbox = self._bbox_to_list(bbox)
            confidence = float(confidence)
            class_name, detection_class_candidates = self._primary_class_from_metadata(
                metadata
            )
            detected_team = metadata["team"]
            field_position = metadata["field_position"]
            raw_detection_idx = metadata["raw_det_idx"]
            if class_name == "ball":
                ball_candidates.append(
                    {
                        "bbox": bbox,
                        "confidence": confidence,
                        "metadata": metadata,
                        "source": "tracked",
                        "raw_det_idx": raw_detection_idx,
                    }
                )
                continue

            raw_tracker_id = int(tracker_id)
            canonical_id = raw_to_canonical_id.get(raw_tracker_id)
            output_class_name = class_name
            if canonical_id is not None:
                previous_state = canonical_state.get(canonical_id)
                if previous_state is None:
                    if collect_visual_debug and raw_detection_idx is not None:
                        bytetrack_discard_reason_by_raw_idx[int(raw_detection_idx)] = {
                            "reason_pre": "canonical_state_missing",
                            "class_tracker": output_class_name,
                        }
                    canonical_id = None
                elif canonical_id in used_canonical_ids_in_frame:
                    if collect_visual_debug and raw_detection_idx is not None:
                        bytetrack_discard_reason_by_raw_idx[int(raw_detection_idx)] = {
                            "reason_pre": "canonical_id_used_in_frame",
                            "class_tracker": output_class_name,
                        }
                    canonical_id = None
                else:
                    output_class_name = previous_state["class_name"]
                    resolved_class, resolved_reason = (
                        self._resolve_raw_tracker_continuity_for_detection_debug(
                            canonical_id,
                            previous_state,
                            class_name,
                            detected_team,
                            bbox,
                            field_position,
                            n_frame,
                            detection_class_candidates=detection_class_candidates,
                        )
                    )
                    if resolved_class is None:
                        if collect_visual_debug and raw_detection_idx is not None:
                            bytetrack_discard_reason_by_raw_idx[int(raw_detection_idx)] = {
                                "reason_pre": str(resolved_reason or "canonical_gate_failed"),
                                "class_tracker": output_class_name,
                            }
                        canonical_id = None
                    else:
                        output_class_name = resolved_class
                        self._commit_assignment(
                            canonical_to_raw_id,
                            raw_to_canonical_id,
                            raw_tracker_id,
                            canonical_state,
                            n_frame,
                            used_canonical_ids_in_frame,
                            tracks,
                            accepted_raw_detection_indexes,
                            collect_visual_debug,
                            canonical_id,
                            output_class_name,
                            bbox,
                            confidence,
                            detected_team,
                            field_position,
                            metadata,
                            raw_detection_idx=raw_detection_idx,
                            detection_class_candidates=detection_class_candidates,
                        )
                        continue

            if class_name in {"player", "goalkeeper"}:
                (
                    special_override_id,
                    special_override_class_name,
                    special_override_distance_sq,
                ) = (
                    self._best_special_penalty_seed_id(
                        bbox,
                        [
                            candidate_id
                            for candidate_id in canonical_state.keys()
                            if candidate_id not in used_canonical_ids_in_frame
                        ],
                        canonical_state,
                        class_name,
                        detected_team,
                        n_frame,
                        field_position=field_position,
                        detection_class_candidates=detection_class_candidates,
                    )
                )
                should_apply_special_override = (
                    special_override_id is not None
                    and special_override_class_name is not None
                )
                if (
                    should_apply_special_override
                    and canonical_id is not None
                    and not canonical_state.get(canonical_id, {}).get(
                        "special_penalty_seed",
                        False,
                    )
                ):
                    current_state = canonical_state.get(canonical_id)
                    current_lost_frames = max(
                        0,
                        n_frame - int(current_state.get("last_frame", n_frame)),
                    )
                    current_distance_sq = self._bbox_distance_sq(
                        current_state.get("bbox"),
                        bbox,
                        class_name=output_class_name,
                        field_position_a=current_state.get("field_position"),
                        field_position_b=field_position,
                    )
                    should_apply_special_override = (
                        current_lost_frames > 1
                        and special_override_distance_sq is not None
                        and (
                            current_distance_sq is None
                            or special_override_distance_sq < current_distance_sq
                        )
                    )
                if should_apply_special_override:
                    canonical_id = special_override_id
                    output_class_name = special_override_class_name

            if canonical_id is None:
                output_class_name = class_name
                if collect_visual_debug and raw_detection_idx is not None:
                    entry = bytetrack_discard_reason_by_raw_idx.setdefault(
                        int(raw_detection_idx),
                        {},
                    )
                    entry.setdefault("reason_pre", "raw_tracker_id_unmapped_or_rejected")
                    entry.setdefault("class_tracker", output_class_name)
                pending_detections.append(
                    {
                        "raw_tracker_id": raw_tracker_id,
                        "bbox": bbox,
                        "confidence": confidence,
                        "detected_team": detected_team,
                        "field_position": field_position,
                        "metadata": metadata,
                        "preferred_class_name": output_class_name,
                        "raw_detection_idx": raw_detection_idx,
                        "detection_class_candidates": detection_class_candidates,
                    }
                )
                continue

            self._commit_assignment(
                canonical_to_raw_id,
                raw_to_canonical_id,
                raw_tracker_id,
                canonical_state,
                n_frame,
                used_canonical_ids_in_frame,
                tracks,
                accepted_raw_detection_indexes,
                collect_visual_debug,
                canonical_id,
                output_class_name,
                bbox,
                confidence,
                detected_team,
                field_position,
                metadata,
                raw_detection_idx=raw_detection_idx,
                detection_class_candidates=detection_class_candidates,
            )

    def _phase_assign_pending_detections(
        self,
        pending_detections,
        canonical_state,
        used_canonical_ids_in_frame,
        n_frame,
        collect_visual_debug,
        bytetrack_discard_reason_by_raw_idx,
        canonical_to_raw_id,
        raw_to_canonical_id,
        tracks,
        accepted_raw_detection_indexes,
        forced_absorption_state,
    ):
        available_ids = [
            canonical_id
            for canonical_id in canonical_state
            if canonical_id not in used_canonical_ids_in_frame
        ]
        pending_assignments = self._assign_pending_by_lost_order(
            pending_detections,
            available_ids,
            canonical_state,
            n_frame,
        )
        forced_absorption_assignments, active_forced_absorption_raw_ids = (
            self._build_forced_absorption_assignments(
                pending_detections,
                canonical_state,
                used_canonical_ids_in_frame,
                n_frame,
                forced_absorption_state,
                excluded_pending_indexes=set(pending_assignments.keys()),
                excluded_canonical_ids={
                    canonical_id
                    for canonical_id, _output_class_name in pending_assignments.values()
                },
            )
        )

        for pending_idx, pending in enumerate(pending_detections):
            assignment = pending_assignments.get(pending_idx)
            if assignment is not None:
                canonical_id, output_class_name = assignment
            else:
                forced_assignment = forced_absorption_assignments.get(pending_idx)
                if forced_assignment is not None:
                    canonical_id, output_class_name, forced_absorption_info = (
                        forced_assignment
                    )
                    self._commit_assignment(
                        canonical_to_raw_id,
                        raw_to_canonical_id,
                        pending["raw_tracker_id"],
                        canonical_state,
                        n_frame,
                        used_canonical_ids_in_frame,
                        tracks,
                        accepted_raw_detection_indexes,
                        collect_visual_debug,
                        canonical_id,
                        output_class_name,
                        pending["bbox"],
                        pending["confidence"],
                        pending["detected_team"],
                        pending["field_position"],
                        pending["metadata"],
                        raw_detection_idx=pending.get("raw_detection_idx"),
                        detection_class_candidates=pending.get(
                            "detection_class_candidates"
                        ),
                        forced_absorption_info=forced_absorption_info,
                    )
                    forced_absorption_state.pop(
                        int(pending["raw_tracker_id"]),
                        None,
                    )
                    active_forced_absorption_raw_ids.discard(
                        int(pending["raw_tracker_id"])
                    )
                    if collect_visual_debug and pending.get("raw_detection_idx") is not None:
                        bytetrack_discard_reason_by_raw_idx.pop(
                            int(pending["raw_detection_idx"]),
                            None,
                        )
                    continue
                output_class_name = pending["preferred_class_name"]
                if output_class_name == "player":
                    pending_team = self._normalize_team_name(pending.get("detected_team"))
                    if pending_team is None:
                        if collect_visual_debug and pending.get("raw_detection_idx") is not None:
                            entry = bytetrack_discard_reason_by_raw_idx.setdefault(
                                int(pending["raw_detection_idx"]),
                                {},
                            )
                            entry["reason_post"] = "canonical_player_team_unknown"
                        continue
                    class_count = self._count_ids_for_class(
                        canonical_state,
                        output_class_name,
                        current_frame=n_frame,
                        team_name=pending_team,
                    )
                    if class_count >= self._player_team_count_limit():
                        if collect_visual_debug and pending.get("raw_detection_idx") is not None:
                            entry = bytetrack_discard_reason_by_raw_idx.setdefault(
                                int(pending["raw_detection_idx"]),
                                {},
                            )
                            entry["reason_post"] = "canonical_player_team_limit_reached"
                        continue
                else:
                    class_limit = self.max_tracks_per_class.get(output_class_name)
                    if class_limit is not None:
                        class_count = self._count_ids_for_class(
                            canonical_state,
                            output_class_name,
                            current_frame=n_frame,
                        )
                        if class_count >= int(class_limit):
                            if collect_visual_debug and pending.get("raw_detection_idx") is not None:
                                entry = bytetrack_discard_reason_by_raw_idx.setdefault(
                                    int(pending["raw_detection_idx"]),
                                    {},
                                )
                                entry["reason_post"] = "canonical_class_limit_reached"
                            continue
                next_free_id = self._next_free_canonical_id(
                    canonical_state,
                    class_name=output_class_name,
                    field_position=pending.get("field_position"),
                    team_name=pending.get("detected_team"),
                )
                if next_free_id is None:
                    if collect_visual_debug and pending.get("raw_detection_idx") is not None:
                        entry = bytetrack_discard_reason_by_raw_idx.setdefault(
                            int(pending["raw_detection_idx"]),
                            {},
                        )
                        entry["reason_post"] = (
                            "canonical_player_team_slot_unavailable"
                            if output_class_name == "player"
                            else "canonical_no_free_id"
                        )
                    continue
                canonical_id = next_free_id

            self._commit_assignment(
                canonical_to_raw_id,
                raw_to_canonical_id,
                pending["raw_tracker_id"],
                canonical_state,
                n_frame,
                used_canonical_ids_in_frame,
                tracks,
                accepted_raw_detection_indexes,
                collect_visual_debug,
                canonical_id,
                output_class_name,
                pending["bbox"],
                pending["confidence"],
                pending["detected_team"],
                pending["field_position"],
                pending["metadata"],
                raw_detection_idx=pending.get("raw_detection_idx"),
                detection_class_candidates=pending.get("detection_class_candidates"),
            )
            forced_absorption_state.pop(int(pending["raw_tracker_id"]), None)
            if collect_visual_debug and pending.get("raw_detection_idx") is not None:
                bytetrack_discard_reason_by_raw_idx.pop(
                    int(pending["raw_detection_idx"]),
                    None,
                )
        self._finalize_forced_absorption_state(
            forced_absorption_state,
            active_forced_absorption_raw_ids,
        )
