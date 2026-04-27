from __future__ import annotations

import numpy as np


class CanonicalAssignmentMixin:
    @staticmethod
    def _normalize_optional_positive_int(value):
        if value is None:
            return None
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return None
        if normalized <= 0:
            return None
        return normalized

    @classmethod
    def _detection_class_candidates_from_metadata(cls, metadata):
        if not isinstance(metadata, dict):
            return []
        candidates = []
        for key in ("class_tracker", "class_name_td", "class", "class_yolo"):
            value = metadata.get(key)
            if not value or value in candidates:
                continue
            candidates.append(value)
        return candidates

    @classmethod
    def _primary_class_from_metadata(cls, metadata):
        candidates = cls._detection_class_candidates_from_metadata(metadata)
        if not candidates:
            return None, []
        return candidates[0], candidates

    @staticmethod
    def _tracked_detections_from_bytetrack_packet(bytetrack_packet):
        tracked = []
        for item in bytetrack_packet["clean"].get("tracked_detections", []):
            metadata = {
                "team": item.get("team"),
                "field_position": item.get("field_position"),
                "raw_det_idx": item.get("raw_det_idx"),
                "shirt_color": item.get("shirt_color"),
                "class_tracker": item.get("class_tracker"),
                "class_name_td": item.get("class_name_td"),
                "class": item.get("class_name_td"),
                "class_yolo": item.get("class_name"),
                "distances": item.get("distances"),
                "bbox_size": item.get("bbox_size"),
                "ground_point_image": item.get("ground_point_image"),
                "referee_reassign_gate": item.get("referee_reassign_gate"),
                "goalkeeper_reassign_gate": item.get("goalkeeper_reassign_gate"),
            }
            tracked.append(
                (
                    item.get("bbox_xyxy"),
                    None,
                    float(item.get("confidence", 0.0)),
                    None,
                    int(item.get("tracker_id")),
                    metadata,
                )
            )
        return tracked

    @staticmethod
    def _bytetrack_debug_map_from_packet(bytetrack_packet):
        debug_map = {}
        for item in bytetrack_packet["trace"].get("detection_debug", []):
            det_id = item.get("det_id")
            if det_id is None:
                continue
            debug_map[int(det_id)] = {
                key: value
                for key, value in item.items()
                if key != "det_id"
            }
        return debug_map

    def _sort_tracked_detections(self, tracks_detection):
        class_priority = {"referee": 0, "goalkeeper": 1, "player": 2, "ball": 3}

        def _priority_from_metadata(metadata):
            class_name, _ = self._primary_class_from_metadata(metadata)
            return class_priority.get(str(class_name), max(class_priority.values()) + 1)

        return sorted(
            list(tracks_detection),
            key=lambda item: (
                _priority_from_metadata(item[5]),
                -float(item[2]),
            )
        )

    def _collect_bytetrack_detection_info(self, sorted_tracked_detections, collect_visual_debug):
        if not collect_visual_debug:
            return set(), {}

        bytetrack_raw_detection_indexes = set()
        bytetrack_id_by_raw_idx = {}
        for _bbox, _unused1, _confidence, _unused2, tracker_id, metadata in sorted_tracked_detections:
            raw_detection_idx = metadata.get("raw_det_idx")
            if raw_detection_idx is None:
                continue
            raw_idx_int = int(raw_detection_idx)
            bytetrack_raw_detection_indexes.add(raw_idx_int)
            bytetrack_id_by_raw_idx[raw_idx_int] = int(tracker_id)
        return bytetrack_raw_detection_indexes, bytetrack_id_by_raw_idx

    def _update_referee_absorption_context(self, sorted_tracked_detections):
        player_x_positions = []
        for object_detected in sorted_tracked_detections or []:
            _, _, _, _, _, metadata = object_detected
            class_name, _ = self._primary_class_from_metadata(metadata)
            if class_name not in {"player", "goalkeeper"}:
                continue
            field_position = self._field_position_to_tuple(
                metadata.get("field_position") if isinstance(metadata, dict) else None
            )
            if field_position is None:
                continue
            player_x_positions.append(float(field_position[0]))

        if len(player_x_positions) < 4:
            self._current_referee_central_x_bounds = None
            return

        player_x_positions.sort()
        self._current_referee_central_x_bounds = (
            player_x_positions[1],
            player_x_positions[-2],
        )

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
            if class_name is None:
                continue
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
                    if output_class_name not in tracks:
                        if collect_visual_debug and raw_detection_idx is not None:
                            bytetrack_discard_reason_by_raw_idx[int(raw_detection_idx)] = {
                                "reason_pre": "canonical_class_unsupported",
                                "class_tracker": output_class_name,
                            }
                        canonical_id = None
                    else:
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
                    ) if isinstance(current_state, dict) else 0
                    current_distance_sq = self._bbox_distance_sq(
                        current_state.get("bbox") if isinstance(current_state, dict) else None,
                        bbox,
                        class_name=output_class_name,
                        field_position_a=(
                            current_state.get("field_position")
                            if isinstance(current_state, dict)
                            else None
                        ),
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
            candidate_id
            for candidate_id in canonical_state.keys()
            if candidate_id not in used_canonical_ids_in_frame
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
                )
                if next_free_id is None:
                    if collect_visual_debug and pending.get("raw_detection_idx") is not None:
                        entry = bytetrack_discard_reason_by_raw_idx.setdefault(
                            int(pending["raw_detection_idx"]),
                            {},
                        )
                        entry["reason_post"] = "canonical_no_free_id"
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

    def _commit_assignment(
        self,
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
        raw_detection_idx=None,
        detection_class_candidates=None,
        forced_absorption_info=None,
    ):
        previous_owner_raw_id = canonical_to_raw_id.get(canonical_id)
        previous_canonical_id = raw_to_canonical_id.get(raw_tracker_id)
        canonical_relinked = False
        canonical_assignment_mode = "new_canonical"
        if forced_absorption_info is not None:
            canonical_assignment_mode = "forced_absorption"
        elif previous_canonical_id is not None and previous_canonical_id != canonical_id:
            canonical_assignment_mode = "raw_tracker_reassigned_to_other_canonical"
            canonical_relinked = True
        elif previous_owner_raw_id is not None and previous_owner_raw_id != raw_tracker_id:
            canonical_assignment_mode = "canonical_relinked_to_new_raw_tracker"
            canonical_relinked = True
        elif canonical_id in canonical_state:
            canonical_assignment_mode = "raw_continuity"
        if (
            previous_canonical_id is not None
            and previous_canonical_id != canonical_id
        ):
            canonical_to_raw_id.pop(previous_canonical_id, None)
        if (
            previous_owner_raw_id is not None
            and previous_owner_raw_id != raw_tracker_id
        ):
            raw_to_canonical_id.pop(previous_owner_raw_id, None)

        raw_to_canonical_id[raw_tracker_id] = canonical_id
        canonical_to_raw_id[canonical_id] = raw_tracker_id
        previous_state = canonical_state.get(canonical_id, {})
        motion_distance_space = self._motion_distance_space(
            output_class_name,
            previous_field_position=previous_state.get("field_position"),
            new_field_position=field_position,
        )
        previous_motion_distance_space = str(
            previous_state.get("motion_distance_space") or ""
        )
        reset_motion_stats = (
            previous_motion_distance_space not in {"", motion_distance_space}
        )
        prev_samples = 0 if reset_motion_stats else int(previous_state.get("movement_samples", 0))
        prev_mean = 0.0 if reset_motion_stats else float(previous_state.get("mean_step_distance", 0.0))
        prev_last_frame = int(previous_state.get("last_frame", n_frame))
        frame_gap = max(1, n_frame - prev_last_frame)
        prev_step_pf_count = (
            0 if reset_motion_stats else int(previous_state.get("step_per_frame_count", 0))
        )
        prev_step_pf_mean = (
            0.0 if reset_motion_stats else float(previous_state.get("step_per_frame_mean", 0.0))
        )
        prev_step_pf_m2 = (
            0.0 if reset_motion_stats else float(previous_state.get("step_per_frame_m2", 0.0))
        )
        step_distance = self._step_distance(
            previous_state.get("bbox"),
            bbox,
            class_name=output_class_name,
            previous_field_position=previous_state.get("field_position"),
            new_field_position=field_position,
        )
        if step_distance is not None:
            movement_samples = prev_samples + 1
            if prev_samples <= 0:
                mean_step_distance = step_distance
            else:
                mean_step_distance = (
                    (prev_mean * prev_samples) + step_distance
                ) / movement_samples
        else:
            movement_samples = prev_samples
            mean_step_distance = prev_mean

        if step_distance is not None:
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
            class_stats = self.class_motion_stats.setdefault(
                output_class_name,
                {"count": 0, "mean": 0.0, "m2": 0.0},
            )
            (
                class_stats["count"],
                class_stats["mean"],
                class_stats["m2"],
            ) = (
                self._update_running_stats(
                    class_stats["count"],
                    class_stats["mean"],
                    class_stats["m2"],
                    step_per_frame,
                )
                if motion_distance_space == "image"
                else (
                    class_stats["count"],
                    class_stats["mean"],
                    class_stats["m2"],
                )
            )
        else:
            step_per_frame_count = prev_step_pf_count
            step_per_frame_mean = prev_step_pf_mean
            step_per_frame_m2 = prev_step_pf_m2
        special_penalty_seed = bool(previous_state.get("special_penalty_seed", False))
        resolved_team = (
            detected_team
            if detected_team is not None
            else previous_state.get("team")
        )
        if special_penalty_seed:
            resolved_team = None
        resolved_field_position = (
            self._field_position_to_tuple(field_position)
            or previous_state.get("field_position")
        )
        referee_role_zone = previous_state.get("referee_role_zone")
        if output_class_name == "referee":
            referee_role_zone = (
                self._referee_zone_from_field_position(resolved_field_position)
                or referee_role_zone
            )
        else:
            referee_role_zone = None
        canonical_state[canonical_id] = {
            "bbox": bbox,
            "class_name": output_class_name,
            "last_frame": n_frame,
            "team": resolved_team,
            "shirt_color": self._shirt_color_to_tuple(metadata.get("shirt_color")),
            "field_position": resolved_field_position,
            "movement_samples": movement_samples,
            "mean_step_distance": mean_step_distance,
            "step_per_frame_count": step_per_frame_count,
            "step_per_frame_mean": step_per_frame_mean,
            "step_per_frame_m2": step_per_frame_m2,
            "reserved_seed": False,
            "special_penalty_seed": special_penalty_seed,
            "class_candidates": list(detection_class_candidates or []),
            "motion_distance_space": motion_distance_space,
            "referee_role_zone": referee_role_zone,
            "last_raw_tracker_id": int(raw_tracker_id),
            "canonical_assignment_mode": str(canonical_assignment_mode),
            "canonical_relinked": bool(canonical_relinked),
        }
        used_canonical_ids_in_frame.add(canonical_id)

        tracks[output_class_name][n_frame][canonical_id] = {
            "bbox": bbox,
            "confidence": confidence,
            "team": resolved_team,
            "distances": metadata.get("distances"),
            "shirt_color": metadata.get("shirt_color"),
            "class_tracker": output_class_name,
            "class_name_td": metadata.get("class_name_td") or metadata.get("class"),
            "class_relabel": metadata.get("class"),
            "class_yolo": metadata.get("class_yolo"),
            "referee_reassign_gate": metadata.get("referee_reassign_gate"),
            "goalkeeper_reassign_gate": metadata.get("goalkeeper_reassign_gate"),
            "bbox_size": metadata.get("bbox_size"),
            "source_raw_tracker_id": int(raw_tracker_id),
            "canonical_assignment_mode": str(canonical_assignment_mode),
            "canonical_relinked": bool(canonical_relinked),
            "field_position_m": (
                list(self._field_position_to_tuple(field_position))
                if self._field_position_to_tuple(field_position) is not None
                else None
            ),
            "ground_point_image": (
                metadata.get("ground_point_image").tolist()
                if hasattr(metadata.get("ground_point_image"), "tolist")
                else metadata.get("ground_point_image")
            ),
            "reserved_seed": False,
            "special_penalty_seed": special_penalty_seed,
            "forced_absorption": bool(forced_absorption_info),
            "forced_absorption_raw_tracker_streak_frames": (
                int(forced_absorption_info["raw_tracker_streak_frames"])
                if forced_absorption_info is not None
                else None
            ),
            "forced_absorption_canonical_lost_frames": (
                int(forced_absorption_info["canonical_lost_frames"])
                if forced_absorption_info is not None
                else None
            ),
            "forced_absorption_source_raw_tracker_id": (
                int(forced_absorption_info["raw_tracker_id"])
                if forced_absorption_info is not None
                else None
            ),
            "forced_absorption_mode": (
                str(forced_absorption_info.get("mode"))
                if forced_absorption_info is not None
                else None
            ),
            "forced_absorption_reference_frame": (
                int(forced_absorption_info["reference_frame"])
                if forced_absorption_info is not None
                and forced_absorption_info.get("reference_frame") is not None
                else None
            ),
            "forced_absorption_distance_sq": (
                float(forced_absorption_info["distance_sq"])
                if forced_absorption_info is not None
                and forced_absorption_info.get("distance_sq") is not None
                else None
            ),
        }

        if collect_visual_debug and raw_detection_idx is not None:
            accepted_raw_detection_indexes.add(int(raw_detection_idx))
