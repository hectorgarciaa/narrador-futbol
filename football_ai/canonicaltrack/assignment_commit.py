from __future__ import annotations


class CanonicalAssignmentCommitMixin:
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
        field_position_tuple = self._field_position_to_tuple(field_position)
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
            field_position_tuple or previous_state.get("field_position")
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
                list(field_position_tuple) if field_position_tuple is not None else None
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
