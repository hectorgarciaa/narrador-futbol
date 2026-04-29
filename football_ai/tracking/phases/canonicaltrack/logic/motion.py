from __future__ import annotations


class CanonicalMotionMixin:
    def _step_distance(
        self,
        previous_bbox,
        new_bbox,
        class_name=None,
        previous_field_position=None,
        new_field_position=None,
    ):
        if (
            self._motion_distance_space(
                class_name,
                previous_field_position=previous_field_position,
                new_field_position=new_field_position,
            )
            == "field"
        ):
            previous_field_position = self._field_position_to_tuple(previous_field_position)
            new_field_position = self._field_position_to_tuple(new_field_position)
            px, py = previous_field_position
            nx, ny = new_field_position
            return float(((nx - px) ** 2 + (ny - py) ** 2) ** 0.5)
        if self._use_field_position_for_class(
            class_name,
            previous_field_position,
        ) and self._use_field_position_for_class(class_name, new_field_position):
            px, py = self._field_position_to_tuple(previous_field_position)
            nx, ny = self._field_position_to_tuple(new_field_position)
            return float(((nx - px) ** 2 + (ny - py) ** 2) ** 0.5)
        if previous_bbox is None or new_bbox is None:
            return None
        px, py = self._bbox_center(previous_bbox)
        nx, ny = self._bbox_center(new_bbox)
        return float(((nx - px) ** 2 + (ny - py) ** 2) ** 0.5)

    def _field_distance_gate_for_lost_frames(self, lost_frames):
        gate_lost_frames = max(1, int(lost_frames))
        if self.field_distance_gate_max_lost_frames is not None:
            gate_lost_frames = min(
                gate_lost_frames,
                self.field_distance_gate_max_lost_frames,
            )
        if self.field_distance_growth_mode == "linear_decay":
            step_base = self.field_distance_gate_m
            decay = self.field_distance_decay_per_frame
            gate = 0.0
            for step_idx in range(gate_lost_frames):
                step = step_base - (decay * step_idx)
                if step <= 0.0:
                    break
                gate += step
            gate = max(step_base, gate)
        else:
            growth = float(gate_lost_frames) ** self.field_distance_lost_exponent
            gate = self.field_distance_gate_m * growth

        if self.field_distance_gate_cap_m is not None:
            gate = min(gate, self.field_distance_gate_cap_m)
        return float(gate)

    def _is_motion_compatible(
        self,
        previous_state,
        new_bbox,
        current_frame,
        class_name=None,
        new_field_position=None,
        apply_statistical_gate=True,
    ):
        effective_class = class_name or previous_state.get("class_name")
        previous_bbox = previous_state.get("bbox")
        previous_field_position = previous_state.get("field_position")
        motion_distance_space = self._motion_distance_space(
            effective_class,
            previous_field_position=previous_field_position,
            new_field_position=new_field_position,
        )
        step_distance = self._step_distance(
            previous_bbox,
            new_bbox,
            class_name=effective_class,
            previous_field_position=previous_field_position,
            new_field_position=new_field_position,
        )
        if step_distance is None:
            return True

        lost_frames = max(
            1,
            current_frame - int(previous_state.get("last_frame", current_frame)),
        )
        motion_growth_frames = lost_frames
        if self.reassign_motion_growth_cap_frames is not None:
            motion_growth_frames = min(
                lost_frames,
                self.reassign_motion_growth_cap_frames,
            )
        samples = int(previous_state.get("movement_samples", 0))
        mean_step_distance = float(previous_state.get("mean_step_distance", 0.0))
        reserved_seed = bool(previous_state.get("reserved_seed", False))
        uses_field_position = motion_distance_space == "field"
        if not uses_field_position and str(previous_state.get("motion_distance_space") or "") != "image":
            samples = 0
            mean_step_distance = 0.0

        if not reserved_seed:
            max_lost_frames = self._max_lost_frames_for_class(effective_class)
            if max_lost_frames is not None and lost_frames > max_lost_frames:
                return False

        if uses_field_position:
            max_allowed_jump = self._field_distance_gate_for_lost_frames(lost_frames)
        else:
            max_allowed_jump = self.reassign_min_distance

        if reserved_seed:
            max_allowed_jump = max(
                max_allowed_jump,
                float(previous_state.get("reserved_seed_match_distance_m", 0.0)),
            )

        if not uses_field_position and samples >= self.reassign_min_samples:
            expected_jump = mean_step_distance * motion_growth_frames
            max_allowed_jump = max(
                max_allowed_jump,
                expected_jump * self.reassign_motion_factor,
            )

        if (
            apply_statistical_gate
            and self.motion_std_gate_enabled
            and not uses_field_position
        ):
            limits_per_frame = []

            track_stats_count = int(previous_state.get("step_per_frame_count", 0))
            if str(previous_state.get("motion_distance_space") or "") != "image":
                track_stats_count = 0
            if track_stats_count >= self.motion_std_min_samples:
                track_stats_mean = float(
                    previous_state.get("step_per_frame_mean", 0.0)
                )
                track_stats_m2 = float(previous_state.get("step_per_frame_m2", 0.0))
                limits_per_frame.append(
                    self._motion_limit_per_frame(
                        track_stats_count,
                        track_stats_mean,
                        track_stats_m2,
                    )
                )

            if effective_class not in self.field_position_classes:
                class_stats = self.class_motion_stats.get(effective_class, {})
                class_stats_count = int(class_stats.get("count", 0))
                if class_stats_count >= self.motion_std_min_samples:
                    limits_per_frame.append(
                        self._motion_limit_per_frame(
                            class_stats_count,
                            float(class_stats.get("mean", 0.0)),
                            float(class_stats.get("m2", 0.0)),
                        )
                    )

            if limits_per_frame:
                max_per_frame = min(limits_per_frame)
                stats_jump_limit = max_per_frame * motion_growth_frames
                stats_jump_limit = max(
                    stats_jump_limit,
                    self.motion_std_floor * motion_growth_frames,
                )
                max_allowed_jump = min(max_allowed_jump, stats_jump_limit)

        return step_distance <= max_allowed_jump

    def _resolve_candidate_class_for_detection(
        self,
        candidate_canonical_id,
        candidate_state,
        detection_class,
        detection_team,
        detection_bbox,
        detection_field_position,
        current_frame,
        detection_class_candidates=None,
        *,
        apply_statistical_gate,
    ):
        candidate_class = candidate_state.get("class_name")
        if candidate_class is None:
            return None
        detection_class = self._select_detection_class_for_candidate(
            candidate_state,
            detection_class,
            detection_class_candidates=detection_class_candidates,
        )
        if detection_class is None:
            return None

        if (
            candidate_state.get("special_penalty_seed")
            and candidate_class in {"player", "goalkeeper"}
            and detection_class == "goalkeeper"
        ):
            if not self._is_motion_compatible(
                candidate_state,
                detection_bbox,
                current_frame,
                class_name=candidate_class,
                new_field_position=detection_field_position,
                apply_statistical_gate=apply_statistical_gate,
            ):
                return None
            return candidate_class

        if (
            candidate_state.get("reserved_seed")
            and candidate_class == "goalkeeper"
            and detection_class == "goalkeeper"
        ):
            if not self._is_motion_compatible(
                candidate_state,
                detection_bbox,
                current_frame,
                class_name=detection_class,
                new_field_position=detection_field_position,
                apply_statistical_gate=apply_statistical_gate,
            ):
                return None
            return detection_class

        if candidate_class == "referee":
            return self._resolve_referee_candidate_class_for_detection(
                candidate_canonical_id,
                candidate_state,
                detection_class,
                detection_team,
                detection_field_position,
            )

        if self._is_compatible_class(candidate_class, detection_class):
            if not self._is_team_compatible(
                candidate_state.get("team"),
                detection_team,
                candidate_class,
            ):
                return None
            if candidate_class == "player":
                resolved_team = self._normalize_team_name(
                    detection_team if detection_team is not None else candidate_state.get("team")
                )
                if resolved_team is None:
                    return None
                if not self._is_player_canonical_slot_compatible(
                    candidate_canonical_id,
                    resolved_team,
                    create_mapping=True,
                ):
                    return None
            if not self._is_motion_compatible(
                candidate_state,
                detection_bbox,
                current_frame,
                class_name=candidate_class,
                new_field_position=detection_field_position,
                apply_statistical_gate=apply_statistical_gate,
            ):
                return None

            return candidate_class

        return None

    def _resolve_raw_tracker_continuity_for_detection_debug(
        self,
        candidate_canonical_id,
        candidate_state,
        detection_class,
        detection_team,
        detection_bbox,
        detection_field_position,
        current_frame,
        detection_class_candidates=None,
    ):
        resolved_class = self._resolve_candidate_class_for_detection(
            candidate_canonical_id,
            candidate_state,
            detection_class,
            detection_team,
            detection_bbox,
            detection_field_position,
            current_frame,
            detection_class_candidates=detection_class_candidates,
            apply_statistical_gate=False,
        )
        if resolved_class is not None:
            return resolved_class, None

        candidate_class = candidate_state.get("class_name")
        if candidate_class is None:
            return None, "candidate_class_missing"
        detection_class = self._select_detection_class_for_candidate(
            candidate_state,
            detection_class,
            detection_class_candidates=detection_class_candidates,
        )
        if detection_class is None:
            return None, "detection_class_missing"
        if candidate_class == "referee":
            return self._resolve_referee_candidate_class_for_detection_debug(
                candidate_canonical_id,
                candidate_state,
                detection_class,
                detection_team,
                detection_field_position,
            )
        if not self._is_team_compatible(
            candidate_state.get("team"),
            detection_team,
            candidate_class,
        ):
            return None, "team_incompatible"
        if candidate_class == "player":
            resolved_team = self._normalize_team_name(
                detection_team if detection_team is not None else candidate_state.get("team")
            )
            if resolved_team is None:
                return None, "player_team_unknown"
            if not self._is_player_canonical_slot_compatible(
                candidate_canonical_id,
                resolved_team,
                create_mapping=True,
            ):
                return None, "player_team_slot_incompatible"
        if not self._is_motion_compatible(
            candidate_state,
            detection_bbox,
            current_frame,
            class_name=(
                "referee"
                if candidate_class == "player" and detection_class == "referee"
                else candidate_class
            ),
            new_field_position=detection_field_position,
            apply_statistical_gate=False,
        ):
            return None, "continuity_motion_incompatible"
        return None, "continuity_gate_failed"

    def _bbox_distance_sq(
        self,
        bbox_a,
        bbox_b,
        class_name=None,
        field_position_a=None,
        field_position_b=None,
    ):
        if (
            self._motion_distance_space(
                class_name,
                previous_field_position=field_position_a,
                new_field_position=field_position_b,
            )
            == "field"
        ):
            field_position_a = self._field_position_to_tuple(field_position_a)
            field_position_b = self._field_position_to_tuple(field_position_b)
            ax, ay = field_position_a
            bx, by = field_position_b
            return (ax - bx) ** 2 + (ay - by) ** 2
        if self._use_field_position_for_class(
            class_name,
            field_position_a,
        ) and self._use_field_position_for_class(class_name, field_position_b):
            ax, ay = self._field_position_to_tuple(field_position_a)
            bx, by = self._field_position_to_tuple(field_position_b)
            return (ax - bx) ** 2 + (ay - by) ** 2
        if bbox_a is None or bbox_b is None:
            return None
        ax, ay = self._bbox_center(bbox_a)
        bx, by = self._bbox_center(bbox_b)
        return (ax - bx) ** 2 + (ay - by) ** 2
