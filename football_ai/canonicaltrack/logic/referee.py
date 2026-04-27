from __future__ import annotations


class CanonicalRefereeMixin:
    def _referee_state_zone(self, state):
        stored_zone = str(state.get("referee_role_zone") or "").strip().lower()
        if stored_zone in {"sideline_top", "sideline_bottom", "central"}:
            return stored_zone
        return self._referee_zone_from_field_position(state.get("field_position"))

    def _referee_reserved_zone_by_canonical_id(self, canonical_id):
        zone_order = ("central", "sideline_top", "sideline_bottom")
        for zone_name, reserved_id in zip(zone_order, self.referee_canonical_ids):
            if int(reserved_id) == int(canonical_id):
                return zone_name
        return None

    def _required_referee_canonical_id_for_zone(self, referee_zone):
        normalized_zone = str(referee_zone or "").strip().lower()
        zone_order = ("central", "sideline_top", "sideline_bottom")
        for zone_name, reserved_id in zip(zone_order, self.referee_canonical_ids):
            if zone_name == normalized_zone:
                return int(reserved_id)
        return None

    def _referee_zone_from_field_position(self, field_position):
        field_position = self._field_position_to_tuple(field_position)
        if field_position is None:
            return None
        _, y_coord = field_position
        band_distance = max(0.0, float(self.referee_sideline_band_distance_m))
        field_width = self._field_width_m()
        top_distance = abs(float(y_coord))
        bottom_distance = abs(field_width - float(y_coord))
        if top_distance <= band_distance and top_distance <= bottom_distance:
            return "sideline_top"
        if bottom_distance <= band_distance:
            return "sideline_bottom"
        return "central"

    def _is_field_position_within_central_referee_x_bounds(self, field_position):
        field_position = self._field_position_to_tuple(field_position)
        if field_position is None:
            return False
        current_bounds = self._current_referee_central_x_bounds
        if current_bounds is None:
            return True
        min_x_bound, max_x_bound = current_bounds
        return float(min_x_bound) <= float(field_position[0]) <= float(max_x_bound)

    def _is_referee_canonical_slot_compatible(
        self,
        canonical_id,
        detection_field_position,
    ):
        expected_zone = self._referee_reserved_zone_by_canonical_id(canonical_id)
        if expected_zone is None:
            return True
        detection_zone = self._referee_zone_from_field_position(detection_field_position)
        if detection_zone is None or detection_zone != expected_zone:
            return False
        if expected_zone == "central":
            return self._is_field_position_within_central_referee_x_bounds(
                detection_field_position
            )
        return True

    def _resolve_referee_candidate_class_for_detection(
        self,
        candidate_canonical_id,
        candidate_state,
        detection_class,
        detection_team,
        detection_field_position,
    ):
        if detection_class != "referee":
            return None
        if not self._is_team_compatible(
            candidate_state.get("team"),
            detection_team,
            "referee",
        ):
            return None
        candidate_zone = self._referee_state_zone(candidate_state)
        detection_zone = self._referee_zone_from_field_position(detection_field_position)
        if candidate_zone is None or detection_zone is None:
            return None
        if candidate_zone != detection_zone:
            return None
        if not self._is_referee_canonical_slot_compatible(
            candidate_canonical_id,
            detection_field_position,
        ):
            return None
        return "referee"

    def _resolve_referee_candidate_class_for_detection_debug(
        self,
        candidate_canonical_id,
        candidate_state,
        detection_class,
        detection_team,
        detection_field_position,
    ):
        if detection_class != "referee":
            return None, "referee_class_incompatible"
        if not self._is_team_compatible(
            candidate_state.get("team"),
            detection_team,
            "referee",
        ):
            return None, "team_incompatible"
        candidate_zone = self._referee_state_zone(candidate_state)
        if candidate_zone is None:
            return None, "referee_candidate_zone_unknown"
        detection_zone = self._referee_zone_from_field_position(detection_field_position)
        if detection_zone is None:
            return None, "referee_detection_zone_unknown"
        if candidate_zone != detection_zone:
            return None, "referee_zone_incompatible"
        if not self._is_referee_canonical_slot_compatible(
            candidate_canonical_id,
            detection_field_position,
        ):
            expected_zone = self._referee_reserved_zone_by_canonical_id(
                candidate_canonical_id
            )
            if expected_zone == "central":
                return None, "referee_detection_outside_central_slot"
            return None, "referee_detection_outside_reserved_sideline_slot"
        return "referee", None

    def _next_free_canonical_id(
        self,
        canonical_state,
        class_name=None,
        field_position=None,
    ):
        reserved_referee_ids = set(self.referee_canonical_ids)
        reserved_goalkeeper_ids = set(self.special_seed_canonical_ids)
        if class_name == "referee":
            detection_zone = self._referee_zone_from_field_position(field_position)
            required_id = self._required_referee_canonical_id_for_zone(detection_zone)
            if required_id is None:
                return None
            if not self._is_referee_canonical_slot_compatible(required_id, field_position):
                return None
            candidate_range = [required_id]
        elif class_name == "goalkeeper":
            candidate_range = list(self.special_seed_canonical_ids)
        else:
            candidate_range = [
                canonical_id
                for canonical_id in range(1, self.max_total_tracks + 1)
                if canonical_id not in reserved_referee_ids
                and canonical_id not in reserved_goalkeeper_ids
            ]
        for canonical_id in candidate_range:
            if canonical_id not in canonical_state:
                return canonical_id
        return None
