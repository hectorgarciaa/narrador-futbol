from __future__ import annotations

import math

import numpy as np


class CanonicalCommonMixin:
    @staticmethod
    def _bbox_to_list(bbox):
        if bbox is None:
            return None
        if hasattr(bbox, "tolist"):
            return bbox.tolist()
        return list(bbox)

    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @staticmethod
    def _bbox_area(bbox):
        x1, y1, x2, y2 = bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    def _ball_size_measure(self, bbox):
        return float(math.sqrt(max(self._bbox_area(bbox), 0.0)))

    @staticmethod
    def _field_position_to_tuple(field_position):
        if field_position is None:
            return None
        field_position = np.asarray(field_position, dtype=np.float32).reshape(-1)
        if field_position.size < 2 or not np.all(np.isfinite(field_position[:2])):
            return None
        return (float(field_position[0]), float(field_position[1]))

    @staticmethod
    def _shirt_color_to_tuple(shirt_color):
        if shirt_color is None:
            return None
        color = np.asarray(shirt_color, dtype=np.float32).reshape(-1)
        if color.size < 3 or not np.all(np.isfinite(color[:3])):
            return None
        return (float(color[0]), float(color[1]), float(color[2]))

    def _ordered_detection_class_candidates(self, detection_class, detection_class_candidates):
        ordered = []
        for candidate_class in [detection_class, *(detection_class_candidates or [])]:
            if candidate_class is None or candidate_class in ordered:
                continue
            ordered.append(candidate_class)
        return ordered

    def _select_detection_class_for_candidate(
        self,
        candidate_state,
        detection_class,
        detection_class_candidates=None,
    ):
        candidate_class = candidate_state.get("class_name")
        ordered_candidates = self._ordered_detection_class_candidates(
            detection_class,
            detection_class_candidates,
        )
        if not ordered_candidates:
            return detection_class

        if (
            candidate_state.get("special_penalty_seed")
            and candidate_class in {"player", "goalkeeper"}
        ):
            for class_option in ordered_candidates:
                if class_option == "goalkeeper":
                    return "goalkeeper"
            return None

        if candidate_state.get("reserved_seed") and candidate_class == "goalkeeper":
            for class_option in ordered_candidates:
                if class_option == "goalkeeper":
                    return "goalkeeper"
            return None

        if candidate_class is not None:
            for class_option in ordered_candidates:
                if self._is_compatible_class(candidate_class, class_option):
                    return class_option

        if candidate_class == "referee":
            for class_option in ordered_candidates:
                if class_option in {"referee", "player"}:
                    return class_option

        if candidate_class == "player":
            for class_option in ordered_candidates:
                if class_option in {"player", "goalkeeper", "referee"}:
                    return class_option

        return ordered_candidates[0]

    @staticmethod
    def _update_running_stats(prev_count, prev_mean, prev_m2, value):
        count = int(prev_count) + 1
        delta = float(value) - float(prev_mean)
        mean = float(prev_mean) + (delta / count)
        delta2 = float(value) - mean
        m2 = float(prev_m2) + (delta * delta2)
        return count, mean, m2

    def _motion_limit_per_frame(self, stats_count, stats_mean, stats_m2):
        variance = float(stats_m2) / max(1, int(stats_count) - 1)
        std_per_frame = max(
            float(np.sqrt(max(variance, 0.0))),
            self.motion_std_floor,
        )
        return float(stats_mean) + (self.motion_std_factor * std_per_frame)

    def _use_field_position_for_class(self, class_name, field_position=None):
        if not self.use_field_positions:
            return False
        if class_name not in self.field_position_classes:
            return False
        return self._field_position_to_tuple(field_position) is not None

    def _motion_distance_space(
        self,
        class_name,
        previous_field_position=None,
        new_field_position=None,
    ):
        if not self.use_field_positions or class_name not in self.field_position_classes:
            return "image"
        previous_field_position = self._field_position_to_tuple(previous_field_position)
        new_field_position = self._field_position_to_tuple(new_field_position)
        if previous_field_position is not None and new_field_position is not None:
            return "field"
        return "image"

    def _max_lost_frames_for_class(self, class_name):
        if class_name in self.max_reassign_lost_frames_by_class:
            class_limit = self.max_reassign_lost_frames_by_class.get(class_name)
            return class_limit
        return self.max_reassign_lost_frames

    def _field_width_m(self):
        geometry = self.field_projector.geometry if self.field_projector is not None else None
        if geometry is not None and np.isfinite(geometry.field_width_m) and geometry.field_width_m > 0.0:
            return float(geometry.field_width_m)
        return float(self.referee_field_width_m)

    def _is_compatible_class(self, previous_class, new_class):
        if previous_class is None or new_class is None:
            return previous_class == new_class
        person_classes = {"player", "goalkeeper"}
        if previous_class in person_classes and new_class in person_classes:
            if self.strict_person_class_separation:
                return previous_class == new_class
            return True
        return previous_class == new_class

    @staticmethod
    def _is_team_compatible(previous_team, new_team, class_name):
        if class_name not in {"player", "goalkeeper"}:
            return True
        if previous_team is None or new_team is None:
            return True
        return previous_team == new_team
