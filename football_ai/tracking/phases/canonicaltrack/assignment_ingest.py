from __future__ import annotations


class CanonicalAssignmentIngestMixin:
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
                "team": item["team"],
                "field_position": item["field_position"],
                "raw_det_idx": item["raw_det_idx"],
                "shirt_color": item["shirt_color"],
                "class_tracker": item["class_tracker"],
                "class_name_td": item["class_name_td"],
                "class": item["class_name_td"],
                "class_yolo": item["class_name"],
                "distances": item["distances"],
                "bbox_size": item["bbox_size"],
                "ground_point_image": item["ground_point_image"],
                "referee_reassign_gate": item["referee_reassign_gate"],
                "goalkeeper_reassign_gate": item["goalkeeper_reassign_gate"],
            }
            tracked.append(
                (
                    item["bbox_xyxy"],
                    None,
                    float(item["confidence"]),
                    None,
                    int(item["tracker_id"]),
                    metadata,
                )
            )
        return tracked

    @staticmethod
    def _bytetrack_debug_map_from_packet(bytetrack_packet):
        debug_map = {}
        for item in bytetrack_packet["trace"].get("detection_debug", []):
            debug_map[int(item["det_id"])] = {
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
            raw_idx_int = int(metadata["raw_det_idx"])
            bytetrack_raw_detection_indexes.add(raw_idx_int)
            bytetrack_id_by_raw_idx[raw_idx_int] = int(tracker_id)
        return bytetrack_raw_detection_indexes, bytetrack_id_by_raw_idx

    def _update_referee_absorption_context(self, sorted_tracked_detections):
        player_x_positions = []
        for object_detected in sorted_tracked_detections:
            _, _, _, _, _, metadata = object_detected
            class_name, _ = self._primary_class_from_metadata(metadata)
            if class_name not in {"player", "goalkeeper"}:
                continue
            field_position = self._field_position_to_tuple(metadata.get("field_position"))
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
