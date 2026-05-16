from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import supervision as sv

from football_ai.core import PHASE_BYTETRACK, Phase, make_phase_packet

from .byte_tracker import ByteTrack


def _serialize_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    return value


class ByteTrackPhase(Phase):
    def __init__(self, bytetracker_conf=None):
        self.tracker = ByteTrack(**dict(bytetracker_conf or {}))

    def reset(self):
        self.tracker.reset()

    def active_track_boxes_xyxy(self):
        boxes = []
        for track in getattr(self.tracker, "tracked_tracks", []):
            if not bool(getattr(track, "is_activated", False)):
                continue
            tlbr = np.asarray(getattr(track, "tlbr", None), dtype=np.float32).reshape(-1)
            if tlbr.size >= 4 and np.all(np.isfinite(tlbr[:4])):
                boxes.append(tlbr[:4])
        return boxes

    def execute(self, identification_packet, *, execution_mode="runtime"):
        clean_in = identification_packet["clean"]
        detections = self._build_detections(clean_in)
        collect_debug = str(execution_mode).strip().lower() == "debug"
        setattr(self.tracker, "collect_internal_matching_debug", collect_debug)
        tracked = self.tracker.update_with_detections(detections)
        return self._build_packet(
            identification_packet,
            tracked,
            collect_internal_debug=collect_debug,
        )

    @staticmethod
    def _build_detections(clean):
        detections = sv.Detections(
            xyxy=np.asarray(clean["bbox_xyxy"], dtype=np.float32).reshape(-1, 4),
            confidence=np.asarray(clean["confidence"], dtype=np.float32).reshape(-1),
        )
        detections.data = {
            "team": np.asarray(clean["team"], dtype=object),
            "class_td": np.asarray(clean["class_td"], dtype=object),
            "class_yolo": np.asarray(clean["class_name"], dtype=object),
            "distances": np.asarray(clean["distances"], dtype=object),
            "shirt_color": np.asarray(clean["shirt_color"], dtype=object),
            "bbox_size": np.asarray(clean["bbox_size"], dtype=np.float32),
            "field_position": np.asarray(clean["field_positions_m"], dtype=np.float32),
            "ground_point_image": np.asarray(clean["ground_points_image_original"], dtype=np.float32),
            "raw_det_idx": np.asarray(clean["det_id"], dtype=np.int32),
        }
        return detections

    def _build_packet(self, identification_packet, tracked, *, collect_internal_debug=False):
        clean_in = identification_packet["clean"]
        num_detections = int(clean_in["num_detections"])
        det_id_to_index = {
            int(det_id): index for index, det_id in enumerate(clean_in["det_id"])
        }
        tracker_ids = [None] * num_detections
        class_trackers = [None] * num_detections
        tracked_mask = [False] * num_detections
        tracked_detections = []

        for bbox, _mask, confidence, _class_id, tracker_id, metadata in list(tracked):
            raw_det_idx = metadata.get("raw_det_idx")
            if raw_det_idx is None:
                continue
            raw_det_idx = int(raw_det_idx)
            index = det_id_to_index.get(raw_det_idx)
            if index is None:
                continue
            tracker_ids[index] = int(tracker_id)
            class_trackers[index] = (
                metadata.get("class_tracker")
                or metadata.get("class_td")
                or metadata.get("class_yolo")
            )
            tracked_mask[index] = True
            tracked_detections.append(
                {
                    "raw_det_idx": raw_det_idx,
                    "tracker_id": int(tracker_id),
                    "bbox_xyxy": _serialize_value(bbox),
                    "confidence": float(confidence),
                    "class_tracker": class_trackers[index],
                    "class_td": metadata.get("class_td"),
                    "class_name": metadata.get("class_yolo"),
                    "team": metadata.get("team"),
                    "field_position": _serialize_value(metadata.get("field_position")),
                    "ground_point_image": _serialize_value(metadata.get("ground_point_image")),
                    "distances": _serialize_value(metadata.get("distances")),
                    "shirt_color": _serialize_value(metadata.get("shirt_color")),
                    "bbox_size": _serialize_value(metadata.get("bbox_size")),
                }
            )

        tracked_count = int(sum(tracked_mask))
        clean_out = {
            "det_id": list(clean_in["det_id"]),
            "bbox_xyxy": [list(bbox) for bbox in clean_in["bbox_xyxy"]],
            "confidence": list(clean_in["confidence"]),
            "class_name": list(clean_in["class_name"]),
            "field_positions_m": [list(point) for point in clean_in["field_positions_m"]],
            "ground_points_image_original": [
                list(point) for point in clean_in["ground_points_image_original"]
            ],
            "tracked_detections": tracked_detections,
        }
        trace = {
            "summary": {
                "total_input_detections": num_detections,
                "total_tracked_detections": tracked_count,
                "total_untracked_detections": int(num_detections - tracked_count),
            },
            "tracking_alignment": {
                "tracker_id": tracker_ids,
                "class_tracker": class_trackers,
                "tracked_mask": tracked_mask,
                "tracked_count": tracked_count,
            },
        }
        if collect_internal_debug:
            trace["input_detection_metadata"] = {
                "team": _serialize_value(clean_in["team"]),
                "class_td": _serialize_value(clean_in["class_td"]),
                "distances": _serialize_value(clean_in["distances"]),
                "shirt_color": _serialize_value(clean_in["shirt_color"]),
                "bbox_size": _serialize_value(clean_in["bbox_size"]),
            }
            debug_by_raw_idx = dict(
                getattr(self.tracker, "last_detection_debug_by_raw_idx", {}) or {}
            )
            detection_debug = []
            for index, det_id in enumerate(clean_in["det_id"]):
                debug_payload = dict(debug_by_raw_idx.get(int(det_id), {}) or {})
                detection_debug.append(
                    {
                        "det_id": int(det_id),
                        "tracked": bool(tracked_mask[index]),
                        "tracker_id": tracker_ids[index],
                        "class_tracker": class_trackers[index],
                        **{
                            str(key): _serialize_value(value)
                            for key, value in debug_payload.items()
                        },
                    }
                )
            trace["detection_debug"] = detection_debug
            trace["matching_debug"] = _serialize_value(
                getattr(self.tracker, "last_matching_debug", {}) or {}
            )
        return make_phase_packet(
            phase_name=PHASE_BYTETRACK,
            frame_index=identification_packet["frame_index"],
            frame_time_ms=identification_packet["frame_time_ms"],
            image_width=identification_packet["image_width"],
            image_height=identification_packet["image_height"],
            clean=clean_out,
            trace=trace,
        )


__all__ = ["ByteTrackPhase"]
