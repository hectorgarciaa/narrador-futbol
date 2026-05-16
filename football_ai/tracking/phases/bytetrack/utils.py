from __future__ import annotations

from typing import Optional

import numpy as np
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.single_object_track import STrack


def track_tlbr(track: STrack) -> Optional[np.ndarray]:
    tlbr = getattr(track, "tlbr", None)
    if tlbr is None:
        return None
    tlbr = np.asarray(tlbr, dtype=np.float32).reshape(-1)
    if tlbr.size < 4 or not np.all(np.isfinite(tlbr[:4])):
        return None
    return tlbr[:4]


def tlbr_iou(box_a: Optional[np.ndarray], box_b: Optional[np.ndarray]) -> float:
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0, float(box_a[3] - box_a[1])
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0, float(box_b[3] - box_b[1])
    )
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def field_position_to_array(field_position) -> Optional[np.ndarray]:
    if field_position is None:
        return None
    field_position = np.asarray(field_position, dtype=np.float32).reshape(-1)
    if field_position.size < 2 or not np.all(np.isfinite(field_position[:2])):
        return None
    return field_position[:2]


def shirt_color_to_array(shirt_color) -> Optional[np.ndarray]:
    if shirt_color is None:
        return None
    array = np.asarray(shirt_color, dtype=np.float32).reshape(-1)
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return array[:3]


def bbox_size_from_tlbr(tlbr) -> tuple[Optional[float], Optional[float]]:
    if tlbr is None:
        return None, None
    tlbr = np.asarray(tlbr, dtype=np.float32).reshape(-1)
    if tlbr.size < 4 or not np.all(np.isfinite(tlbr[:4])):
        return None, None
    x1, y1, x2, y2 = tlbr[:4]
    width = float(x2 - x1)
    height = float(y2 - y1)
    if width <= 0.0 or height <= 0.0:
        return None, None
    return width, height


def bbox_center_from_tlbr(tlbr) -> np.ndarray:
    tlbr = np.asarray(tlbr, dtype=np.float32).reshape(-1)
    if tlbr.size < 4 or not np.all(np.isfinite(tlbr[:4])):
        return np.array([np.nan, np.nan], dtype=np.float32)
    x1, y1, x2, y2 = tlbr[:4]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def joint_tracks(track_list_a: list[STrack], track_list_b: list[STrack]) -> list[STrack]:
    seen_track_ids = set()
    result = []
    for track in track_list_a + track_list_b:
        if track.internal_track_id not in seen_track_ids:
            seen_track_ids.add(track.internal_track_id)
            result.append(track)
    return result


def sub_tracks(track_list_a: list[STrack], track_list_b: list[STrack]) -> list[STrack]:
    tracks = {track.internal_track_id: track for track in track_list_a}
    for track in track_list_b:
        tracks.pop(track.internal_track_id, None)
    return list(tracks.values())


def remove_duplicate_tracks(
    tracks_a: list[STrack],
    tracks_b: list[STrack],
) -> tuple[list[STrack], list[STrack]]:
    pairwise_distance = matching.iou_distance(tracks_a, tracks_b)
    matching_pairs = np.where(pairwise_distance < 0.15)

    duplicates_a, duplicates_b = set(), set()
    for track_index_a, track_index_b in zip(*matching_pairs):
        time_a = tracks_a[track_index_a].frame_id - tracks_a[track_index_a].start_frame
        time_b = tracks_b[track_index_b].frame_id - tracks_b[track_index_b].start_frame
        if time_a > time_b:
            duplicates_b.add(track_index_b)
        else:
            duplicates_a.add(track_index_a)

    result_a = [
        track for index, track in enumerate(tracks_a) if index not in duplicates_a
    ]
    result_b = [
        track for index, track in enumerate(tracks_b) if index not in duplicates_b
    ]
    return result_a, result_b


__all__ = [
    "bbox_center_from_tlbr",
    "bbox_size_from_tlbr",
    "field_position_to_array",
    "joint_tracks",
    "remove_duplicate_tracks",
    "shirt_color_to_array",
    "sub_tracks",
    "tlbr_iou",
    "track_tlbr",
]
