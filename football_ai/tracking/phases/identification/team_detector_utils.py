from __future__ import annotations

import numpy as np


CANDIDATE_CLASSES = ("player", "goalkeeper", "referee")
OUTFIELD_CLASSES = {"player", "goalkeeper"}


def field_position_to_tuple(field_position):
    if field_position is None:
        return None
    values = np.asarray(field_position, dtype=np.float32).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values[:2])):
        return None
    return float(values[0]), float(values[1])


def person_axis_positions(field_positions, class_labels, axis):
    positions = []
    for class_name, field_position in zip(class_labels, field_positions):
        if str(class_name) not in OUTFIELD_CLASSES:
            continue
        position = field_position_to_tuple(field_position)
        if position is not None:
            positions.append(float(position[axis]))
    positions.sort()
    return positions


def extract_shirt_crop(frame_bgr, bbox_xyxy):
    if frame_bgr is None or bbox_xyxy is None:
        return None
    box = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
    if box.size < 4:
        return None
    x1, y1, x2, y2 = box[:4].astype(int)
    if x2 <= x1 or y2 <= y1:
        return None
    player_pixels = frame_bgr[y1:y2, x1:x2]
    if player_pixels.size <= 0:
        return None
    return player_pixels[: max(1, player_pixels.shape[0] // 2), :]


def bbox_area(bbox_xyxy):
    box = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
    if box.size < 4:
        return 0.0
    x1, y1, x2, y2 = box[:4]
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def serialize_color(color):
    if color is None:
        return None
    return [float(value) for value in np.asarray(color, dtype=np.float32).reshape(-1)]


def serialize_distances(distances):
    if distances is None:
        return None
    return {
        str(team_name): None if distance is None else float(distance)
        for team_name, distance in distances.items()
    }


def nearest_outfield_team(distances):
    if not distances:
        return None
    valid = [
        (str(team_name), float(distance))
        for team_name, distance in distances.items()
        if team_name != "referee" and distance is not None
    ]
    return min(valid, key=lambda item: item[1])[0] if valid else None


def referee_position_gate(
    x_positions,
    y_positions,
    field_position,
    field_width_m,
    sideline_band_distance_m,
):
    if field_position is None:
        return False, False

    is_in_middle = False
    is_middle_ref = False
    if len(x_positions) >= 8 and len(y_positions) >= 8:
        left_x_bound = float(x_positions[3])
        right_x_bound = float(x_positions[-4])
        top_y_bound = float(y_positions[3])
        bottom_y_bound = float(y_positions[-4])
        is_in_middle = left_x_bound <= field_position[0] <= right_x_bound
        is_middle_ref = is_in_middle and top_y_bound <= field_position[1] <= bottom_y_bound

    lower_sideline_limit = float(sideline_band_distance_m)
    upper_sideline_limit = float(field_width_m) - lower_sideline_limit
    is_near_sideline = field_position[1] <= lower_sideline_limit or field_position[1] >= upper_sideline_limit
    return is_near_sideline or is_in_middle, is_middle_ref


def goalkeeper_position_gate(
    x_positions,
    field_position,
    field_width_m,
    sideline_band_distance_m,
):
    if field_position is None or len(x_positions) < 6:
        return False

    left_x_bound = float(x_positions[2])
    right_x_bound = float(x_positions[-3])
    lower_sideline_limit = float(sideline_band_distance_m)
    upper_sideline_limit = float(field_width_m) - lower_sideline_limit

    return (
        (field_position[0] < left_x_bound or field_position[0] > right_x_bound)
        and lower_sideline_limit < field_position[1] < upper_sideline_limit
    )
