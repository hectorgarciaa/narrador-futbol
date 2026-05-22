from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .lineup_spec import normalize_slot_token


def safe_field_position_m(track_data):
    field_position = track_data.get("field_position_m") if isinstance(track_data, dict) else None
    if field_position is None:
        return None
    arr = np.asarray(field_position, dtype=np.float32).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    return float(arr[0]), float(arr[1])


def frame_track_keys(track_id):
    return (track_id, str(track_id))


def special_seed_frame_track_keys(frame_tracks, special_id):
    return [track_key for track_key in frame_track_keys(special_id) if track_key in frame_tracks]


def first_existing_track_payload(tracks_frame, track_id):
    for class_name in ("player", "goalkeeper"):
        frame_tracks = tracks_frame.get(class_name, {})
        if not isinstance(frame_tracks, dict):
            continue
        for track_key in frame_track_keys(track_id):
            payload = frame_tracks.get(track_key)
            if isinstance(payload, dict):
                return class_name, track_key, payload
    return None, None, None


def orient_normalized_point(x_value, y_value, attack_direction):
    x_value = float(x_value)
    y_value = float(y_value)
    if int(attack_direction) >= 0:
        return x_value, y_value
    return 1.0 - x_value, 1.0 - y_value


def segment_anchor_for_slot(layout_by_team, team_id, slot_name, attack_direction=+1):
    coords = layout_by_team.get(str(team_id), {}).get(normalize_slot_token(slot_name))
    if not isinstance(coords, dict):
        return None
    x_value = pd.to_numeric(coords.get("x"), errors="coerce")
    y_value = pd.to_numeric(coords.get("y"), errors="coerce")
    if pd.isna(x_value) or pd.isna(y_value):
        return None
    return orient_normalized_point(float(x_value) / 100.0, float(y_value) / 100.0, attack_direction)


def solve_assignment(cost_matrix, linear_sum_assignment_fn):
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    if cost_matrix.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    if linear_sum_assignment_fn is not None:
        return linear_sum_assignment_fn(cost_matrix)

    num_rows, num_cols = cost_matrix.shape
    best_cost = None
    best_pairs = None
    for chosen_cols in itertools.permutations(range(num_cols), min(num_rows, num_cols)):
        total_cost = 0.0
        pairs = []
        for row_idx, col_idx in enumerate(chosen_cols):
            total_cost += float(cost_matrix[row_idx, col_idx])
            pairs.append((row_idx, col_idx))
        if best_cost is None or total_cost < best_cost:
            best_cost = total_cost
            best_pairs = pairs
    if not best_pairs:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    return (
        np.asarray([pair[0] for pair in best_pairs], dtype=np.int64),
        np.asarray([pair[1] for pair in best_pairs], dtype=np.int64),
    )
