from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import pandas as pd

from football_ai.core.config import Config
from football_ai.visualization import Drawer

try:
    from experiments.positions.position_dataset import (
        find_project_root,
        load_tracks_json,
        resolve_tracks_path_for_video,
        sanitize_video_stem,
    )
except ImportError:  # pragma: no cover - soporte ejecucion directa del archivo.
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from experiments.positions.position_dataset import (  # type: ignore
        find_project_root,
        load_tracks_json,
        resolve_tracks_path_for_video,
        sanitize_video_stem,
    )


logger = logging.getLogger(__name__)

DEFAULT_VIDEO_PATH = Path("data/partidoPrueba/partido_ajustado.mp4")
DEFAULT_OUTPUT_DIR = Path("output/predictions/possession")
TRACKED_DRAW_CLASSES: tuple[str, ...] = ("ball", "goalkeeper", "player", "referee")


@dataclass(frozen=True)
class PossessionConfig:
    strict_control_distance_px: float = 60.0
    strict_control_distance_ratio_to_height: float = 0.65
    loose_control_distance_px: float = 95.0
    loose_control_distance_ratio_to_height: float = 1.0
    continuation_distance_px: float = 35.0
    immediate_opponent_switch_distance_px: float = 24.0
    opponent_switch_confirmation_frames: int = 2
    opponent_switch_window_frames: int = 4
    slow_ball_speed_px: float = 14.0
    same_player_control_speed_px: float = 18.0
    speed_drop_threshold_px: float = 8.0
    direction_change_threshold_deg: float = 35.0
    max_ball_motion_step_px: float = 180.0
    max_ball_reacquisition_gap_frames: int = 4
    ball_prediction_base_error_px: float = 70.0
    ball_prediction_speed_error_gain: float = 1.75
    min_confidence_to_accept_large_ball_jump: float = 0.55
    carry_predicted_ball_frames: int = 2
    opponent_takeover_margin_px: float = 12.0
    ball_missing_release_frames: int = 10
    touch_timeout_frames: int = 90
    fill_unknown_gap_frames: int = 3
    min_possession_segment_frames: int = 2
    overlay_margin_px: int = 18


@dataclass(frozen=True)
class Candidate:
    track_id: str
    class_name: str
    team_id: str | None
    distance_px: float
    player_height_px: float
    inside_bbox: bool


def _visualization_colors_from_config(project_root: Path) -> dict[str, tuple[int, int, int]]:
    config = Config.from_yaml(project_root / "config.yaml")
    colors_raw = config.get("visualization", "colors", default={}) or {}
    colors: dict[str, tuple[int, int, int]] = {}
    for class_name, color_values in colors_raw.items():
        if not isinstance(color_values, (list, tuple)) or len(color_values) < 3:
            continue
        colors[str(class_name)] = tuple(int(v) for v in color_values[:3])
    if not colors:
        colors = {
            "player": (0, 255, 0),
            "goalkeeper": (0, 255, 255),
            "referee": (255, 0, 0),
            "ball": (0, 0, 255),
        }
    return colors


def _ball_center(frame_ball_tracks: Mapping[str, Any]) -> tuple[float, float] | None:
    if not isinstance(frame_ball_tracks, Mapping) or not frame_ball_tracks:
        return None
    _, data = next(iter(frame_ball_tracks.items()))
    bbox = data.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _ball_confidence(frame_ball_tracks: Mapping[str, Any]) -> float | None:
    if not isinstance(frame_ball_tracks, Mapping) or not frame_ball_tracks:
        return None
    _, data = next(iter(frame_ball_tracks.items()))
    confidence = data.get("confidence")
    if confidence is None:
        return None
    try:
        return float(confidence)
    except (TypeError, ValueError):
        return None


def _player_footpoint(track_data: Mapping[str, Any]) -> tuple[float, float] | None:
    bbox = track_data.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return (x1 + x2) / 2.0, y2


def _player_height_px(track_data: Mapping[str, Any]) -> float | None:
    bbox = track_data.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    y1 = float(bbox[1])
    y2 = float(bbox[3])
    return max(1.0, y2 - y1)


def _ball_inside_expanded_bbox(ball_center: tuple[float, float], track_data: Mapping[str, Any]) -> bool:
    bbox = track_data.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bx, by = ball_center
    return (x1 - 8.0) <= bx <= (x2 + 8.0) and (y1 - 8.0) <= by <= (y2 + 16.0)


def _ball_motion(
    ball_centers: Sequence[tuple[float, float] | None],
    config: PossessionConfig,
) -> tuple[list[float | None], list[float | None]]:
    num_frames = len(ball_centers)
    speeds: list[float | None] = [None] * num_frames
    direction_changes: list[float | None] = [None] * num_frames

    for frame_id in range(1, num_frames):
        curr = ball_centers[frame_id]
        prev = ball_centers[frame_id - 1]
        if curr is None or prev is None:
            continue
        distance = math.dist(curr, prev)
        if distance <= config.max_ball_motion_step_px:
            speeds[frame_id] = distance

    for frame_id in range(2, num_frames):
        curr = ball_centers[frame_id]
        prev = ball_centers[frame_id - 1]
        prev_prev = ball_centers[frame_id - 2]
        if curr is None or prev is None or prev_prev is None:
            continue
        if speeds[frame_id] is None or speeds[frame_id - 1] is None:
            continue
        v1 = (prev[0] - prev_prev[0], prev[1] - prev_prev[1])
        v2 = (curr[0] - prev[0], curr[1] - prev[1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 <= 1e-6 or n2 <= 1e-6:
            continue
        cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        direction_changes[frame_id] = math.degrees(math.acos(cosine))

    return speeds, direction_changes


def _recent_valid_history(
    stabilized_centers: Sequence[tuple[float, float] | None],
    frame_id: int,
    max_items: int = 2,
) -> list[tuple[int, tuple[float, float]]]:
    history: list[tuple[int, tuple[float, float]]] = []
    lookback = frame_id - 1
    while lookback >= 0 and len(history) < max_items:
        center = stabilized_centers[lookback]
        if center is not None:
            history.append((lookback, center))
        lookback -= 1
    return history


def _stabilize_ball_track(
    raw_ball_centers: Sequence[tuple[float, float] | None],
    raw_ball_confidences: Sequence[float | None],
    config: PossessionConfig,
) -> tuple[list[tuple[float, float] | None], list[str]]:
    stabilized_centers: list[tuple[float, float] | None] = [None] * len(raw_ball_centers)
    center_sources: list[str] = ["missing"] * len(raw_ball_centers)

    for frame_id, observed_center in enumerate(raw_ball_centers):
        observed_conf = raw_ball_confidences[frame_id] or 0.0
        history = _recent_valid_history(stabilized_centers, frame_id, max_items=2)
        predicted_center: tuple[float, float] | None = None
        allowed_prediction_error = config.ball_prediction_base_error_px
        gap_from_last_valid: int | None = None

        if history:
            last_frame_id, last_center = history[0]
            gap_from_last_valid = frame_id - last_frame_id
            if len(history) >= 2:
                prev_frame_id, prev_center = history[1]
                frame_delta = max(1, last_frame_id - prev_frame_id)
                vx = (last_center[0] - prev_center[0]) / frame_delta
                vy = (last_center[1] - prev_center[1]) / frame_delta
                predicted_center = (
                    last_center[0] + (vx * gap_from_last_valid),
                    last_center[1] + (vy * gap_from_last_valid),
                )
                speed = math.hypot(vx, vy)
                allowed_prediction_error = (
                    config.ball_prediction_base_error_px
                    + (speed * config.ball_prediction_speed_error_gain * gap_from_last_valid)
                )
            else:
                predicted_center = last_center
                allowed_prediction_error = (
                    config.ball_prediction_base_error_px
                    + (config.max_ball_motion_step_px * max(0, gap_from_last_valid - 1))
                )

        if observed_center is not None:
            accept_observation = True
            if history and gap_from_last_valid is not None:
                last_frame_id, last_center = history[0]
                gap_frames = frame_id - last_frame_id
                max_direct_jump = config.max_ball_motion_step_px * max(1, gap_frames)
                direct_jump = math.dist(observed_center, last_center)
                predicted_error = (
                    math.dist(observed_center, predicted_center)
                    if predicted_center is not None
                    else direct_jump
                )
                plausible_jump = direct_jump <= max_direct_jump
                plausible_prediction = predicted_error <= allowed_prediction_error
                large_jump_high_conf = (
                    observed_conf >= config.min_confidence_to_accept_large_ball_jump
                    and predicted_error <= (allowed_prediction_error * 1.5)
                )
                accept_observation = (
                    gap_frames <= config.max_ball_reacquisition_gap_frames
                    and (plausible_jump or plausible_prediction or large_jump_high_conf)
                )

            if accept_observation:
                stabilized_centers[frame_id] = observed_center
                center_sources[frame_id] = "observed"
                continue

            center_sources[frame_id] = "rejected_observation"

        if (
            predicted_center is not None
            and gap_from_last_valid is not None
            and gap_from_last_valid <= config.carry_predicted_ball_frames
        ):
            stabilized_centers[frame_id] = predicted_center
            center_sources[frame_id] = "predicted"
        elif center_sources[frame_id] == "rejected_observation":
            stabilized_centers[frame_id] = None
        else:
            center_sources[frame_id] = "missing"

    return stabilized_centers, center_sources


def _frame_candidates(
    tracks: Mapping[str, Any],
    frame_id: int,
    ball_center: tuple[float, float] | None,
) -> list[Candidate]:
    if ball_center is None:
        return []

    candidates: list[Candidate] = []
    for class_name in ("player", "goalkeeper"):
        frame_tracks = tracks.get(class_name, [])
        if frame_id >= len(frame_tracks):
            continue
        frame_data = frame_tracks[frame_id]
        if not isinstance(frame_data, Mapping):
            continue
        for track_id_raw, track_data in frame_data.items():
            footpoint = _player_footpoint(track_data)
            height = _player_height_px(track_data)
            if footpoint is None or height is None:
                continue
            distance = math.dist(ball_center, footpoint)
            candidates.append(
                Candidate(
                    track_id=str(track_id_raw),
                    class_name=str(class_name),
                    team_id=track_data.get("team"),
                    distance_px=float(distance),
                    player_height_px=float(height),
                    inside_bbox=_ball_inside_expanded_bbox(ball_center, track_data),
                )
            )
    candidates.sort(key=lambda candidate: candidate.distance_px)
    return candidates


def _segment_slices(values: Sequence[str | None]) -> list[tuple[int, int, str | None]]:
    if not values:
        return []
    segments: list[tuple[int, int, str | None]] = []
    start = 0
    current = values[0]
    for idx in range(1, len(values) + 1):
        if idx == len(values) or values[idx] != current:
            segments.append((start, idx, current))
            if idx < len(values):
                start = idx
                current = values[idx]
    return segments


def _fill_short_unknown_gaps(
    values: Sequence[str | None],
    max_gap_frames: int,
) -> list[str | None]:
    result = list(values)
    if max_gap_frames <= 0:
        return result
    for seg_idx, (start, end, team_id) in enumerate(_segment_slices(result)):
        if team_id is not None or (end - start) > max_gap_frames:
            continue
        prev_team = _segment_slices(result)[seg_idx - 1][2] if seg_idx > 0 else None
        next_team = (
            _segment_slices(result)[seg_idx + 1][2]
            if seg_idx + 1 < len(_segment_slices(result))
            else None
        )
        if prev_team is not None and prev_team == next_team:
            for frame_id in range(start, end):
                result[frame_id] = prev_team
    return result


def _suppress_short_team_segments(
    values: Sequence[str | None],
    min_segment_frames: int,
) -> list[str | None]:
    result = list(values)
    if min_segment_frames <= 1:
        return result

    changed = True
    while changed:
        changed = False
        segments = _segment_slices(result)
        for seg_idx, (start, end, team_id) in enumerate(segments):
            if team_id is None or (end - start) >= min_segment_frames:
                continue
            prev_team = segments[seg_idx - 1][2] if seg_idx > 0 else None
            next_team = segments[seg_idx + 1][2] if seg_idx + 1 < len(segments) else None
            replacement: str | None = None
            if prev_team is not None and prev_team == next_team:
                replacement = prev_team
            elif prev_team is None and next_team is not None:
                replacement = next_team
            elif next_team is None and prev_team is not None:
                replacement = prev_team
            if replacement is None:
                continue
            for frame_id in range(start, end):
                result[frame_id] = replacement
            changed = True
            break
    return result


def _stabilize_possession_df(
    possession_df: pd.DataFrame,
    config: PossessionConfig,
) -> pd.DataFrame:
    stable_df = possession_df.copy()
    raw_team_values = [
        team if pd.notna(team) else None for team in stable_df["raw_possession_team_id"].tolist()
    ]
    stable_team_values = _fill_short_unknown_gaps(
        raw_team_values,
        max_gap_frames=config.fill_unknown_gap_frames,
    )
    stable_team_values = _suppress_short_team_segments(
        stable_team_values,
        min_segment_frames=config.min_possession_segment_frames,
    )
    stable_team_values = _fill_short_unknown_gaps(
        stable_team_values,
        max_gap_frames=config.fill_unknown_gap_frames,
    )

    stable_reasons: list[str] = []
    stable_players: list[str | None] = []
    for raw_team, raw_player, raw_reason, stable_team in zip(
        raw_team_values,
        stable_df["raw_possession_player_id"].tolist(),
        stable_df["raw_possession_reason"].tolist(),
        stable_team_values,
    ):
        if raw_team == stable_team:
            stable_reasons.append(str(raw_reason))
            stable_players.append(raw_player if pd.notna(raw_player) else None)
            continue
        if stable_team is None:
            stable_reasons.append("stabilized_unknown")
            stable_players.append(None)
            continue
        stable_reasons.append("stabilized_team_segment")
        stable_players.append(raw_player if raw_team == stable_team and pd.notna(raw_player) else None)

    stable_df["possession_team_id"] = stable_team_values
    stable_df["possession_player_id"] = stable_players
    stable_df["possession_reason"] = stable_reasons
    return stable_df


def infer_team_possession(
    tracks: Mapping[str, Any],
    config: PossessionConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = config or PossessionConfig()
    num_frames = max(len(tracks.get(class_name, [])) for class_name in TRACKED_DRAW_CLASSES)
    ball_tracks = tracks.get("ball", [])
    raw_ball_centers = [
        _ball_center(ball_tracks[frame_id]) if frame_id < len(ball_tracks) else None
        for frame_id in range(num_frames)
    ]
    raw_ball_confidences = [
        _ball_confidence(ball_tracks[frame_id]) if frame_id < len(ball_tracks) else None
        for frame_id in range(num_frames)
    ]
    ball_centers, ball_center_sources = _stabilize_ball_track(
        raw_ball_centers=raw_ball_centers,
        raw_ball_confidences=raw_ball_confidences,
        config=config,
    )
    ball_speeds, direction_changes = _ball_motion(ball_centers, config)

    current_team: str | None = None
    current_player: str | None = None
    last_touch_frame = -10_000
    last_ball_frame = -10_000
    pending_switch_team: str | None = None
    pending_switch_player: str | None = None
    pending_switch_count = 0
    pending_switch_last_frame = -10_000
    rows: list[dict[str, Any]] = []

    for frame_id in range(num_frames):
        raw_ball_center = raw_ball_centers[frame_id]
        ball_center = ball_centers[frame_id]
        ball_center_source = ball_center_sources[frame_id]
        raw_ball_confidence = raw_ball_confidences[frame_id]
        if ball_center is not None:
            last_ball_frame = frame_id
        if (
            pending_switch_team is not None
            and (frame_id - pending_switch_last_frame) > config.opponent_switch_window_frames
        ):
            pending_switch_team = None
            pending_switch_player = None
            pending_switch_count = 0

        candidates = _frame_candidates(tracks, frame_id, ball_center)
        team_candidates = [candidate for candidate in candidates if candidate.team_id]
        best = team_candidates[0] if team_candidates else None
        second = team_candidates[1] if len(team_candidates) > 1 else None

        touch_candidate: Candidate | None = None
        touch_reason: str | None = None
        possession_reason = "carry"
        ball_speed = ball_speeds[frame_id]
        previous_speed = ball_speeds[frame_id - 1] if frame_id > 0 else None
        direction_change = direction_changes[frame_id]
        speed_drop = (
            previous_speed - ball_speed
            if previous_speed is not None and ball_speed is not None
            else None
        )

        if best is not None:
            strict_threshold = min(
                config.strict_control_distance_px,
                best.player_height_px * config.strict_control_distance_ratio_to_height,
            )
            loose_threshold = min(
                config.loose_control_distance_px,
                best.player_height_px * config.loose_control_distance_ratio_to_height,
            )
            strict_contact = best.distance_px <= strict_threshold
            loose_contact = best.distance_px <= loose_threshold
            separation = (
                second.distance_px - best.distance_px if second is not None else float("inf")
            )
            motion_touch = (
                (ball_speed is not None and ball_speed <= config.slow_ball_speed_px)
                or (speed_drop is not None and speed_drop >= config.speed_drop_threshold_px)
                or (
                    direction_change is not None
                    and direction_change >= config.direction_change_threshold_deg
                )
            )
            opponent_takeover_candidate = (
                current_team is not None
                and best.team_id != current_team
                and strict_contact
                and separation >= config.opponent_takeover_margin_px
            )
            start_touch = current_team is None and strict_contact
            same_player_control = (
                current_player == best.track_id
                and loose_contact
                and (
                    (ball_speed is not None and ball_speed <= config.same_player_control_speed_px)
                    or best.distance_px <= config.continuation_distance_px
                )
            )
            immediate_opponent_switch = opponent_takeover_candidate and (
                best.inside_bbox
                or motion_touch
                or best.distance_px <= config.immediate_opponent_switch_distance_px
            )
            confirmed_pending_switch = False

            if opponent_takeover_candidate and not immediate_opponent_switch:
                same_pending_team = (
                    pending_switch_team == best.team_id
                    and (frame_id - pending_switch_last_frame) <= config.opponent_switch_window_frames
                )
                if same_pending_team:
                    pending_switch_count += 1
                else:
                    pending_switch_team = best.team_id
                    pending_switch_player = best.track_id
                    pending_switch_count = 1
                pending_switch_last_frame = frame_id
                confirmed_pending_switch = (
                    pending_switch_count >= config.opponent_switch_confirmation_frames
                )
            elif best.team_id == current_team or current_team is None:
                pending_switch_team = None
                pending_switch_player = None
                pending_switch_count = 0

            if immediate_opponent_switch:
                touch_candidate = best
                touch_reason = "opponent_touch"
            elif confirmed_pending_switch:
                touch_candidate = best
                touch_reason = "opponent_touch_confirmed"
            elif same_player_control:
                touch_candidate = best
                touch_reason = "same_player_control"
            elif start_touch:
                touch_candidate = best
                touch_reason = "start_touch"
            elif best.inside_bbox and strict_contact:
                touch_candidate = best
                touch_reason = "inside_bbox_touch"
            elif strict_contact and motion_touch:
                touch_candidate = best
                touch_reason = "motion_touch"

        if touch_candidate is not None:
            current_team = touch_candidate.team_id
            current_player = touch_candidate.track_id
            last_touch_frame = frame_id
            possession_reason = touch_reason or "touch"
            pending_switch_team = None
            pending_switch_player = None
            pending_switch_count = 0
        else:
            if current_team is not None:
                if ball_center is None and (frame_id - last_ball_frame) > config.ball_missing_release_frames:
                    current_team = None
                    current_player = None
                    possession_reason = "ball_missing_timeout"
                elif ball_center is not None and (frame_id - last_touch_frame) > config.touch_timeout_frames:
                    current_team = None
                    current_player = None
                    possession_reason = "touch_timeout"
            if current_team is None:
                possession_reason = "unknown"

        row = {
            "frame_id": int(frame_id),
            "ball_detected": bool(ball_center is not None),
            "ball_source": ball_center_source,
            "ball_x_px": float(ball_center[0]) if ball_center is not None else None,
            "ball_y_px": float(ball_center[1]) if ball_center is not None else None,
            "raw_ball_detected": bool(raw_ball_center is not None),
            "raw_ball_x_px": float(raw_ball_center[0]) if raw_ball_center is not None else None,
            "raw_ball_y_px": float(raw_ball_center[1]) if raw_ball_center is not None else None,
            "raw_ball_confidence": (
                float(raw_ball_confidence) if raw_ball_confidence is not None else None
            ),
            "ball_speed_px": float(ball_speed) if ball_speed is not None else None,
            "ball_direction_change_deg": (
                float(direction_change) if direction_change is not None else None
            ),
            "nearest_track_id": best.track_id if best is not None else None,
            "nearest_class_name": best.class_name if best is not None else None,
            "nearest_team_id": best.team_id if best is not None else None,
            "nearest_distance_px": float(best.distance_px) if best is not None else None,
            "second_nearest_distance_px": (
                float(second.distance_px) if second is not None else None
            ),
            "touch_track_id": touch_candidate.track_id if touch_candidate is not None else None,
            "touch_team_id": touch_candidate.team_id if touch_candidate is not None else None,
            "touch_reason": touch_reason,
            "raw_possession_player_id": current_player,
            "raw_possession_team_id": current_team,
            "raw_possession_reason": possession_reason,
        }
        rows.append(row)

    possession_df = _stabilize_possession_df(pd.DataFrame(rows), config=config)
    known_df = possession_df[possession_df["possession_team_id"].notna()].copy()
    team_frame_counts = (
        known_df["possession_team_id"].value_counts(dropna=False).sort_index().to_dict()
        if not known_df.empty
        else {}
    )
    touch_counts = (
        possession_df["touch_team_id"].dropna().value_counts().sort_index().to_dict()
        if possession_df["touch_team_id"].notna().any()
        else {}
    )

    switch_count = 0
    previous_team: str | None = None
    for team_id in possession_df["possession_team_id"].tolist():
        if team_id != previous_team:
            if previous_team is not None or team_id is not None:
                switch_count += 1
            previous_team = team_id

    summary = {
        "num_frames": int(num_frames),
        "raw_ball_detected_frames": int(possession_df["raw_ball_detected"].sum()),
        "ball_detected_frames": int(possession_df["ball_detected"].sum()),
        "predicted_ball_frames": int((possession_df["ball_source"] == "predicted").sum()),
        "rejected_ball_frames": int((possession_df["ball_source"] == "rejected_observation").sum()),
        "known_possession_frames": int(len(known_df)),
        "unknown_possession_frames": int(num_frames - len(known_df)),
        "switch_count": int(max(0, switch_count - 1)),
        "team_frame_counts": {str(k): int(v) for k, v in team_frame_counts.items()},
        "team_frame_share_known": {
            str(k): float(v / max(len(known_df), 1)) for k, v in team_frame_counts.items()
        },
        "touch_counts_by_team": {str(k): int(v) for k, v in touch_counts.items()},
        "config": asdict(config),
    }
    return possession_df, summary


def _draw_possession_overlay(
    frame: Any,
    team_id: str | None,
    config: PossessionConfig,
) -> None:
    label = f"Posesion: {team_id}" if team_id else "Posesion: sin asignar"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    box_width = text_size[0] + 24
    box_height = text_size[1] + baseline + 18
    x = max(config.overlay_margin_px, frame.shape[1] - box_width - config.overlay_margin_px)
    y = config.overlay_margin_px
    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (255, 255, 255), 2)
    cv2.putText(
        frame,
        label,
        (x + 12, y + box_height - baseline - 8),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def render_possession_video(
    video_path: Path,
    tracks: Mapping[str, Any],
    possession_df: pd.DataFrame,
    project_root: Path | None = None,
    output_path: Path | None = None,
    show: bool = False,
    config: PossessionConfig | None = None,
) -> dict[str, Any]:
    config = config or PossessionConfig()
    project_root = find_project_root(project_root)
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"No existe el video de entrada: {video_path}")

    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_possession_annotated.mp4")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    drawer = Drawer(colors=_visualization_colors_from_config(project_root))
    cap, out = drawer.create_writer(str(video_path), str(output_path))
    show_window = bool(show)
    if show_window and not drawer._can_show_gui():
        logger.warning(
            "show=True pero no hay entorno grafico. Se desactiva la visualizacion en tiempo real."
        )
        show_window = False

    possession_by_frame = possession_df.set_index("frame_id")["possession_team_id"].to_dict()
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_id >= num_frames:
                break

            for class_name in TRACKED_DRAW_CLASSES:
                class_tracks = tracks.get(class_name, [])
                if frame_id < len(class_tracks):
                    drawer.draw_all_detections_in_frame(frame, class_name, class_tracks, frame_id)

            _draw_possession_overlay(
                frame=frame,
                team_id=possession_by_frame.get(frame_id),
                config=config,
            )

            out.write(frame)
            if show_window:
                cv2.imshow("Posesion de balon", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_id += 1
    finally:
        cap.release()
        out.release()
        if show_window:
            cv2.destroyAllWindows()

    return {
        "video_path": output_path,
        "source_video_path": video_path,
    }


def process_team_possession(
    video_path: Path = DEFAULT_VIDEO_PATH,
    project_root: Path | None = None,
    tracks_path: Path | None = None,
    output_dir: Path | None = None,
    show: bool = False,
    config: PossessionConfig | None = None,
) -> dict[str, Any]:
    config = config or PossessionConfig()
    project_root = find_project_root(project_root)
    video_path = (project_root / video_path).resolve() if not Path(video_path).is_absolute() else Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"No existe el video de entrada: {video_path}")

    if tracks_path is None:
        tracks_path = resolve_tracks_path_for_video(project_root, video_path)
    tracks_path = (project_root / tracks_path).resolve() if not Path(tracks_path).is_absolute() else Path(tracks_path)
    if not tracks_path.exists():
        raise FileNotFoundError(f"No existe el tracks JSON esperado: {tracks_path}")

    if output_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / DEFAULT_OUTPUT_DIR / f"{sanitize_video_stem(video_path.stem)}_{run_id}"
    else:
        output_dir = (project_root / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks_json(tracks_path)
    possession_df, summary = infer_team_possession(tracks=tracks, config=config)

    frame_csv_path = output_dir / "frame_possession.csv"
    frame_json_path = output_dir / "frame_possession.json"
    summary_json_path = output_dir / "summary.json"
    video_output_path = output_dir / f"{video_path.stem}_possession_annotated.mp4"

    possession_df.to_csv(frame_csv_path, index=False)
    frame_json_path.write_text(
        possession_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_payload = {
        **summary,
        "video_path": str(video_path),
        "tracks_path": str(tracks_path),
        "frame_possession_path": str(frame_csv_path),
    }
    summary_json_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    render_result = render_possession_video(
        video_path=video_path,
        tracks=tracks,
        possession_df=possession_df,
        project_root=project_root,
        output_path=video_output_path,
        show=show,
        config=config,
    )

    return {
        "output_dir": output_dir,
        "frame_possession_path": frame_csv_path,
        "frame_possession_json_path": frame_json_path,
        "summary_path": summary_json_path,
        "video_path": render_result["video_path"],
        "summary": summary_payload,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inferir posesion de balon por equipo y renderizarla sobre el video."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--tracks-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = process_team_possession(
        video_path=args.video_path,
        project_root=args.project_root,
        tracks_path=args.tracks_path,
        output_dir=args.output_dir,
        show=bool(args.show),
    )
    print(json.dumps(
        {
            "output_dir": str(result["output_dir"]),
            "frame_possession_path": str(result["frame_possession_path"]),
            "summary_path": str(result["summary_path"]),
            "video_path": str(result["video_path"]),
            "summary": result["summary"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
