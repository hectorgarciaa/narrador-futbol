from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_ai.pathcrf_slot_mapping import canonical_id_to_person_slot


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PITCH_CENTER_Y = PITCH_WIDTH_M / 2.0

GOAL_LEFT = np.asarray([0.0, PITCH_CENTER_Y], dtype=np.float32)
GOAL_RIGHT = np.asarray([PITCH_LENGTH_M, PITCH_CENTER_Y], dtype=np.float32)

PLAYER_PREFIXES = ("home_", "away_")
OUTSIDE_PREFIXES = ("out_",)


def safe_float(value: Any) -> float | None:
    try:
        casted = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(casted):
        return None
    return casted


def team_id_from_player_id(player_id: Any) -> str | None:
    text = str(player_id or "").strip()
    canonical_slot = canonical_id_to_person_slot(text)
    if canonical_slot:
        text = canonical_slot
    if text.startswith("home_"):
        return "home"
    if text.startswith("away_"):
        return "away"
    return None


def is_player_slot(player_id: Any) -> bool:
    text = str(player_id or "").strip()
    canonical_slot = canonical_id_to_person_slot(text)
    if canonical_slot:
        text = canonical_slot
    return text.startswith(PLAYER_PREFIXES)


def is_outside_node(player_id: Any) -> bool:
    text = str(player_id or "").strip()
    return text.startswith(OUTSIDE_PREFIXES)


def ensure_semantic_event_columns(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    if "event_type" not in result.columns:
        result["event_type"] = pd.Series(dtype=object)
    if "event_type_raw" not in result.columns:
        result["event_type_raw"] = result["event_type"].astype(object)
    if "event_type_semantic" not in result.columns:
        result["event_type_semantic"] = result["event_type_raw"].astype(object)
    if "semantic_source" not in result.columns:
        result["semantic_source"] = "raw"
    if "semantic_confidence" not in result.columns:
        result["semantic_confidence"] = np.where(
            result["event_type_semantic"].notna(),
            1.0,
            np.nan,
        )
    result["event_type"] = result["event_type_semantic"]
    return result


def sort_events(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    sort_cols = [column for column in ("period_id", "episode_id", "frame_id") if column in result.columns]
    if sort_cols:
        result = result.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return result


def infer_team_attack_directions(events: pd.DataFrame) -> dict[object, dict[str, bool]]:
    working = ensure_semantic_event_columns(sort_events(events))
    if "team_id" not in working.columns:
        working["team_id"] = working["player_id"].map(team_id_from_player_id)

    for column in ("start_x", "end_x"):
        if column not in working.columns:
            working[column] = np.nan

    valid = working[
        working["team_id"].isin(["home", "away"])
        & working["start_x"].map(safe_float).notna()
        & working["end_x"].map(safe_float).notna()
        & ~working["event_type_semantic"].isin(["out", "control"])
    ].copy()

    valid["dx"] = pd.to_numeric(valid["end_x"], errors="coerce") - pd.to_numeric(
        valid["start_x"],
        errors="coerce",
    )

    mapping: dict[object, dict[str, bool]] = {}
    period_values = (
        list(working["period_id"].dropna().unique())
        if "period_id" in working.columns
        else [1]
    )
    if not period_values:
        period_values = [1]

    for period_id in period_values:
        period_map = {"home": True, "away": False}
        period_valid = valid[valid["period_id"] == period_id] if "period_id" in valid.columns else valid
        if not period_valid.empty:
            team_dx = period_valid.groupby("team_id")["dx"].mean()
            for team_id, mean_dx in team_dx.items():
                if pd.notna(mean_dx):
                    period_map[str(team_id)] = float(mean_dx) >= 0.0
            if "home" in team_dx and "away" not in team_dx:
                period_map["away"] = not period_map["home"]
            elif "away" in team_dx and "home" not in team_dx:
                period_map["home"] = not period_map["away"]
        mapping[period_id] = period_map
    return mapping


def team_attacks_right(
    period_id: Any,
    team_id: Any,
    attack_directions: dict[object, dict[str, bool]] | None = None,
) -> bool:
    team_key = str(team_id or "").strip()
    if team_key not in {"home", "away"}:
        return team_key == "home"
    if attack_directions is None:
        return team_key == "home"
    period_map = attack_directions.get(period_id)
    if period_map is None:
        return team_key == "home"
    return bool(period_map.get(team_key, team_key == "home"))


def target_goal_for_team(
    period_id: Any,
    team_id: Any,
    attack_directions: dict[object, dict[str, bool]] | None = None,
) -> np.ndarray:
    return GOAL_RIGHT if team_attacks_right(period_id, team_id, attack_directions) else GOAL_LEFT


def parse_timestamp_seconds(value: Any) -> float | None:
    numeric = safe_float(value)
    if numeric is not None:
        return numeric

    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return (minutes * 60.0) + seconds
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return (hours * 3600.0) + (minutes * 60.0) + seconds
    except ValueError:
        return None
    return None
