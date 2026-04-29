from __future__ import annotations

from typing import Any

import pandas as pd

from .pathcrf_semantics import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    ensure_semantic_event_columns,
    infer_team_attack_directions,
    is_player_slot,
    safe_float,
    sort_events,
    team_attacks_right,
    team_id_from_player_id,
)


CORNER_MARGIN_M = 2.0
GOALKICK_MARGIN_M = 3.0
SIX_YARD_BOX_LENGTH_M = 5.5
SIX_YARD_BOX_WIDTH_M = 18.32
THROW_IN_MARGIN_M = 1.0


def _classify_start_event(
    *,
    period_id: Any,
    player_id: Any,
    start_x: Any,
    start_y: Any,
    attack_directions: dict[object, dict[str, bool]],
) -> str | None:
    x = safe_float(start_x)
    y = safe_float(start_y)
    if x is None or y is None:
        return None
    if not is_player_slot(player_id):
        return None

    team_id = team_id_from_player_id(player_id)
    attacks_right = team_attacks_right(period_id, team_id, attack_directions)

    defending_goal_x = 0.0 if attacks_right else PITCH_LENGTH_M
    attacking_goal_x = PITCH_LENGTH_M if attacks_right else 0.0

    corner_points = [
        (attacking_goal_x, 0.0),
        (attacking_goal_x, PITCH_WIDTH_M),
    ]
    if any(((x - cx) ** 2 + (y - cy) ** 2) <= (CORNER_MARGIN_M**2) for cx, cy in corner_points):
        return "corner"

    if y <= THROW_IN_MARGIN_M or y >= (PITCH_WIDTH_M - THROW_IN_MARGIN_M):
        return "throw_in"

    six_box_y_min = (PITCH_WIDTH_M / 2.0) - (SIX_YARD_BOX_WIDTH_M / 2.0)
    six_box_y_max = (PITCH_WIDTH_M / 2.0) + (SIX_YARD_BOX_WIDTH_M / 2.0)
    near_goal_line_box = (
        abs(x - (defending_goal_x + (SIX_YARD_BOX_LENGTH_M if attacks_right else -SIX_YARD_BOX_LENGTH_M)))
        <= GOALKICK_MARGIN_M
    )
    if six_box_y_min <= y <= six_box_y_max and near_goal_line_box:
        return "goalkick"

    return None


def classify_setpieces(events: pd.DataFrame) -> pd.DataFrame:
    result = ensure_semantic_event_columns(sort_events(events))
    if result.empty:
        return result

    attack_directions = infer_team_attack_directions(result)
    first_events = result.groupby("episode_id", sort=False, dropna=False).head(1)

    for idx, row in first_events.iterrows():
        set_piece_type = _classify_start_event(
            period_id=row.get("period_id"),
            player_id=row.get("player_id"),
            start_x=row.get("start_x"),
            start_y=row.get("start_y"),
            attack_directions=attack_directions,
        )
        if set_piece_type is None:
            continue
        result.at[idx, "event_type_semantic"] = str(set_piece_type)
        result.at[idx, "semantic_source"] = "set_piece_rule"
        result.at[idx, "semantic_confidence"] = 1.0

    result["event_type"] = result["event_type_semantic"]
    return result


def classify_episode_starts(events: pd.DataFrame) -> pd.DataFrame:
    return classify_setpieces(events)
