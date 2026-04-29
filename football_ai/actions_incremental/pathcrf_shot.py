from __future__ import annotations

import numpy as np
import pandas as pd

from .pathcrf_semantics import (
    GOAL_LEFT,
    GOAL_RIGHT,
    PITCH_CENTER_Y,
    PITCH_LENGTH_M,
    ensure_semantic_event_columns,
    infer_team_attack_directions,
    sort_events,
    target_goal_for_team,
    team_attacks_right,
    team_id_from_player_id,
)


MAX_EFFECTIVE_DISTANCE_M = 50.0
ATTACKING_ZONE_THRESHOLD_M = 70.0
GK_INTERVENTION_MAX_DISTANCE_M = 5.0
LOOKAHEAD_STEPS = 10
MAX_GOAL_OUT_FRAME_DIFF = 100

SPEED_DISTANCE_BOUNDARY_M = 18.0
SPEED_THRESHOLD_NEAR_MPS = 12.0
SPEED_THRESHOLD_FAR_BASE_MPS = 12.0
SPEED_THRESHOLD_FAR_SLOPE = 0.25

WEIGHT_SPEED = 1.0
WEIGHT_DIRECTION = 40.0
WEIGHT_POSITION = 70.0
WEIGHT_CROSS_PENALTY = 50.0

BONUS_ATTACKING_TEAM = 25.0
BONUS_GK_INTERVENTION = 30.0
BONUS_GOAL_OUT = 25.0

BETA_LATERAL = 2.2
BASE_SCORE_VALUE = 47.0
CUTOFF_SCORE = 145.0


def _prepare_prediction_data(events: pd.DataFrame, fps: float) -> pd.DataFrame:
    pred_events = ensure_semantic_event_columns(sort_events(events))
    if pred_events.empty:
        return pred_events

    group_cols = ["period_id", "episode_id"]
    pred_events["team_id"] = pred_events["player_id"].map(team_id_from_player_id)
    pred_events["next_player_id"] = pred_events.groupby(group_cols, dropna=False)["player_id"].shift(-1)
    pred_events["next_team_id"] = pred_events["next_player_id"].map(team_id_from_player_id)
    pred_events["next_event_type"] = pred_events.groupby(group_cols, dropna=False)["event_type_semantic"].shift(-1)
    pred_events["next_frame_id"] = pred_events.groupby(group_cols, dropna=False)["frame_id"].shift(-1)
    pred_events["next_start_x"] = pred_events.groupby(group_cols, dropna=False)["start_x"].shift(-1)
    pred_events["next_start_y"] = pred_events.groupby(group_cols, dropna=False)["start_y"].shift(-1)

    pred_events["start_x"] = pd.to_numeric(pred_events["start_x"], errors="coerce")
    pred_events["start_y"] = pd.to_numeric(pred_events["start_y"], errors="coerce")
    pred_events["end_x"] = pd.to_numeric(pred_events["end_x"], errors="coerce")
    pred_events["end_y"] = pd.to_numeric(pred_events["end_y"], errors="coerce")
    pred_events["frame_id"] = pd.to_numeric(pred_events["frame_id"], errors="coerce")

    pred_events["resolved_end_x"] = pred_events["end_x"]
    pred_events["resolved_end_y"] = pred_events["end_y"]
    pred_events.loc[
        pred_events["resolved_end_x"].isna() & pred_events["next_start_x"].notna(),
        "resolved_end_x",
    ] = pred_events["next_start_x"]
    pred_events.loc[
        pred_events["resolved_end_y"].isna() & pred_events["next_start_y"].notna(),
        "resolved_end_y",
    ] = pred_events["next_start_y"]

    curr_pos = pred_events[["start_x", "start_y"]].to_numpy(dtype=np.float32)
    pred_events["dist_left"] = np.linalg.norm(curr_pos - GOAL_LEFT, axis=1)
    pred_events["dist_right"] = np.linalg.norm(curr_pos - GOAL_RIGHT, axis=1)

    attack_directions = infer_team_attack_directions(pred_events)
    target_goals = np.asarray(
        [
            target_goal_for_team(row.period_id, row.team_id, attack_directions)
            if pd.notna(row.start_x) and pd.notna(row.start_y)
            else np.asarray([np.nan, np.nan], dtype=np.float32)
            for row in pred_events.itertuples(index=False)
        ],
        dtype=np.float32,
    )
    pred_events["target_goal_x"] = target_goals[:, 0]
    pred_events["target_goal_y"] = target_goals[:, 1]
    pred_events["dist_to_goal"] = np.sqrt(
        (pred_events["target_goal_x"] - pred_events["start_x"]) ** 2
        + (pred_events["target_goal_y"] - pred_events["start_y"]) ** 2
    )

    dx = pred_events["resolved_end_x"] - pred_events["start_x"]
    dy = pred_events["resolved_end_y"] - pred_events["start_y"]
    distance = np.sqrt((dx**2) + (dy**2))
    frame_diff = pred_events["next_frame_id"] - pred_events["frame_id"]
    pred_events["speed_mps"] = np.where(
        frame_diff > 0,
        distance / (frame_diff / max(float(fps), 1e-6)),
        0.0,
    )
    pred_events["attack_direction_right"] = [
        team_attacks_right(row.period_id, row.team_id, attack_directions)
        for row in pred_events.itertuples(index=False)
    ]
    return pred_events


def _infer_goalkeepers(pred_events: pd.DataFrame) -> dict[object, dict[str, str]]:
    gk_map: dict[object, dict[str, str]] = {}
    valid = pred_events.dropna(subset=["period_id", "team_id", "player_id", "start_x"]).copy()
    if valid.empty:
        return gk_map

    avg_x = valid.groupby(["period_id", "team_id", "player_id"], dropna=False)["start_x"].mean().reset_index()
    attack_directions = infer_team_attack_directions(pred_events)

    for period_id in avg_x["period_id"].dropna().unique():
        gk_map[period_id] = {}
        for team_id in ("home", "away"):
            team_rows = avg_x[(avg_x["period_id"] == period_id) & (avg_x["team_id"] == team_id)]
            if team_rows.empty:
                continue
            attacks_right = team_attacks_right(period_id, team_id, attack_directions)
            if attacks_right:
                gk_row = team_rows.loc[team_rows["start_x"].idxmin()]
            else:
                gk_row = team_rows.loc[team_rows["start_x"].idxmax()]
            gk_map[period_id][team_id] = str(gk_row["player_id"])
    return gk_map


def _get_goal_out_bonus_mask(pred_events: pd.DataFrame) -> pd.Series:
    rule_mask = pd.Series(False, index=pred_events.index)
    for offset in range(1, LOOKAHEAD_STEPS + 1):
        future_type = pred_events["event_type_semantic"].shift(-offset)
        future_x = pred_events["start_x"].shift(-offset)
        future_frame = pred_events["frame_id"].shift(-offset)
        future_period = pred_events["period_id"].shift(-offset)
        future_episode = pred_events["episode_id"].shift(-offset)
        is_out = future_type == "out"
        is_goal_line = (future_x <= 0.0) | (future_x >= PITCH_LENGTH_M)
        is_within_time = (future_frame - pred_events["frame_id"]) <= MAX_GOAL_OUT_FRAME_DIFF
        same_context = (future_period == pred_events["period_id"]) & (future_episode == pred_events["episode_id"])
        rule_mask = rule_mask | (is_out & is_goal_line & is_within_time & same_context).fillna(False)
    return rule_mask


def _is_in_attacking_third(pred_events: pd.DataFrame) -> pd.Series:
    return np.where(
        pred_events["attack_direction_right"].astype(bool),
        pred_events["start_x"] > ATTACKING_ZONE_THRESHOLD_M,
        pred_events["start_x"] < (PITCH_LENGTH_M - ATTACKING_ZONE_THRESHOLD_M),
    )


def _is_attacking_team_event(pred_events: pd.DataFrame) -> pd.Series:
    dx = pred_events["resolved_end_x"] - pred_events["start_x"]
    toward_right = dx > 0
    return np.where(pred_events["attack_direction_right"].astype(bool), toward_right, ~toward_right)


def _has_gk_intervention(pred_events: pd.DataFrame, gk_map: dict[object, dict[str, str]]) -> pd.Series:
    next_start_x = pred_events["next_start_x"]
    next_start_y = pred_events["next_start_y"]
    turnover = (
        (pred_events["team_id"] != pred_events["next_team_id"])
        & pred_events["team_id"].notna()
        & pred_events["next_team_id"].notna()
    )

    def _next_is_goalkeeper(row: pd.Series) -> bool:
        period_id = row.get("period_id")
        next_team_id = row.get("next_team_id")
        next_player_id = row.get("next_player_id")
        if pd.isna(period_id) or pd.isna(next_team_id) or pd.isna(next_player_id):
            return False
        return str(next_player_id) == str(gk_map.get(period_id, {}).get(str(next_team_id)))

    is_next_gk = pred_events.apply(_next_is_goalkeeper, axis=1)
    next_goal_x = np.where(pred_events["next_team_id"] == "home", 0.0, PITCH_LENGTH_M)
    dist_to_goal = np.sqrt(((next_start_x - next_goal_x) ** 2) + ((next_start_y - PITCH_CENTER_Y) ** 2))
    return turnover & is_next_gk & (dist_to_goal < GK_INTERVENTION_MAX_DISTANCE_M) & (pred_events["next_event_type"] != "out")


def _calculate_geometric_score(pred_events: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    vec_x = pred_events["resolved_end_x"] - pred_events["start_x"]
    vec_y = pred_events["resolved_end_y"] - pred_events["start_y"]
    goal_vec_x = pred_events["target_goal_x"] - pred_events["start_x"]
    goal_vec_y = pred_events["target_goal_y"] - pred_events["start_y"]

    norm_pass = np.sqrt((vec_x**2) + (vec_y**2))
    norm_goal = np.sqrt((goal_vec_x**2) + (goal_vec_y**2))
    dot = (vec_x * goal_vec_x) + (vec_y * goal_vec_y)
    similarity = np.divide(
        dot,
        (norm_pass * norm_goal),
        out=np.zeros_like(dot, dtype=np.float32),
        where=(norm_pass * norm_goal) > 0,
    )
    base_dir_score = similarity * WEIGHT_DIRECTION

    goal_dx = np.abs(pred_events["target_goal_x"] - pred_events["start_x"])
    goal_dy = np.abs(PITCH_CENTER_Y - pred_events["start_y"])
    effective_distance = np.sqrt((goal_dx**2) + ((BETA_LATERAL * goal_dy) ** 2))
    base_pos_score = np.maximum(0.0, 1.0 - (effective_distance / MAX_EFFECTIVE_DISTANCE_M)) * WEIGHT_POSITION

    lateral_ratio = np.abs(vec_y) / (np.abs(vec_x) + 1e-6)
    penalty_lat = (lateral_ratio > 2.0) & (goal_dy > 12.0)
    penalty_sideline = goal_dy > 30.0
    penalty_deep_wide = (goal_dy > 12.0) & (pred_events["dist_to_goal"] > 20.0) & (similarity < 0.9)
    penalty_score = (penalty_lat | penalty_sideline | penalty_deep_wide).astype(float) * WEIGHT_CROSS_PENALTY

    return (base_dir_score + base_pos_score - penalty_score), pd.Series(similarity, index=pred_events.index)


def _calculate_speed_score(pred_events: pd.DataFrame) -> pd.Series:
    thresholds = np.where(
        pred_events["dist_to_goal"] < SPEED_DISTANCE_BOUNDARY_M,
        SPEED_THRESHOLD_NEAR_MPS,
        SPEED_THRESHOLD_FAR_BASE_MPS + ((pred_events["dist_to_goal"] - SPEED_DISTANCE_BOUNDARY_M) * SPEED_THRESHOLD_FAR_SLOPE),
    )
    surplus = pred_events["speed_mps"] - thresholds
    return pd.Series(np.clip(surplus, a_min=None, a_max=20.0), index=pred_events.index)


def apply_shot_heuristic(events: pd.DataFrame, fps: float = 25.0) -> pd.DataFrame:
    result = ensure_semantic_event_columns(sort_events(events))
    if result.empty:
        return result

    pred_events = _prepare_prediction_data(result, fps=fps)
    gk_map = _infer_goalkeepers(pred_events)
    is_kick_event = pred_events["event_type_semantic"] == "kick"
    has_goal_out = _get_goal_out_bonus_mask(pred_events)
    in_attacking_zone = _is_in_attacking_third(pred_events)
    attacking_team_event = _is_attacking_team_event(pred_events)
    gk_intervention = _has_gk_intervention(pred_events, gk_map)

    geometric_score, direction_similarity = _calculate_geometric_score(pred_events)
    speed_score = _calculate_speed_score(pred_events) * WEIGHT_SPEED
    total_score = (
        BASE_SCORE_VALUE
        + speed_score
        + geometric_score
        + (attacking_team_event.astype(float) * BONUS_ATTACKING_TEAM)
        + (gk_intervention.astype(float) * BONUS_GK_INTERVENTION)
        + (has_goal_out.astype(float) * BONUS_GOAL_OUT)
    )

    is_pred_shot = (
        is_kick_event
        & in_attacking_zone
        & pred_events["dist_to_goal"].lt(MAX_EFFECTIVE_DISTANCE_M)
        & total_score.ge(CUTOFF_SCORE)
    )

    result["shot_score_total"] = total_score
    result["shot_score_speed"] = speed_score
    result["shot_score_geom"] = geometric_score
    result["shot_direction_similarity"] = direction_similarity
    result["is_shot_candidate"] = is_kick_event & in_attacking_zone
    result["is_pred_shot"] = is_pred_shot

    result.loc[is_pred_shot, "event_type_semantic"] = "shot"
    result.loc[is_pred_shot, "semantic_source"] = "shot_heuristic"
    result.loc[is_pred_shot, "semantic_confidence"] = 1.0
    result["event_type"] = result["event_type_semantic"]
    return result
