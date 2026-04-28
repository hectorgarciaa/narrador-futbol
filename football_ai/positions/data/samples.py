from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .core import FeatureSpec


def infer_attack_direction_by_team(
    observations: pd.DataFrame,
    goalkeeper_class_name: str = "goalkeeper",
) -> tuple[dict[str, int], pd.DataFrame]:
    if observations.empty:
        return {}, pd.DataFrame(columns=["team_id", "method", "reference_median_x"])

    valid = observations[observations["team_id"].notna()].copy()
    if valid.empty:
        return {}, pd.DataFrame(columns=["team_id", "method", "reference_median_x"])

    teams = sorted(valid["team_id"].unique().tolist())
    goalkeepers = valid[valid["class_name"] == goalkeeper_class_name]
    if len(teams) == 2 and not goalkeepers.empty and set(teams).issubset(set(goalkeepers["team_id"].unique().tolist())):
        ref = goalkeepers.groupby("team_id")["x"].median()
        method = "goalkeeper_median_x"
    else:
        ref = valid.groupby("team_id")["x"].median()
        method = "team_median_x"

    ref = ref.sort_values(ascending=True)
    direction_by_team: dict[str, int] = {}
    if len(ref.index) == 2:
        direction_by_team[str(ref.index[0])] = +1
        direction_by_team[str(ref.index[1])] = -1
    else:
        for team_id, x_med in ref.items():
            direction_by_team[str(team_id)] = +1 if float(x_med) <= 0.5 else -1

    details = pd.DataFrame(
        {
            "team_id": [str(team_id) for team_id in ref.index.tolist()],
            "method": [method] * len(ref.index),
            "reference_median_x": [float(v) for v in ref.tolist()],
            "attack_direction": [direction_by_team[str(team)] for team in ref.index.tolist()],
        }
    )
    return direction_by_team, details


def _oriented_xy(x: np.ndarray, y: np.ndarray, attack_direction: int) -> tuple[np.ndarray, np.ndarray]:
    if int(attack_direction) >= 0:
        return x, y
    return 1.0 - x, 1.0 - y


def build_role_samples(
    observations_with_roles: pd.DataFrame,
    attack_direction_by_team: Mapping[str, int],
    max_teammates: int = 10,
    drop_unlabeled: bool = True,
    include_all_targets: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, FeatureSpec]:
    df = observations_with_roles.copy()
    if df.empty:
        return df, np.zeros((0, int(max_teammates), 0), dtype=np.float32), np.zeros((0, int(max_teammates)), dtype=bool), FeatureSpec(tuple(), tuple())

    if drop_unlabeled:
        df = df[df["role_label"].notna()].copy()
    if df.empty:
        return df, np.zeros((0, int(max_teammates), 0), dtype=np.float32), np.zeros((0, int(max_teammates)), dtype=bool), FeatureSpec(tuple(), tuple())

    objective_feature_names = (
        "x",
        "y",
        "dist_left_sideline",
        "dist_right_sideline",
        "dist_own_goal",
        "dist_opponent_goal",
        "team_centroid_x",
        "team_centroid_y",
        "dx_centroid",
        "dy_centroid",
        "rank_x_team",
        "rank_y_team",
        "vx",
        "vy",
    )
    teammate_feature_names = (
        "dx_to_obj",
        "dy_to_obj",
        "dist_to_obj",
        "x",
        "y",
        "dist_left_sideline",
        "dist_right_sideline",
        "dist_own_goal",
        "dist_opponent_goal",
        "vx",
        "vy",
    )

    rows: list[dict[str, Any]] = []
    teammates_tensor: list[np.ndarray] = []
    teammates_mask: list[np.ndarray] = []
    for (match_id, frame_id, team_id), team_frame in observations_with_roles.groupby(["match_id", "frame_id", "team_id"], sort=True):
        team_frame = team_frame.copy()
        if "role_label" not in team_frame.columns:
            continue

        attack_direction = int(attack_direction_by_team.get(str(team_id), +1))
        team_frame["x_ori"], team_frame["y_ori"] = _oriented_xy(
            team_frame["x"].to_numpy(dtype=np.float32),
            team_frame["y"].to_numpy(dtype=np.float32),
            attack_direction,
        )
        team_frame["vx_ori"] = team_frame["vx"].to_numpy(dtype=np.float32) if "vx" in team_frame.columns else np.zeros((len(team_frame),), dtype=np.float32)
        team_frame["vy_ori"] = team_frame["vy"].to_numpy(dtype=np.float32) if "vy" in team_frame.columns else np.zeros((len(team_frame),), dtype=np.float32)
        if attack_direction < 0:
            team_frame["vx_ori"] = -team_frame["vx_ori"]
            team_frame["vy_ori"] = -team_frame["vy_ori"]
        team_frame["rank_x_team"] = team_frame["x_ori"].rank(method="dense", ascending=True).astype(np.float32)
        team_frame["rank_y_team"] = team_frame["y_ori"].rank(method="dense", ascending=True).astype(np.float32)

        if include_all_targets:
            objectives = team_frame.copy()
        elif drop_unlabeled:
            objectives = team_frame[team_frame["role_label"].notna()].copy()
        elif "role_inference_target" in team_frame.columns:
            objectives = team_frame[pd.to_numeric(team_frame["role_inference_target"], errors="coerce").fillna(0).astype(int) > 0].copy()
        else:
            objectives = team_frame.copy()
        if objectives.empty:
            continue

        for _, obj in objectives.iterrows():
            teammates = team_frame[team_frame["player_id"] != obj["player_id"]].copy()
            x_obj = float(obj["x_ori"])
            y_obj = float(obj["y_ori"])
            vx_obj = float(obj.get("vx_ori", 0.0))
            vy_obj = float(obj.get("vy_ori", 0.0))
            centroid_x = x_obj if teammates.empty else float(teammates["x_ori"].mean())
            centroid_y = y_obj if teammates.empty else float(teammates["y_ori"].mean())

            rows.append(
                {
                    "match_id": str(match_id),
                    "frame_id": int(frame_id),
                    "team_id": str(team_id),
                    "player_id": int(obj["player_id"]),
                    "label": str(obj["role_label"]),
                    "attack_direction": int(attack_direction),
                    "visible_teammates": int(len(teammates)),
                    "x": x_obj,
                    "y": y_obj,
                    "dist_left_sideline": y_obj,
                    "dist_right_sideline": 1.0 - y_obj,
                    "dist_own_goal": x_obj,
                    "dist_opponent_goal": 1.0 - x_obj,
                    "team_centroid_x": centroid_x,
                    "team_centroid_y": centroid_y,
                    "dx_centroid": x_obj - centroid_x,
                    "dy_centroid": y_obj - centroid_y,
                    "rank_x_team": float(obj["rank_x_team"]),
                    "rank_y_team": float(obj["rank_y_team"]),
                    "vx": vx_obj,
                    "vy": vy_obj,
                }
            )

            teammate_features = np.zeros((int(max_teammates), len(teammate_feature_names)), dtype=np.float32)
            teammate_exists = np.zeros((int(max_teammates),), dtype=bool)
            if not teammates.empty:
                teammates["dx_to_obj"] = teammates["x_ori"] - x_obj
                teammates["dy_to_obj"] = teammates["y_ori"] - y_obj
                teammates["dist_to_obj"] = np.sqrt((teammates["dx_to_obj"] ** 2) + (teammates["dy_to_obj"] ** 2))
                teammates = teammates.sort_values(["dist_to_obj", "player_id"], ascending=[True, True])
                for idx_tm, (_, tm) in enumerate(teammates.head(int(max_teammates)).iterrows()):
                    teammate_features[idx_tm, :] = np.array(
                        [
                            float(tm["dx_to_obj"]),
                            float(tm["dy_to_obj"]),
                            float(tm["dist_to_obj"]),
                            float(tm["x_ori"]),
                            float(tm["y_ori"]),
                            float(tm["y_ori"]),
                            float(1.0 - tm["y_ori"]),
                            float(tm["x_ori"]),
                            float(1.0 - tm["x_ori"]),
                            float(tm.get("vx_ori", 0.0)),
                            float(tm.get("vy_ori", 0.0)),
                        ],
                        dtype=np.float32,
                    )
                    teammate_exists[idx_tm] = True

            teammates_tensor.append(teammate_features)
            teammates_mask.append(teammate_exists)

    if not rows:
        return (
            pd.DataFrame(columns=["match_id", "frame_id", "team_id", "player_id", "label"]),
            np.zeros((0, int(max_teammates), len(teammate_feature_names)), dtype=np.float32),
            np.zeros((0, int(max_teammates)), dtype=bool),
            FeatureSpec(objective_feature_names, teammate_feature_names),
        )

    return (
        pd.DataFrame(rows),
        np.asarray(teammates_tensor, dtype=np.float32),
        np.asarray(teammates_mask, dtype=bool),
        FeatureSpec(objective_feature_names, teammate_feature_names),
    )
