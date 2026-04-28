from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import ROLE_LABELS_V1


def build_periodic_label_schedule(
    observations: pd.DataFrame,
    fps: float,
    every_seconds: float = 6.0,
    start_seconds: float = 0.0,
    min_players: int = 18,
    window_frames: int = 0,
) -> pd.DataFrame:
    columns = ["target_second", "target_frame", "frame_second", "frame_id", "num_players"]
    if observations.empty:
        return pd.DataFrame(columns=columns)
    if not np.isfinite(fps) or fps <= 1e-6:
        raise ValueError(f"FPS inválido: {fps}")
    if every_seconds <= 0:
        raise ValueError(f"every_seconds debe ser > 0 (recibido {every_seconds}).")
    if int(window_frames) < 0:
        raise ValueError(f"window_frames debe ser >= 0 (recibido {window_frames}).")

    per_frame_players = observations.groupby("frame_id").size().sort_index()
    eligible = per_frame_players[per_frame_players >= int(min_players)]
    if eligible.empty:
        eligible = per_frame_players
    if eligible.empty:
        return pd.DataFrame(columns=columns)

    available_frames = eligible.index.to_numpy(dtype=np.int64)
    step_frames = max(1, int(round(float(every_seconds) * float(fps))))
    start_frame = max(0, int(round(float(start_seconds) * float(fps))))
    rows: list[dict[str, Any]] = []
    for target_frame in range(start_frame, int(per_frame_players.index.max()) + 1, step_frames):
        selected_frame = _pick_schedule_frame(eligible, available_frames, target_frame, int(window_frames))
        if selected_frame is None:
            continue
        rows.append(
            {
                "target_second": float(target_frame / fps),
                "target_frame": int(target_frame),
                "frame_second": float(selected_frame / fps),
                "frame_id": int(selected_frame),
                "num_players": int(per_frame_players.loc[selected_frame]),
            }
        )
    return pd.DataFrame(rows, columns=columns).drop_duplicates(subset=["frame_id"], keep="first").sort_values(["target_frame", "frame_id"]).reset_index(drop=True)


def build_periodic_role_label_template(
    observations: pd.DataFrame,
    match_id: str,
    video_path: Path,
    fps: float,
    every_seconds: float = 6.0,
    start_seconds: float = 0.0,
    min_players: int = 18,
    window_frames: int = 0,
) -> dict[str, Any]:
    schedule_df = build_periodic_label_schedule(
        observations=observations,
        fps=fps,
        every_seconds=every_seconds,
        start_seconds=start_seconds,
        min_players=min_players,
        window_frames=window_frames,
    )
    labels: list[dict[str, Any]] = []
    for _, schedule_row in schedule_df.iterrows():
        frame_id = int(schedule_row["frame_id"])
        frame_observations = observations[observations["frame_id"] == frame_id].sort_values(["team_id", "player_id", "class_name"])
        for _, row in frame_observations.iterrows():
            labels.append(
                {
                    "frame_id": frame_id,
                    "frame_second": float(schedule_row["frame_second"]),
                    "target_second": float(schedule_row["target_second"]),
                    "team_id": str(row["team_id"]),
                    "player_id": int(row["player_id"]),
                    "class_name": str(row["class_name"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "x_m": float(row["x_m"]),
                    "y_m": float(row["y_m"]),
                    "bbox": _bbox_to_list(row.get("bbox")),
                    "role_label": None,
                }
            )
    return {
        "schema_version": 2,
        "created_at": datetime.now().isoformat(),
        "match_id": str(match_id),
        "video_path": str(video_path),
        "fps": float(fps),
        "label_every_seconds": float(every_seconds),
        "label_start_seconds": float(start_seconds),
        "min_players_per_label_frame": int(min_players),
        "schedule": schedule_df.to_dict(orient="records"),
        "labels": labels,
    }


def save_role_label_template_json(template: Mapping[str, Any], output_path: Path, overwrite: bool = False) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    return output_path


def load_role_label_entries_json(label_json_path: Path) -> pd.DataFrame:
    with Path(label_json_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("labels", []) if isinstance(payload, Mapping) else payload
    if not isinstance(entries, list) or not entries:
        return pd.DataFrame(columns=["frame_id", "team_id", "player_id", "role_label"])

    labels_df = pd.DataFrame(entries)
    missing = {"frame_id", "team_id", "player_id"}.difference(labels_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en labels JSON ({', '.join(sorted(missing))}) en {label_json_path}.")
    labels_df["frame_id"] = pd.to_numeric(labels_df["frame_id"], errors="coerce").astype("Int64")
    labels_df["player_id"] = pd.to_numeric(labels_df["player_id"], errors="coerce").astype("Int64")
    labels_df["team_id"] = labels_df["team_id"].astype(str)
    labels_df["role_label"] = labels_df.get("role_label", pd.Series([pd.NA] * len(labels_df))).astype("string").str.strip()
    labels_df.loc[labels_df["role_label"] == "", "role_label"] = pd.NA
    labels_df = labels_df.dropna(subset=["frame_id", "player_id"]).copy()
    labels_df["frame_id"] = labels_df["frame_id"].astype(int)
    labels_df["player_id"] = labels_df["player_id"].astype(int)
    return labels_df.sort_values(["frame_id", "team_id", "player_id"]).drop_duplicates(subset=["frame_id", "team_id", "player_id"], keep="last").reset_index(drop=True)


def validate_periodic_role_labels(
    observations: pd.DataFrame,
    label_entries: pd.DataFrame,
    allowed_labels: Sequence[str] = ROLE_LABELS_V1,
) -> dict[str, Any]:
    if observations.empty:
        return _empty_validation(len(label_entries), 0)
    if label_entries.empty:
        return _empty_validation(0, len(observations))
    normalized = _normalized_labeled_entries(label_entries)
    labeled = normalized[normalized["role_label"].notna()].copy()
    merge_keys = ["frame_id", "team_id", "player_id"]
    invalid = sorted(labeled.loc[~labeled["role_label"].isin(set(allowed_labels)), "role_label"].astype(str).unique().tolist())
    unmatched = 0
    if not labeled.empty:
        check_df = labeled[merge_keys].drop_duplicates().merge(observations[merge_keys].drop_duplicates(), on=merge_keys, how="left", indicator=True)
        unmatched = int((check_df["_merge"] == "left_only").sum())
    covered_rows = int(observations.merge(labeled[merge_keys + ["role_label"]], on=merge_keys, how="left")["role_label"].notna().sum())
    return {
        "invalid_labels": invalid,
        "total_entries": int(len(normalized)),
        "labeled_entries": int(len(labeled)),
        "expected_frames": int(normalized["frame_id"].nunique()),
        "labeled_frames": int(labeled["frame_id"].nunique()) if not labeled.empty else 0,
        "unmatched_labeled_entries": unmatched,
        "covered_rows": covered_rows,
        "total_rows": int(len(observations)),
        "coverage_ratio": float(covered_rows / len(observations)),
    }


def apply_periodic_role_labels(observations: pd.DataFrame, label_entries: pd.DataFrame) -> pd.DataFrame:
    result = observations.copy()
    if result.empty:
        result["role_label"] = pd.NA
        return result
    labels = _normalized_labeled_entries(label_entries)
    if labels.empty:
        result["role_label"] = pd.NA
        return result
    return result.merge(labels[["frame_id", "team_id", "player_id", "role_label"]], on=["frame_id", "team_id", "player_id"], how="left")


def upsert_role_labels_in_template_json(
    label_json_path: Path,
    frame_role_maps: Mapping[Any, Mapping[Any, Mapping[Any, Any]]],
) -> dict[str, Any]:
    path = Path(label_json_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Formato inválido en {path}; se esperaba un objeto JSON.")
    labels = payload.get("labels", [])
    if not isinstance(labels, list):
        raise ValueError(f"Formato inválido en {path}; 'labels' debe ser una lista.")

    key_to_idx = {}
    for idx, row in enumerate(labels):
        if isinstance(row, Mapping):
            try:
                key_to_idx[(int(row.get("frame_id")), str(row.get("team_id")), int(row.get("player_id")))] = idx
            except (TypeError, ValueError):
                pass

    updated = 0
    inserted = 0
    for frame_id_raw, team_map in frame_role_maps.items():
        if not isinstance(team_map, Mapping):
            continue
        try:
            frame_id = int(frame_id_raw)
        except (TypeError, ValueError):
            continue
        for team_id_raw, player_map in team_map.items():
            if not isinstance(player_map, Mapping):
                continue
            for player_id_raw, role_raw in player_map.items():
                try:
                    player_id = int(player_id_raw)
                except (TypeError, ValueError):
                    continue
                role_label = None if role_raw is None else str(role_raw).strip() or None
                key = (frame_id, str(team_id_raw), player_id)
                row = {"frame_id": frame_id, "team_id": str(team_id_raw), "player_id": player_id, "role_label": role_label}
                if key in key_to_idx:
                    labels[key_to_idx[key]] = {**dict(labels[key_to_idx[key]]), **row}
                    updated += 1
                else:
                    labels.append(row)
                    key_to_idx[key] = len(labels) - 1
                    inserted += 1
    payload = dict(payload)
    payload["labels"] = labels
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"label_json_path": str(path), "updated": int(updated), "inserted": int(inserted), "total_labels": int(len(labels))}


def apply_periodic_role_labels_propagated(
    observations: pd.DataFrame,
    label_entries: pd.DataFrame,
    schedule_frames: Sequence[int] | None = None,
) -> pd.DataFrame:
    result = observations.copy()
    result["role_label"] = pd.NA
    if result.empty:
        return result
    labels = _normalized_labeled_entries(label_entries)
    labels = labels[labels["role_label"].notna()].copy()
    if labels.empty:
        return result

    keyframes = sorted(labels["frame_id"].unique().tolist()) if schedule_frames is None else sorted({int(frame) for frame in schedule_frames})
    if not keyframes:
        return result
    max_obs_frame = int(result["frame_id"].max())
    for idx, start_frame in enumerate(keyframes):
        end_frame = keyframes[idx + 1] if idx + 1 < len(keyframes) else max_obs_frame + 1
        frame_labels = labels[labels["frame_id"] == int(start_frame)][["team_id", "player_id", "role_label"]].drop_duplicates(subset=["team_id", "player_id"], keep="last")
        if frame_labels.empty:
            continue
        interval_mask = (result["frame_id"] >= int(start_frame)) & (result["frame_id"] < int(end_frame))
        if bool(interval_mask.any()):
            joined = result.loc[interval_mask, ["team_id", "player_id"]].merge(frame_labels, on=["team_id", "player_id"], how="left")
            result.loc[interval_mask, "role_label"] = joined["role_label"].to_numpy()
    return result


def _normalized_labeled_entries(label_entries: pd.DataFrame) -> pd.DataFrame:
    if label_entries.empty:
        return label_entries.copy()
    labels = label_entries.copy()
    labels["role_label"] = labels.get("role_label", pd.Series([pd.NA] * len(labels))).astype("string").str.strip()
    labels.loc[labels["role_label"] == "", "role_label"] = pd.NA
    labels["frame_id"] = pd.to_numeric(labels["frame_id"], errors="coerce").astype("Int64")
    labels["player_id"] = pd.to_numeric(labels["player_id"], errors="coerce").astype("Int64")
    labels["team_id"] = labels["team_id"].astype(str)
    labels = labels.dropna(subset=["frame_id", "player_id"]).copy()
    labels["frame_id"] = labels["frame_id"].astype(int)
    labels["player_id"] = labels["player_id"].astype(int)
    return labels.drop_duplicates(subset=["frame_id", "team_id", "player_id"], keep="last")


def _bbox_to_list(bbox: Any) -> list[float] | None:
    if bbox is None:
        return None
    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]


def _pick_schedule_frame(
    eligible: pd.Series,
    available_frames: np.ndarray,
    target_frame: int,
    window_frames: int,
) -> int | None:
    if window_frames > 0:
        in_window = eligible[(eligible.index >= target_frame) & (eligible.index <= target_frame + window_frames)]
        if not in_window.empty:
            max_players = int(in_window.max())
            return int(in_window[in_window == max_players].index[0])
    idx = int(np.searchsorted(available_frames, target_frame, side="left"))
    candidates = []
    if idx < len(available_frames):
        candidates.append(int(available_frames[idx]))
    if idx > 0:
        candidates.append(int(available_frames[idx - 1]))
    return min(candidates, key=lambda frame_id: (abs(frame_id - target_frame), frame_id)) if candidates else None


def _empty_validation(total_entries: int, total_rows: int) -> dict[str, Any]:
    return {
        "invalid_labels": [],
        "total_entries": int(total_entries),
        "labeled_entries": 0,
        "expected_frames": 0,
        "labeled_frames": 0,
        "unmatched_labeled_entries": 0,
        "covered_rows": 0,
        "total_rows": int(total_rows),
        "coverage_ratio": 0.0,
    }
