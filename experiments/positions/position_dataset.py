from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd


ROLE_LABELS_V1 = (
    "POR",
    "LI",
    "DFC_IZQ",
    "DFC_DER",
    "LD",
    "MC",
    "MI",
    "MD",
    "EI",
    "ED",
    "DC",
)


@dataclass(frozen=True)
class FeatureSpec:
    objective_feature_names: tuple[str, ...]
    teammate_feature_names: tuple[str, ...]


def find_project_root(start: Path | None = None) -> Path:
    search_start = (start or Path.cwd()).resolve()
    for candidate in (search_start, *search_start.parents):
        if (candidate / "config.yaml").exists():
            return candidate
    raise FileNotFoundError(
        "No se encontró config.yaml desde el directorio actual hacia arriba."
    )


def list_position_videos(project_root: Path) -> list[Path]:
    folder = project_root / "data" / "partidosPosiciones"
    if not folder.exists():
        return []
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
        ]
    )


def sanitize_video_stem(raw_stem: str) -> str:
    stem = str(raw_stem).strip()
    if not stem:
        return "video"
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "video"


def resolve_tracks_path_for_video(project_root: Path, video_path: Path) -> Path:
    tracks_dir = project_root / "output" / "tracks_json" / "tracker"
    named_path = tracks_dir / f"{sanitize_video_stem(video_path.stem)}_tracks.json"
    legacy_path = tracks_dir / "tracks.json"
    if named_path.exists():
        return named_path
    if legacy_path.exists():
        return legacy_path
    return named_path


def load_tracks_json(tracks_path: Path) -> dict[str, Any]:
    with tracks_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_field_position(track_data: Mapping[str, Any]) -> tuple[float, float] | None:
    field_position = track_data.get("field_position_m")
    if field_position is None:
        return None
    arr = np.asarray(field_position, dtype=np.float32).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    return float(arr[0]), float(arr[1])


def _safe_bbox(track_data: Mapping[str, Any]) -> list[float] | None:
    bbox = track_data.get("bbox")
    if bbox is None:
        return None
    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]


def build_observations_from_tracks(
    tracks: Mapping[str, Any],
    match_id: str,
    field_length_m: float = 106.0,
    field_width_m: float = 68.0,
    tracked_classes: Sequence[str] = ("player", "goalkeeper"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for class_name in tracked_classes:
        frames = tracks.get(class_name, [])
        for frame_id, frame_tracks in enumerate(frames):
            if not isinstance(frame_tracks, Mapping):
                continue
            for player_id_raw, track_data in frame_tracks.items():
                field_pos = _safe_field_position(track_data)
                if field_pos is None:
                    continue
                x_m, y_m = field_pos
                x = float(np.clip(x_m / max(field_length_m, 1e-6), 0.0, 1.0))
                y = float(np.clip(y_m / max(field_width_m, 1e-6), 0.0, 1.0))
                bbox = _safe_bbox(track_data)

                try:
                    player_id = int(player_id_raw)
                except (TypeError, ValueError):
                    continue

                rows.append(
                    {
                        "match_id": str(match_id),
                        "frame_id": int(frame_id),
                        "team_id": track_data.get("team"),
                        "player_id": player_id,
                        "class_name": class_name,
                        "x_m": x_m,
                        "y_m": y_m,
                        "x": x,
                        "y": y,
                        "visible": 1,
                        "confidence_tracking": float(track_data.get("confidence", np.nan)),
                        "bbox": bbox,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "match_id",
                "frame_id",
                "team_id",
                "player_id",
                "class_name",
                "x_m",
                "y_m",
                "x",
                "y",
                "visible",
                "confidence_tracking",
                "bbox",
            ]
        )

    df = pd.DataFrame(rows)
    df = df[df["team_id"].notna()].copy()
    df["team_id"] = df["team_id"].astype(str)
    df = df.sort_values(["frame_id", "team_id", "player_id"]).reset_index(drop=True)
    return df


def add_velocity_features(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        result = observations.copy()
        result["vx"] = np.nan
        result["vy"] = np.nan
        return result

    df = observations.copy()
    df = df.sort_values(["team_id", "player_id", "frame_id"]).reset_index(drop=True)
    grouped = df.groupby(["team_id", "player_id"], sort=False)

    dx = grouped["x"].diff()
    dy = grouped["y"].diff()
    dt = grouped["frame_id"].diff()
    valid_dt = dt.where(dt > 0, np.nan)

    df["vx"] = (dx / valid_dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["vy"] = (dy / valid_dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def choose_label_frame(observations: pd.DataFrame, min_players: int = 18) -> int:
    if observations.empty:
        raise ValueError("No hay observaciones para seleccionar frame de etiquetado.")
    per_frame = observations.groupby("frame_id")["player_id"].nunique().sort_index()
    candidates = per_frame[per_frame >= int(min_players)]
    if len(candidates) > 0:
        return int(candidates.index[0])
    return int(per_frame.index[0])


def get_video_fps(video_path: Path, default_fps: float = 25.0) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 1e-6:
        return float(default_fps)
    return float(fps)


def build_periodic_label_schedule(
    observations: pd.DataFrame,
    fps: float,
    every_seconds: float = 6.0,
    start_seconds: float = 0.0,
    min_players: int = 18,
) -> pd.DataFrame:
    columns = [
        "target_second",
        "target_frame",
        "frame_second",
        "frame_id",
        "num_players",
    ]
    if observations.empty:
        return pd.DataFrame(columns=columns)
    if not np.isfinite(fps) or fps <= 1e-6:
        raise ValueError(f"FPS inválido: {fps}")
    if every_seconds <= 0:
        raise ValueError(
            f"every_seconds debe ser > 0 (recibido {every_seconds})."
        )

    per_frame_players = (
        observations.groupby("frame_id")["player_id"]
        .nunique()
        .sort_index()
    )
    eligible = per_frame_players[per_frame_players >= int(min_players)]
    if eligible.empty:
        eligible = per_frame_players
    if eligible.empty:
        return pd.DataFrame(columns=columns)

    available_frames = eligible.index.to_numpy(dtype=np.int64)
    max_frame = int(per_frame_players.index.max())
    start_frame = max(0, int(round(float(start_seconds) * float(fps))))
    step_frames = max(1, int(round(float(every_seconds) * float(fps))))

    target_frames = list(range(start_frame, max_frame + 1, step_frames))
    rows: list[dict[str, Any]] = []
    for target_frame in target_frames:
        idx = int(np.searchsorted(available_frames, target_frame, side="left"))
        candidates: list[int] = []
        if idx < len(available_frames):
            candidates.append(int(available_frames[idx]))
        if idx > 0:
            candidates.append(int(available_frames[idx - 1]))
        if not candidates:
            continue

        selected_frame = min(
            candidates,
            key=lambda frame_id: (abs(frame_id - target_frame), frame_id),
        )
        rows.append(
            {
                "target_second": float(target_frame / fps),
                "target_frame": int(target_frame),
                "frame_second": float(selected_frame / fps),
                "frame_id": int(selected_frame),
                "num_players": int(per_frame_players.loc[selected_frame]),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    schedule_df = pd.DataFrame(rows)
    schedule_df = (
        schedule_df.drop_duplicates(subset=["frame_id"], keep="first")
        .sort_values(["target_frame", "frame_id"])
        .reset_index(drop=True)
    )
    return schedule_df


def build_periodic_role_label_template(
    observations: pd.DataFrame,
    match_id: str,
    video_path: Path,
    fps: float,
    every_seconds: float = 6.0,
    start_seconds: float = 0.0,
    min_players: int = 18,
) -> dict[str, Any]:
    schedule_df = build_periodic_label_schedule(
        observations=observations,
        fps=fps,
        every_seconds=every_seconds,
        start_seconds=start_seconds,
        min_players=min_players,
    )

    labels: list[dict[str, Any]] = []
    for _, schedule_row in schedule_df.iterrows():
        frame_id = int(schedule_row["frame_id"])
        frame_observations = observations[observations["frame_id"] == frame_id].copy()
        frame_observations = frame_observations.sort_values(
            ["team_id", "player_id", "class_name"]
        )

        for _, row in frame_observations.iterrows():
            bbox = row.get("bbox")
            bbox_list: list[float] | None = None
            if bbox is not None:
                arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
                if arr.size >= 4 and np.all(np.isfinite(arr[:4])):
                    bbox_list = [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]

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
                    "bbox": bbox_list,
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


def save_role_label_template_json(
    template: Mapping[str, Any],
    output_path: Path,
    overwrite: bool = False,
) -> Path:
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

    entries = payload
    if isinstance(payload, Mapping):
        entries = payload.get("labels", [])

    if not isinstance(entries, list) or len(entries) == 0:
        return pd.DataFrame(
            columns=["frame_id", "team_id", "player_id", "role_label"]
        )

    labels_df = pd.DataFrame(entries)
    required_columns = {"frame_id", "team_id", "player_id"}
    missing_columns = required_columns.difference(labels_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Faltan columnas en labels JSON ({missing}) en {label_json_path}."
        )

    labels_df["frame_id"] = pd.to_numeric(
        labels_df["frame_id"], errors="coerce"
    ).astype("Int64")
    labels_df["player_id"] = pd.to_numeric(
        labels_df["player_id"], errors="coerce"
    ).astype("Int64")
    labels_df["team_id"] = labels_df["team_id"].astype(str)

    if "role_label" not in labels_df.columns:
        labels_df["role_label"] = pd.NA
    labels_df["role_label"] = labels_df["role_label"].astype("string").str.strip()
    labels_df.loc[labels_df["role_label"] == "", "role_label"] = pd.NA

    labels_df = labels_df.dropna(subset=["frame_id", "player_id"]).copy()
    labels_df["frame_id"] = labels_df["frame_id"].astype(int)
    labels_df["player_id"] = labels_df["player_id"].astype(int)

    labels_df = labels_df.sort_values(
        ["frame_id", "team_id", "player_id"]
    ).drop_duplicates(
        subset=["frame_id", "team_id", "player_id"],
        keep="last",
    )
    labels_df = labels_df.reset_index(drop=True)
    return labels_df


def validate_periodic_role_labels(
    observations: pd.DataFrame,
    label_entries: pd.DataFrame,
    allowed_labels: Sequence[str] = ROLE_LABELS_V1,
) -> dict[str, Any]:
    labels_set = set(allowed_labels)
    if observations.empty:
        return {
            "invalid_labels": [],
            "total_entries": int(len(label_entries)),
            "labeled_entries": 0,
            "expected_frames": 0,
            "labeled_frames": 0,
            "unmatched_labeled_entries": 0,
            "covered_rows": 0,
            "total_rows": 0,
            "coverage_ratio": 0.0,
        }

    if label_entries.empty:
        return {
            "invalid_labels": [],
            "total_entries": 0,
            "labeled_entries": 0,
            "expected_frames": 0,
            "labeled_frames": 0,
            "unmatched_labeled_entries": 0,
            "covered_rows": 0,
            "total_rows": int(len(observations)),
            "coverage_ratio": 0.0,
        }

    normalized = label_entries.copy()
    if "role_label" not in normalized.columns:
        normalized["role_label"] = pd.NA
    normalized["role_label"] = normalized["role_label"].astype("string").str.strip()
    normalized.loc[normalized["role_label"] == "", "role_label"] = pd.NA

    labeled = normalized[normalized["role_label"].notna()].copy()
    invalid = sorted(
        labeled.loc[~labeled["role_label"].isin(labels_set), "role_label"]
        .astype(str)
        .unique()
        .tolist()
    )

    merge_keys = ["frame_id", "team_id", "player_id"]
    obs_keys = observations[merge_keys].drop_duplicates()
    unmatched_entries = 0
    if not labeled.empty:
        check_df = labeled[merge_keys].drop_duplicates().merge(
            obs_keys,
            on=merge_keys,
            how="left",
            indicator=True,
        )
        unmatched_entries = int((check_df["_merge"] == "left_only").sum())

    merged = observations.merge(
        labeled[merge_keys + ["role_label"]],
        on=merge_keys,
        how="left",
    )
    covered_rows = int(merged["role_label"].notna().sum())

    return {
        "invalid_labels": invalid,
        "total_entries": int(len(normalized)),
        "labeled_entries": int(len(labeled)),
        "expected_frames": int(normalized["frame_id"].nunique()),
        "labeled_frames": int(labeled["frame_id"].nunique()) if not labeled.empty else 0,
        "unmatched_labeled_entries": unmatched_entries,
        "covered_rows": covered_rows,
        "total_rows": int(len(observations)),
        "coverage_ratio": float(covered_rows / len(observations)) if len(observations) else 0.0,
    }


def apply_periodic_role_labels(
    observations: pd.DataFrame,
    label_entries: pd.DataFrame,
) -> pd.DataFrame:
    result = observations.copy()
    if result.empty:
        result["role_label"] = pd.NA
        return result

    if label_entries.empty:
        result["role_label"] = pd.NA
        return result

    labels = label_entries.copy()
    if "role_label" not in labels.columns:
        labels["role_label"] = pd.NA
    labels["role_label"] = labels["role_label"].astype("string").str.strip()
    labels.loc[labels["role_label"] == "", "role_label"] = pd.NA
    labels = labels[labels["role_label"].notna()].copy()
    if labels.empty:
        result["role_label"] = pd.NA
        return result

    merge_keys = ["frame_id", "team_id", "player_id"]
    labels = labels[merge_keys + ["role_label"]].drop_duplicates(
        subset=merge_keys,
        keep="last",
    )
    return result.merge(labels, on=merge_keys, how="left")


def append_run_to_common_dataset(
    project_root: Path,
    labeled_obs_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    teammates_tensor: np.ndarray,
    teammate_mask: np.ndarray,
    feature_spec: FeatureSpec,
    source_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    common_dir = project_root / "output" / "datasets" / "positions" / "common"
    common_dir.mkdir(parents=True, exist_ok=True)

    base_table_path = common_dir / "base_table.csv"
    samples_csv_path = common_dir / "samples_metadata_and_obj_features.csv"
    samples_npz_path = common_dir / "samples_teammates.npz"
    meta_json_path = common_dir / "dataset_meta.json"
    sources_log_path = common_dir / "sources.jsonl"

    if base_table_path.exists():
        existing_base = pd.read_csv(base_table_path)
        combined_base = pd.concat(
            [existing_base, labeled_obs_df],
            ignore_index=True,
            sort=False,
        )
    else:
        combined_base = labeled_obs_df.copy()
    combined_base.to_csv(base_table_path, index=False)

    if samples_csv_path.exists():
        existing_samples = pd.read_csv(samples_csv_path)
        combined_samples = pd.concat(
            [existing_samples, samples_df],
            ignore_index=True,
            sort=False,
        )
    else:
        combined_samples = samples_df.copy()
    combined_samples.to_csv(samples_csv_path, index=False)

    new_tensor = np.asarray(teammates_tensor, dtype=np.float32)
    new_mask = np.asarray(teammate_mask, dtype=np.uint8)
    if new_tensor.ndim != 3:
        raise ValueError(
            f"teammates_tensor debe ser 3D; shape actual: {new_tensor.shape}"
        )
    if new_mask.ndim != 2:
        raise ValueError(
            f"teammate_mask debe ser 2D; shape actual: {new_mask.shape}"
        )

    if samples_npz_path.exists():
        with np.load(samples_npz_path) as old_npz:
            old_tensor = np.asarray(old_npz["teammates_tensor"], dtype=np.float32)
            old_mask = np.asarray(old_npz["teammate_mask"], dtype=np.uint8)

        if old_tensor.ndim != 3 or old_mask.ndim != 2:
            raise ValueError(
                "El dataset común existente tiene una forma inválida en samples_teammates.npz."
            )

        if old_tensor.shape[0] > 0 and new_tensor.shape[0] > 0:
            if old_tensor.shape[1:] != new_tensor.shape[1:]:
                raise ValueError(
                    "Incompatibilidad en teammates_tensor: "
                    f"existente {old_tensor.shape[1:]} vs nuevo {new_tensor.shape[1:]}"
                )
            if old_mask.shape[1:] != new_mask.shape[1:]:
                raise ValueError(
                    "Incompatibilidad en teammate_mask: "
                    f"existente {old_mask.shape[1:]} vs nuevo {new_mask.shape[1:]}"
                )

        if old_tensor.shape[0] == 0:
            combined_tensor = new_tensor
            combined_mask = new_mask
        elif new_tensor.shape[0] == 0:
            combined_tensor = old_tensor
            combined_mask = old_mask
        else:
            combined_tensor = np.concatenate([old_tensor, new_tensor], axis=0)
            combined_mask = np.concatenate([old_mask, new_mask], axis=0)
    else:
        combined_tensor = new_tensor
        combined_mask = new_mask

    np.savez_compressed(
        samples_npz_path,
        teammates_tensor=combined_tensor.astype(np.float32),
        teammate_mask=combined_mask.astype(np.uint8),
    )

    source_record = dict(source_info or {})
    source_record["appended_at"] = datetime.now().isoformat()
    with sources_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(source_record, ensure_ascii=False, default=str) + "\n")

    previous_meta = {}
    if meta_json_path.exists():
        try:
            with meta_json_path.open("r", encoding="utf-8") as f:
                previous_meta = json.load(f)
        except Exception:
            previous_meta = {}

    meta = {
        "updated_at": datetime.now().isoformat(),
        "num_runs": int(previous_meta.get("num_runs", 0)) + 1,
        "num_base_rows": int(len(combined_base)),
        "num_samples": int(len(combined_samples)),
        "num_matches": int(combined_samples["match_id"].nunique())
        if "match_id" in combined_samples.columns and len(combined_samples) > 0
        else 0,
        "max_teammates": int(combined_tensor.shape[1]) if combined_tensor.ndim == 3 else 0,
        "teammate_feature_dim": int(combined_tensor.shape[2]) if combined_tensor.ndim == 3 else 0,
        "objective_feature_names": list(feature_spec.objective_feature_names),
        "teammate_feature_names": list(feature_spec.teammate_feature_names),
        "sources_log_path": str(sources_log_path),
    }
    with meta_json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "common_dir": common_dir,
        "base_table_path": base_table_path,
        "samples_csv_path": samples_csv_path,
        "samples_npz_path": samples_npz_path,
        "meta_json_path": meta_json_path,
        "sources_log_path": sources_log_path,
        "num_base_rows": int(len(combined_base)),
        "num_samples": int(len(combined_samples)),
    }


def get_video_frame(video_path: Path, frame_id: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"No se pudo leer frame {frame_id} en {video_path}.")
        return frame
    finally:
        cap.release()


def render_frame_with_player_ids(
    video_path: Path,
    frame_observations: pd.DataFrame,
    frame_id: int,
) -> np.ndarray:
    frame = get_video_frame(video_path, frame_id)
    if frame_observations.empty:
        return frame

    teams = sorted(frame_observations["team_id"].dropna().unique().tolist())
    palette = [
        (56, 56, 255),
        (46, 204, 113),
        (255, 164, 32),
        (255, 255, 0),
    ]
    team_colors = {team: palette[idx % len(palette)] for idx, team in enumerate(teams)}

    for _, row in frame_observations.iterrows():
        bbox = row.get("bbox")
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = team_colors.get(row["team_id"], (255, 255, 255))
        label = f'{row["player_id"]}:{row["team_id"]}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    return frame


def role_map_to_frame(role_map: Mapping[str, Mapping[int, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for team_id, players in role_map.items():
        for player_id, role_label in players.items():
            rows.append(
                {
                    "team_id": str(team_id),
                    "player_id": int(player_id),
                    "role_label": str(role_label),
                }
            )
    return pd.DataFrame(rows)


def validate_role_map(
    observations: pd.DataFrame,
    role_map: Mapping[str, Mapping[int, str]],
    allowed_labels: Sequence[str] = ROLE_LABELS_V1,
) -> dict[str, Any]:
    labels_set = set(allowed_labels)
    role_df = role_map_to_frame(role_map)
    if role_df.empty:
        return {
            "invalid_labels": [],
            "missing_team_players": {},
            "covered_rows": 0,
            "total_rows": int(len(observations)),
            "coverage_ratio": 0.0,
        }

    invalid = sorted(
        role_df.loc[~role_df["role_label"].isin(labels_set), "role_label"]
        .astype(str)
        .unique()
        .tolist()
    )

    missing_team_players: dict[str, list[int]] = {}
    grouped_obs = observations.groupby("team_id")
    for team_id, team_obs in grouped_obs:
        expected_ids = sorted(team_obs["player_id"].unique().tolist())
        provided = set()
        if team_id in role_map:
            provided = {int(pid) for pid in role_map[team_id].keys()}
        missing = [int(pid) for pid in expected_ids if int(pid) not in provided]
        missing_team_players[str(team_id)] = missing

    merged = observations.merge(role_df, on=["team_id", "player_id"], how="left")
    covered_rows = int(merged["role_label"].notna().sum())
    return {
        "invalid_labels": invalid,
        "missing_team_players": missing_team_players,
        "covered_rows": covered_rows,
        "total_rows": int(len(merged)),
        "coverage_ratio": float(covered_rows / len(merged)) if len(merged) else 0.0,
    }


def apply_role_map(
    observations: pd.DataFrame,
    role_map: Mapping[str, Mapping[int, str]],
) -> pd.DataFrame:
    role_df = role_map_to_frame(role_map)
    if role_df.empty:
        result = observations.copy()
        result["role_label"] = pd.NA
        return result
    return observations.merge(role_df, on=["team_id", "player_id"], how="left")


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

    if (
        len(teams) == 2
        and not goalkeepers.empty
        and set(teams).issubset(set(goalkeepers["team_id"].unique().tolist()))
    ):
        ref = goalkeepers.groupby("team_id")["x"].median()
        method = "goalkeeper_median_x"
    else:
        ref = valid.groupby("team_id")["x"].median()
        method = "team_median_x"

    ref = ref.sort_values(ascending=True)
    direction_by_team: dict[str, int] = {}
    if len(ref.index) == 2:
        left_team = str(ref.index[0])
        right_team = str(ref.index[1])
        direction_by_team[left_team] = +1
        direction_by_team[right_team] = -1
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


def _oriented_xy(
    x: np.ndarray,
    y: np.ndarray,
    attack_direction: int,
) -> tuple[np.ndarray, np.ndarray]:
    if int(attack_direction) >= 0:
        return x, y
    return 1.0 - x, y


def build_role_samples(
    observations_with_roles: pd.DataFrame,
    attack_direction_by_team: Mapping[str, int],
    max_teammates: int = 10,
    drop_unlabeled: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, FeatureSpec]:
    df = observations_with_roles.copy()
    if df.empty:
        return (
            df,
            np.zeros((0, int(max_teammates), 0), dtype=np.float32),
            np.zeros((0, int(max_teammates)), dtype=bool),
            FeatureSpec(tuple(), tuple()),
        )

    if drop_unlabeled:
        df = df[df["role_label"].notna()].copy()
    if df.empty:
        return (
            df,
            np.zeros((0, int(max_teammates), 0), dtype=np.float32),
            np.zeros((0, int(max_teammates)), dtype=bool),
            FeatureSpec(tuple(), tuple()),
        )

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

    group_keys = ["match_id", "frame_id", "team_id"]
    grouped = observations_with_roles.groupby(group_keys, sort=True)
    for (match_id, frame_id, team_id), team_frame in grouped:
        team_frame = team_frame.copy()
        if "role_label" not in team_frame.columns:
            continue

        attack_direction = int(attack_direction_by_team.get(str(team_id), +1))
        team_frame["x_ori"], team_frame["y_ori"] = _oriented_xy(
            team_frame["x"].to_numpy(dtype=np.float32),
            team_frame["y"].to_numpy(dtype=np.float32),
            attack_direction,
        )
        team_frame["vx_ori"] = (
            team_frame["vx"].to_numpy(dtype=np.float32)
            if "vx" in team_frame.columns
            else np.zeros((len(team_frame),), dtype=np.float32)
        )
        if attack_direction < 0:
            team_frame["vx_ori"] = -team_frame["vx_ori"]
        team_frame["vy_ori"] = (
            team_frame["vy"].to_numpy(dtype=np.float32)
            if "vy" in team_frame.columns
            else np.zeros((len(team_frame),), dtype=np.float32)
        )
        team_frame["rank_x_team"] = (
            team_frame["x_ori"].rank(method="dense", ascending=True).astype(np.float32)
        )
        team_frame["rank_y_team"] = (
            team_frame["y_ori"].rank(method="dense", ascending=True).astype(np.float32)
        )

        if drop_unlabeled:
            objectives = team_frame[team_frame["role_label"].notna()].copy()
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

            if teammates.empty:
                centroid_x = x_obj
                centroid_y = y_obj
            else:
                centroid_x = float(teammates["x_ori"].mean())
                centroid_y = float(teammates["y_ori"].mean())

            row = {
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
            rows.append(row)

            teammate_features = np.zeros(
                (int(max_teammates), len(teammate_feature_names)),
                dtype=np.float32,
            )
            teammate_exists = np.zeros((int(max_teammates),), dtype=bool)

            if not teammates.empty:
                teammates["dx_to_obj"] = teammates["x_ori"] - x_obj
                teammates["dy_to_obj"] = teammates["y_ori"] - y_obj
                teammates["dist_to_obj"] = np.sqrt(
                    (teammates["dx_to_obj"] ** 2) + (teammates["dy_to_obj"] ** 2)
                )
                teammates = teammates.sort_values(
                    ["dist_to_obj", "player_id"],
                    ascending=[True, True],
                )

                usable = teammates.head(int(max_teammates)).reset_index(drop=True)
                for idx_tm, tm in usable.iterrows():
                    values = np.array(
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
                    teammate_features[idx_tm, :] = values
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

    samples_df = pd.DataFrame(rows)
    return (
        samples_df,
        np.asarray(teammates_tensor, dtype=np.float32),
        np.asarray(teammates_mask, dtype=bool),
        FeatureSpec(objective_feature_names, teammate_feature_names),
    )
