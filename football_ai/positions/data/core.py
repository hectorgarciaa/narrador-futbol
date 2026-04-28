from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROLE_LABELS_V1 = (
    "POR",
    "CI",
    "LI",
    "DFC_IZQ",
    "DFC_CENT",
    "DFC_DER",
    "LD",
    "CD",
    "MC",
    "MI",
    "MD",
    "EI",
    "ED",
    "DC",
)

POSITIONS_DATASET_DIR = Path("data/posiciones_etiquetadas")
POSITIONS_COMMON_DIR = POSITIONS_DATASET_DIR / "common"
POSITIONS_LABELS_DIR = POSITIONS_DATASET_DIR / "labels"


@dataclass(frozen=True)
class FeatureSpec:
    objective_feature_names: tuple[str, ...]
    teammate_feature_names: tuple[str, ...]


def find_project_root(start: Path | None = None) -> Path:
    search_start = (start or Path.cwd()).resolve()
    for candidate in (search_start, *search_start.parents):
        if (candidate / "config.yaml").exists():
            return candidate
    raise FileNotFoundError("No se encontró config.yaml desde el directorio actual hacia arriba.")


def list_position_videos(project_root: Path) -> list[Path]:
    folder = project_root / "data" / "partidosPosiciones"
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
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
    return tracks_dir / f"{sanitize_video_stem(video_path.stem)}_tracks.json"


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
        for frame_id, frame_tracks in enumerate(tracks.get(class_name, [])):
            if not isinstance(frame_tracks, Mapping):
                continue
            for player_id_raw, track_data in frame_tracks.items():
                field_pos = _safe_field_position(track_data)
                if field_pos is None:
                    continue
                try:
                    player_id = int(player_id_raw)
                except (TypeError, ValueError):
                    continue
                x_m, y_m = field_pos
                rows.append(
                    {
                        "match_id": str(match_id),
                        "frame_id": int(frame_id),
                        "team_id": track_data.get("team"),
                        "player_id": player_id,
                        "class_name": class_name,
                        "x_m": float(x_m),
                        "y_m": float(y_m),
                        "x": float(np.clip(x_m / max(field_length_m, 1e-6), 0.0, 1.0)),
                        "y": float(np.clip(y_m / max(field_width_m, 1e-6), 0.0, 1.0)),
                        "visible": 1,
                        "confidence_tracking": float(track_data.get("confidence", np.nan)),
                        "bbox": _safe_bbox(track_data),
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
    return df.sort_values(["frame_id", "team_id", "player_id"]).reset_index(drop=True)


def add_velocity_features(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        result = observations.copy()
        result["vx"] = np.nan
        result["vy"] = np.nan
        return result

    df = observations.copy()
    df = df.sort_values(["team_id", "player_id", "frame_id"]).reset_index(drop=True)
    grouped = df.groupby(["team_id", "player_id"], sort=False)
    dt = grouped["frame_id"].diff().where(lambda series: series > 0, np.nan)
    df["vx"] = (grouped["x"].diff() / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["vy"] = (grouped["y"].diff() / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def get_video_fps(video_path: Path, default_fps: float = 25.0) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 1e-6:
        return float(default_fps)
    return float(fps)


def get_video_frame(video_path: Path, frame_id: int) -> np.ndarray:
    import cv2

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
    show_team_id: bool = False,
    show_player_id: bool = True,
) -> np.ndarray:
    import cv2

    frame = get_video_frame(video_path, frame_id)
    if frame_observations.empty:
        return frame

    teams = sorted(frame_observations["team_id"].dropna().unique().tolist())
    palette = [(56, 56, 255), (46, 204, 113), (255, 164, 32), (255, 255, 0)]
    team_colors = {team: palette[idx % len(palette)] for idx, team in enumerate(teams)}

    for _, row in frame_observations.iterrows():
        bbox = row.get("bbox")
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = team_colors.get(row["team_id"], (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if show_player_id:
            label = f'{row["player_id"]}:{row["team_id"]}' if show_team_id else str(int(row["player_id"]))
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
