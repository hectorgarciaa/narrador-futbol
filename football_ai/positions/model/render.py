from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_ai.core.config import Config
from football_ai.visualization import Drawer

from ..data import find_project_root


def render_role_video(
    video_path: Path,
    tracks_path: Path,
    project_root: Path | None = None,
    output_path: Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    video_path = Path(video_path)
    tracks_path = Path(tracks_path)
    if not video_path.exists():
        raise FileNotFoundError(f"No existe el vídeo de entrada: {video_path}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"No existe el tracks JSON enriquecido: {tracks_path}")
    with tracks_path.open("r", encoding="utf-8") as f:
        tracks = json.load(f)
    if output_path is None:
        output_path = tracks_path.with_name(f"{video_path.stem}_roles_annotated.mp4")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drawer = Drawer(colors=_visualization_colors_from_config(project_root))
    drawer.draw_tracks(
        tracks=tracks,
        video=str(video_path),
        output_path=str(output_path),
        show=bool(show),
        window_name="Roles Posicionales",
    )
    return {
        "video_path": output_path,
        "tracks_path": tracks_path,
        "source_video_path": video_path,
    }


def _visualization_colors_from_config(project_root: Path) -> dict[str, tuple[int, int, int]]:
    colors_raw = Config.from_yaml(project_root / "config.yaml").get("visualization", "colors", default={}) or {}
    colors = {
        str(class_name): tuple(int(v) for v in color_values[:3])
        for class_name, color_values in colors_raw.items()
        if isinstance(color_values, (list, tuple)) and len(color_values) >= 3
    }
    return colors or {
        "player": (0, 255, 0),
        "goalkeeper": (0, 255, 255),
        "referee": (255, 0, 0),
        "ball": (0, 0, 255),
    }
