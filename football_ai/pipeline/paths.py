import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_video_path(config, video_shortcut):
    """
    Resolve video path either from a config shortcut (paths.data.<key>)
    or from a direct filesystem path.
    """
    data_paths = config.paths.get("data", {})
    if video_shortcut:
        if video_shortcut in data_paths:
            return str(config.get_path("paths", "data", video_shortcut)), video_shortcut

        candidate = Path(video_shortcut).expanduser()
        if not candidate.is_absolute():
            candidate = (config.project_root / candidate).resolve()
        if candidate.exists():
            return str(candidate), str(candidate)

        available_keys = ", ".join(sorted(data_paths.keys()))
        raise FileNotFoundError(
            f"No se encontró el shortcut/ruta de video '{video_shortcut}'. "
            f"Shortcuts disponibles en paths.data: {available_keys}"
        )

    default_key = (
        "video_prueba_corto"
        if data_paths.get("video_prueba_corto") is not None
        else "video_prueba"
    )
    return str(config.get_path("paths", "data", default_key)), default_key


def build_output_video_path(config, video_path):
    """Build output path using the input video filename."""
    output_dir = config.get_path(
        "paths", "output", "prueba_tracker", create_if_missing=True
    )
    input_stem = Path(video_path).stem or "video"
    output_name = f"{input_stem}_tracking.mp4"
    return str(output_dir / output_name)


def sanitize_video_stem(raw_stem):
    """Sanitize video stem to align with experiments/positions naming convention."""
    stem = str(raw_stem).strip()
    if not stem:
        return "video"
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "video"


def build_tracks_output_paths(config, video_path):
    """
    Build JSON output paths for tracking results:
    - named path: required by experiments/positions notebook
    - legacy path: backwards compatibility
    """
    tracks_dir = config.get_path(
        "paths", "output", "tracks_json", create_if_missing=True
    ) / "tracker"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    video_stem = Path(video_path).stem
    sanitized_stem = sanitize_video_stem(video_stem)
    named_path = tracks_dir / f"{sanitized_stem}_tracks.json"
    legacy_path = tracks_dir / "tracks.json"
    return str(named_path), str(legacy_path)


def build_tracking_metrics_output_paths(config, video_path):
    """
    Build output paths for per-video summary and aggregated metrics dataset.
    """
    tracks_dir = config.get_path(
        "paths", "output", "tracks_json", create_if_missing=True
    ) / "tracker"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = (
        PROJECT_ROOT / "data" / "posiciones_etiquetadas" / "common" / "tracking_metrics.csv"
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    sanitized_stem = sanitize_video_stem(Path(video_path).stem)
    summary_path = tracks_dir / f"{sanitized_stem}_summary.json"
    return str(summary_path), str(dataset_path)


def build_debug_frames_output_path(config, video_path):
    """
    Build output path for per-frame debug metadata used by the four-panel drawer.
    """
    tracks_dir = config.get_path(
        "paths", "output", "tracks_json", create_if_missing=True
    ) / "tracker"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    sanitized_stem = sanitize_video_stem(Path(video_path).stem)
    debug_path = tracks_dir / f"{sanitized_stem}_debug_frames.json"
    return str(debug_path)
