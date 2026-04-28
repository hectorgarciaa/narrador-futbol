from __future__ import annotations

import re
import shutil
from pathlib import Path


def sanitize_video_stem(raw_stem):
    stem = str(raw_stem).strip()
    if not stem:
        return "video"
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "video"


def build_role_artifacts_output_dir(config, video_path):
    role_artifacts_root = config.get_path("paths", "output", "role_artifacts", create_if_missing=True)
    role_artifacts_root.mkdir(parents=True, exist_ok=True)
    role_artifacts_dir = role_artifacts_root / f"{sanitize_video_stem(Path(video_path).stem)}_role_artifacts"
    role_artifacts_dir.mkdir(parents=True, exist_ok=True)
    return role_artifacts_dir


def build_role_predictions_output_paths(config, video_path, use_artifacts_dir=True):
    sanitized_stem = sanitize_video_stem(Path(video_path).stem)
    if use_artifacts_dir:
        output_dir = build_role_artifacts_output_dir(config, video_path)
    else:
        output_dir = config.get_path("paths", "output", "tracks_json", create_if_missing=True) / "tracker"
        output_dir.mkdir(parents=True, exist_ok=True)
    return (
        str(output_dir / f"{sanitized_stem}_frame_role_predictions.csv"),
        str(output_dir / f"{sanitized_stem}_player_role_summary.csv"),
        str(output_dir / f"{sanitized_stem}_greedy_role_diagnostics.csv"),
    )


def save_dataframe_csv(df, output_path, logger, description):
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("%s saved to: %s", description, output_path)
    except Exception as exc:  # pragma: no cover
        logger.error("Error saving %s to %s: %s", description, output_path, exc)


def copy_output_artifact(source_path, target_path, logger, description):
    try:
        source_path = Path(source_path)
        target_path = Path(target_path)
        if source_path.resolve() == target_path.resolve():
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        logger.info("%s copied to: %s", description, target_path)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Error copying %s from %s to %s: %s",
            description,
            source_path,
            target_path,
            exc,
        )
