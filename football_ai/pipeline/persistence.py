import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from football_ai.core import convert_to_serializable


def save_result(tracks, output_path, logger):
    """Saves tracks in JSON format with error handling."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                convert_to_serializable(tracks),
                f,
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
            )
        logger.info(f"Tracks saved to: {output_path}")
    except IOError as e:
        logger.error(f"Error saving tracks to {output_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving tracks: {e}")


def save_summary(summary, output_path, logger):
    """Save evaluation summary JSON for a single video."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                convert_to_serializable(summary),
                f,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        logger.info(f"Tracking summary saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving tracking summary to {output_path}: {e}")


def save_debug_frames(debug_frames, output_path, logger):
    """Save per-frame debug metadata (four-panel discarded detections, reasons, etc.)."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                convert_to_serializable(debug_frames),
                f,
                indent=2,
                ensure_ascii=False,
                sort_keys=False,
            )
        logger.info(f"Debug frames saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving debug frames to {output_path}: {e}")


def _read_csv_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def upsert_tracking_metrics_dataset(
    dataset_path,
    video_path,
    video_source,
    tracks_json_path,
    summary_json_path,
    summary,
    logger,
):
    """
    Upsert a row in tracking_metrics.csv keyed by video_source.
    """
    try:
        dataset_path = Path(dataset_path)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        video_path_obj = Path(video_path)
        row = {
            "video_source": str(video_source),
            "video_filename": video_path_obj.name,
            "video_stem": video_path_obj.stem,
            "video_path": str(video_path),
            "tracks_json_path": str(tracks_json_path),
            "summary_json_path": str(summary_json_path),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        for key, value in summary.items():
            row[str(key)] = value

        rows = []
        fieldnames = list(row.keys())
        if dataset_path.exists():
            existing_rows, existing_fieldnames = _read_csv_rows(dataset_path)
            rows = [
                existing_row
                for existing_row in existing_rows
                if existing_row.get("video_source") != row["video_source"]
            ]
            for key in existing_fieldnames:
                if key not in fieldnames:
                    fieldnames.append(key)
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        rows.append(row)
        rows.sort(key=lambda item: item.get("video_source", ""))

        with open(dataset_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in rows:
                writer.writerow({name: item.get(name, "") for name in fieldnames})

        logger.info(f"Tracking metrics dataset updated: {dataset_path}")
    except Exception as e:
        logger.error(f"Error updating tracking metrics dataset {dataset_path}: {e}")
