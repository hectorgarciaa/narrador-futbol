import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

# Permite ejecutar `python scripts/track.py` sin instalar el paquete en editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.tracking import Tracker
from football_ai.evaluation import Evaluator
from football_ai.visualization import Drawer
from football_ai.core import get_config, get_logger, Logger, convert_to_serializable


def parse_bool_env(name):
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_int_env(name):
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def sanitize_filename_stem(raw_stem):
    """Converts a video stem into a filesystem-friendly stem."""
    stem = str(raw_stem).strip()
    if not stem:
        return "video"
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "video"


def build_tracks_output_path(video_path, tracks_output_root):
    video_stem = sanitize_filename_stem(Path(video_path).stem)
    return Path(tracks_output_root) / "tracker" / f"{video_stem}_tracks.json"


def save_result(tracks, output_path, logger):
    """Saves tracks in JSON format with error handling."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(tracks), f, indent=4, ensure_ascii=False, sort_keys=True)
        logger.info(f"Tracks saved to: {output_path}")
    except IOError as e:
        logger.error(f"Error saving tracks to {output_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving tracks: {e}")

def save_timing_reports(
    timing_output_dir,
    run_id,
    script_phase_times,
    tracker_timing_summary,
    tracker_timing_frames,
    logger,
):
    """Saves timing data (global summary + per-frame tracker timings)."""
    timing_output_dir = Path(timing_output_dir)
    timing_output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_phase_times_s": script_phase_times,
        "tracker_timing_summary": tracker_timing_summary,
    }
    summary_path = timing_output_dir / f"{run_id}_timing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            convert_to_serializable(summary_payload),
            f,
            indent=4,
            ensure_ascii=False,
            sort_keys=True,
        )

    frames_path = None
    if tracker_timing_frames:
        frames_path = timing_output_dir / f"{run_id}_tracking_frame_times.csv"
        fieldnames = list(tracker_timing_frames[0].keys())
        with open(frames_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tracker_timing_frames)

    logger.info(f"Timing summary saved to: {summary_path}")
    if frames_path is not None:
        logger.info(f"Per-frame timing CSV saved to: {frames_path}")
    return summary_path, frames_path


if __name__ == "__main__":
    script_start = perf_counter()
    phase_times = {}

    # Load configuration
    config = get_config()
    
    # Configure logging
    Logger.setup_from_config(config)
    logger = get_logger(__name__)
    
    logger.info("Starting football tracking system")
    
    try:
        # Get paths and parameters from config
        MODEL_PATH = str(config.get_path('paths', 'models', 'finetuned_player'))
        video_key_override = os.getenv("TRACK_VIDEO_KEY")
        if video_key_override:
            video_config_key = video_key_override.strip()
        else:
            video_config_key = (
                "video_prueba_ajustado"
                if config.get('paths', 'data', 'video_prueba_ajustado') is not None
                else "video_prueba"
            )
        VIDEO_PATH = str(config.get_path('paths', 'data', video_config_key))
        OUTPUT = str(config.get_path('paths', 'output', 'prueba_tracker', create_if_missing=True) / "nueva_prueba.mp4")
        tracks_output_root = config.get_path(
            'paths',
            'output',
            'tracks_json',
            create_if_missing=True,
        )
        OUTPUT_PATH = str(build_tracks_output_path(VIDEO_PATH, tracks_output_root))
        
        # Configuration parameters
        CONF = config.get('detection', 'conf_threshold')
        BALL_MIN_CONF = config.get('detection', 'ball_min_conf')
        SHOWKMEANS = config.get('visualization', 'show_kmeans')
        SHOW_OUTPUT = config.get('visualization', 'show_output')
        show_output_override = parse_bool_env("TRACK_SHOW_OUTPUT")
        if show_output_override is not None:
            SHOW_OUTPUT = show_output_override
        track_max_frames = parse_int_env("TRACK_MAX_FRAMES")
        if track_max_frames is not None and track_max_frames <= 0:
            track_max_frames = None
        
        # Tracking configuration
        tracking_cfg = config.tracking
        timing_enabled = bool(tracking_cfg.get("timing_enabled", True))
        timing_log_every_n_frames = int(
            tracking_cfg.get("timing_log_every_n_frames", 25)
        )
        timing_save_per_frame = bool(tracking_cfg.get("timing_save_per_frame", True))
        timing_output_rel = config.get(
            "paths",
            "output",
            "timing_reports",
            default="output/timing_reports",
        )
        timing_output_dir = config.project_root / timing_output_rel

        MAX_TRACKS_PER_CLASS = tracking_cfg.get(
            "max_tracks_per_class",
            {"player": 22, "ball": 1, "referee": 3},
        )
        default_max_total_tracks = int(
            sum(int(value) for value in MAX_TRACKS_PER_CLASS.values())
        )
        TRACKER_CONF = {
            "track_thresh": tracking_cfg['track_thresh'],
            "track_buffer": tracking_cfg['track_buffer'],
            "match_thresh": tracking_cfg['match_thresh'],
            "frame_rate": tracking_cfg['frame_rate'],
            "minimum_consecutive_frames": tracking_cfg['minimum_consecutive_frames'],
            "max_total_tracks": tracking_cfg.get(
                "max_total_tracks", default_max_total_tracks
            ),
            "enforce_internal_class_limits": tracking_cfg.get(
                "enforce_internal_class_limits", False
            ),
            "team_mismatch_penalty": tracking_cfg.get("team_mismatch_penalty", 1000.0),
            "second_match_threshold": tracking_cfg.get("second_match_threshold", 0.9),
            "unconfirmed_match_threshold": tracking_cfg.get(
                "unconfirmed_match_threshold", 0.8
            ),
            "reassign_motion_factor": tracking_cfg.get("reassign_motion_factor", 1.0),
            "reassign_min_distance": tracking_cfg.get("reassign_min_distance", 25.0),
            "reassign_hard_max_distance": tracking_cfg.get(
                "reassign_hard_max_distance", None
            ),
            "reassign_min_samples": tracking_cfg.get("reassign_min_samples", 3),
            "reassign_max_lost_frames": tracking_cfg.get(
                "reassign_max_lost_frames", 12
            ),
            "canonical_cleanup_lost_frames": tracking_cfg.get(
                "canonical_cleanup_lost_frames",
                tracking_cfg.get("track_buffer", 90),
            ),
            "class_limit_lost_frames": tracking_cfg.get(
                "class_limit_lost_frames",
                tracking_cfg.get(
                    "reassign_max_lost_frames",
                    12,
                ),
            ),
            "raw_id_grace_lost_frames": tracking_cfg.get(
                "raw_id_grace_lost_frames", 2
            ),
            "referee_recovery_max_lost_frames": tracking_cfg.get(
                "referee_recovery_max_lost_frames", 3
            ),
            "referee_recovery_max_distance": tracking_cfg.get(
                "referee_recovery_max_distance", 45.0
            ),
            "team_inference_cache_enabled": tracking_cfg.get(
                "team_inference_cache_enabled", False
            ),
            "team_inference_cache_iou_threshold": tracking_cfg.get(
                "team_inference_cache_iou_threshold", 0.35
            ),
            "team_inference_classes": tracking_cfg.get(
                "team_inference_classes", ["player", "goalkeeper"]
            ),
            "timing_enabled": timing_enabled,
            "timing_log_every_n_frames": timing_log_every_n_frames,
            "timing_save_per_frame": timing_save_per_frame,
        }
        FIELD_TRACKING_CONF = {
            "enabled": tracking_cfg.get("use_field_positions", True),
            "method": tracking_cfg.get("field_position_method", "pnlcalib"),
            "classes": tracking_cfg.get("field_position_classes", ["player", "goalkeeper"]),
            "field_length_m": tracking_cfg.get("field_length_m", 106.0),
            "field_width_m": tracking_cfg.get("field_width_m", 68.0),
            "max_width": tracking_cfg.get("field_position_max_width", 1280),
            "bottom_offset_ratio": tracking_cfg.get(
                "field_position_bottom_offset_ratio", 0.04
            ),
            "keypoint_threshold": tracking_cfg.get(
                "field_position_keypoint_threshold", 0.3434
            ),
            "line_threshold": tracking_cfg.get(
                "field_position_line_threshold", 0.7867
            ),
            "pnl_refine": tracking_cfg.get("field_position_pnl_refine", True),
            "temporal_blend": tracking_cfg.get(
                "field_position_temporal_blend", 0.20
            ),
            "pixels_per_meter": tracking_cfg.get(
                "field_position_pixels_per_meter", 8
            ),
            "device": tracking_cfg.get("field_position_device"),
            "strict_field_position_matching": tracking_cfg.get(
                "field_position_strict_matching", True
            ),
            "match_distance_gate_m": tracking_cfg.get(
                "field_position_match_distance_gate_m", 8.0
            ),
            "match_distance_weight": tracking_cfg.get(
                "field_position_match_distance_weight", 0.25
            ),
            "reassign_min_field_distance_m": tracking_cfg.get(
                "reassign_min_field_distance_m", 4.0
            ),
            "reassign_hard_max_field_distance_m": tracking_cfg.get(
                "reassign_hard_max_field_distance_m", None
            ),
        }
        
        # Team colors
        TEAM_COLORS = config.get_team_colors()
        
        logger.info(f"Model: {MODEL_PATH}")
        logger.info(f"Video: {VIDEO_PATH}")
        logger.info(f"Video key: {video_config_key}")
        logger.info(f"Tracking configuration: {TRACKER_CONF}")
        logger.info(f"Field tracking configuration: {FIELD_TRACKING_CONF}")
        if track_max_frames is not None:
            logger.info("TRACK_MAX_FRAMES activo: %d", track_max_frames)
        logger.info(
            (
                "Timing configuration: enabled=%s, log_every_n_frames=%d, "
                "save_per_frame=%s, output_dir=%s"
            ),
            timing_enabled,
            timing_log_every_n_frames,
            timing_save_per_frame,
            timing_output_dir,
        )
        
        # Run tracking
        tracker_init_start = perf_counter()
        tracker = Tracker(
            MODEL_PATH,
            CONF,
            TRACKER_CONF,
            TEAM_COLORS,
            ball_min_conf=BALL_MIN_CONF,
            max_tracks_per_class=MAX_TRACKS_PER_CLASS,
            field_tracking_conf=FIELD_TRACKING_CONF,
            project_root=config.project_root,
        )
        phase_times["tracker_init_s"] = float(perf_counter() - tracker_init_start)

        logger.info("Extracting tracks from video...")
        tracking_start = perf_counter()
        tracks = tracker.get_tracks(
            VIDEO_PATH,
            SHOWKMEANS,
            max_frames=track_max_frames,
        )
        phase_times["tracking_s"] = float(perf_counter() - tracking_start)
        
        # Draw tracks
        colors = config.get_visualization_colors()
        drawer = Drawer(colors=colors)
        logger.info("Drawing tracks on video...")
        drawing_start = perf_counter()
        drawer.draw_tracks(tracks, VIDEO_PATH, OUTPUT, show=SHOW_OUTPUT)
        phase_times["drawing_s"] = float(perf_counter() - drawing_start)
        logger.info(f"Video with tracks saved to: {OUTPUT}")
        
        # Evaluation
        logger.info("Evaluating tracks...")
        evaluation_start = perf_counter()
        evaluator = Evaluator()
        evaluation = evaluator.evaluate(["player"], tracks)
        summary = evaluation["player"]["summary"]
        phase_times["evaluation_s"] = float(perf_counter() - evaluation_start)
        
        # Save tracks JSON
        save_tracks_start = perf_counter()
        save_result(tracks, OUTPUT_PATH, logger)
        phase_times["save_tracks_s"] = float(perf_counter() - save_tracks_start)

        phase_times["total_pipeline_s"] = float(perf_counter() - script_start)
        
        # Show summary
        logger.info("Evaluation summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        print("\n=== TRACKING SUMMARY ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        if timing_enabled:
            tracker_timing_summary = tracker.last_timing_summary
            tracker_timing_frames = tracker.last_timing_frames
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_timing_reports(
                timing_output_dir=timing_output_dir,
                run_id=run_id,
                script_phase_times=phase_times,
                tracker_timing_summary=tracker_timing_summary,
                tracker_timing_frames=tracker_timing_frames,
                logger=logger,
            )

            logger.info("Timing summary (script phases):")
            for phase_name, duration_s in sorted(
                phase_times.items(), key=lambda item: item[1], reverse=True
            ):
                logger.info(f"  {phase_name}: {duration_s:.4f}s")
            if tracker_timing_summary:
                logger.info("Timing summary (tracker per-frame mean):")
                for phase_name, stats in sorted(
                    tracker_timing_summary.get("phases", {}).items(),
                    key=lambda item: item[1].get("mean_s", 0.0),
                    reverse=True,
                ):
                    logger.info(
                        "  %s: mean=%.4fs total=%.4fs p95=%.4fs max=%.4fs",
                        phase_name,
                        stats.get("mean_s", 0.0),
                        stats.get("total_s", 0.0),
                        stats.get("p95_s", 0.0),
                        stats.get("max_s", 0.0),
                    )
                tracker_metrics = tracker_timing_summary.get("metrics", {})
                if tracker_metrics:
                    logger.info("Tracker metrics (no tiempo):")
                    for metric_name, stats in sorted(tracker_metrics.items()):
                        logger.info(
                            "  %s: mean=%.4f min=%.4f p95=%.4f max=%.4f",
                            metric_name,
                            stats.get("mean", 0.0),
                            stats.get("min", 0.0),
                            stats.get("p95", 0.0),
                            stats.get("max", 0.0),
                        )
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Process completed successfully")
