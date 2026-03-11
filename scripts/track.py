import argparse
import json
import re
import sys
from pathlib import Path

# Permite ejecutar `python scripts/track.py` sin instalar el paquete en editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.tracking import Tracker
from football_ai.evaluation import Evaluator
from football_ai.visualization import Drawer
from football_ai.core import get_config, get_logger, Logger, convert_to_serializable


def parse_args():
    """Parse CLI arguments for the tracking script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run full tracking pipeline. Optionally pass a video shortcut from "
            "config.yaml paths.data (e.g. video_prueba_ajustado)."
        )
    )
    parser.add_argument(
        "video_shortcut",
        nargs="?",
        default=None,
        help=(
            "Optional key inside paths.data (config.yaml), e.g. video_prueba_ajustado. "
            "You can also pass a direct video path."
        ),
    )
    return parser.parse_args()


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


if __name__ == "__main__":
    args = parse_args()

    # Load configuration
    config = get_config()
    
    # Configure logging
    Logger.setup_from_config(config)
    logger = get_logger(__name__)
    
    logger.info("Starting football tracking system")
    
    try:
        # Get paths and parameters from config
        MODEL_PATH = str(config.get_path('paths', 'models', 'finetuned_player'))
        VIDEO_PATH, video_source = resolve_video_path(config, args.video_shortcut)
        OUTPUT = build_output_video_path(config, VIDEO_PATH)
        OUTPUT_PATH_NAMED, OUTPUT_PATH_LEGACY = build_tracks_output_paths(
            config, VIDEO_PATH
        )
        
        # Configuration parameters
        CONF = config.get('detection', 'conf_threshold')
        BALL_MIN_CONF = config.get('detection', 'ball_min_conf')
        SHOWKMEANS = config.get('visualization', 'show_kmeans')
        SHOW_OUTPUT = config.get('visualization', 'show_output')
        
        # Tracking configuration
        tracking_cfg = config.tracking
        MAX_TRACKS_PER_CLASS = tracking_cfg.get(
            "max_tracks_per_class",
            {"player": 22, "ball": 1, "referee": 3},
        )
        TRACKER_CONF = {
            "track_thresh": tracking_cfg['track_thresh'],
            "track_buffer": tracking_cfg['track_buffer'],
            "match_thresh": tracking_cfg['match_thresh'],
            "frame_rate": tracking_cfg['frame_rate'],
            "minimum_consecutive_frames": tracking_cfg['minimum_consecutive_frames'],
            "max_total_tracks": tracking_cfg.get("max_total_tracks", 25),
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
            "reassign_min_samples": tracking_cfg.get("reassign_min_samples", 3),
            "referee_recovery_max_lost_frames": tracking_cfg.get(
                "referee_recovery_max_lost_frames", 3
            ),
            "referee_recovery_max_distance": tracking_cfg.get(
                "referee_recovery_max_distance", 45.0
            ),
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
            "match_distance_gate_m": tracking_cfg.get(
                "field_position_match_distance_gate_m", 8.0
            ),
            "match_distance_weight": tracking_cfg.get(
                "field_position_match_distance_weight", 0.25
            ),
            "reassign_min_field_distance_m": tracking_cfg.get(
                "reassign_min_field_distance_m", 4.0
            ),
        }
        
        # Team colors
        TEAM_COLORS = config.get_team_colors()
        
        logger.info(f"Model: {MODEL_PATH}")
        logger.info(f"Video: {VIDEO_PATH}")
        logger.info(f"Video source: {video_source}")
        logger.info(f"Named tracks JSON output: {OUTPUT_PATH_NAMED}")
        logger.info(f"Legacy tracks JSON output: {OUTPUT_PATH_LEGACY}")
        logger.info(f"Tracking configuration: {TRACKER_CONF}")
        logger.info(f"Field tracking configuration: {FIELD_TRACKING_CONF}")
        
        # Run tracking
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
        logger.info("Extracting tracks from video...")
        tracks = tracker.get_tracks(VIDEO_PATH, SHOWKMEANS)
        
        # Draw tracks
        colors = config.get_visualization_colors()
        drawer = Drawer(colors=colors)
        logger.info("Drawing tracks on video...")
        drawer.draw_tracks(tracks, VIDEO_PATH, OUTPUT, show=SHOW_OUTPUT)
        logger.info(f"Video with tracks saved to: {OUTPUT}")
        
        # Evaluation
        logger.info("Evaluating tracks...")
        evaluator = Evaluator()
        evaluation = evaluator.evaluate(["player"], tracks)
        summary = evaluation["player"]["summary"]
        
        # Save tracks JSON
        save_result(tracks, OUTPUT_PATH_NAMED, logger)
        if OUTPUT_PATH_LEGACY != OUTPUT_PATH_NAMED:
            save_result(tracks, OUTPUT_PATH_LEGACY, logger)
        
        # Show summary
        logger.info("Evaluation summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        print("\n=== TRACKING SUMMARY ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Process completed successfully")
