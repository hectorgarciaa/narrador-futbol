import json
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
    # Load configuration
    config = get_config()
    
    # Configure logging
    Logger.setup_from_config(config)
    logger = get_logger(__name__)
    
    logger.info("Starting football tracking system")
    
    try:
        # Get paths and parameters from config
        MODEL_PATH = str(config.get_path('paths', 'models', 'finetuned_player'))
        video_config_key = (
            "video_prueba_ajustado"
            if config.get('paths', 'data', 'video_prueba_ajustado') is not None
            else "video_prueba"
        )
        VIDEO_PATH = str(config.get_path('paths', 'data', video_config_key))
        OUTPUT = str(config.get_path('paths', 'output', 'prueba_tracker', create_if_missing=True) / "nueva_prueba.mp4")
        OUTPUT_PATH = str(config.get_path('paths', 'output', 'tracks_json', create_if_missing=True) / "tracker" / "tracks.json")
        
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
        
        # Team colors
        TEAM_COLORS = config.get_team_colors()
        
        logger.info(f"Model: {MODEL_PATH}")
        logger.info(f"Video: {VIDEO_PATH}")
        logger.info(f"Tracking configuration: {TRACKER_CONF}")
        
        # Run tracking
        tracker = Tracker(
            MODEL_PATH,
            CONF,
            TRACKER_CONF,
            TEAM_COLORS,
            ball_min_conf=BALL_MIN_CONF,
            max_tracks_per_class=MAX_TRACKS_PER_CLASS,
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
        save_result(tracks, OUTPUT_PATH, logger)
        
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
