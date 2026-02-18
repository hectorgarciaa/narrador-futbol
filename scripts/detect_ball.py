import sys
from pathlib import Path
from ultralytics import YOLO

from football_ai.core import get_config, get_logger, Logger
from football_ai.detection import DetectR8

def detect(model_path, video_path, output_path, logger):
    """Detects the ball in a video using a fine-tuned model with DetectR8."""
    try:
        logger.info(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        model.model[-1] = DetectR8(nc=model.model[-1].nc, ch=model.model[-1].ch)
        
        logger.info(f"Processing video: {video_path}")
        results = model(video_path, save=True, project=output_path, name="ball_detection_test.mp4", exist_ok=True)
        
        logger.info(f"Results saved to: {output_path}")
        return results
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during detection: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Load configuration
    config = get_config()
    
    # Configure logging
    Logger.setup_from_config(config)
    logger = get_logger(__name__)
    
    logger.info("Starting ball detection")
    
    try:
        model_path = str(config.get_path('paths', 'models', 'finetuned_ball'))
        video_path = str(config.get_path('paths', 'data', 'video_prueba_medio'))
        output_path = str(config.get_path('paths', 'output', 'prueba_finetuning', create_if_missing=True) / "ballDetectionTest")
        
        detect(model_path, video_path, output_path, logger)
        logger.info("Detection completed successfully")
    except Exception as e:
        logger.error(f"Execution error: {e}")
        sys.exit(1)
