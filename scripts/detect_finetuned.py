import sys
from ultralytics import YOLO

from football_ai.core import get_config, get_logger, Logger


def detect(model_path, video_path, output_path, logger):
    """Performs detection using a fine-tuned YOLO model."""
    try:
        logger.info(f"Loading fine-tuned model from: {model_path}")
        model = YOLO(model_path)

        logger.info(f"Running detection on video: {video_path}")
        results = model(video_path, save=True, project=output_path, name="finetuningTest", exist_ok=True)

        logger.info(f"Results saved to: {output_path}/finetuningTest")
        return results
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during detection: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    config = get_config()
    Logger.setup_from_config(config)
    logger = get_logger(__name__)

    logger.info("Starting detection with fine-tuned model")

    try:
        model_path = str(config.get_path('paths', 'models', 'finetuned_player'))
        video_path = str(config.get_path('paths', 'data', 'video_08fd33'))
        output_path = str(config.get_path('paths', 'output', 'prueba_finetuning', create_if_missing=True))

        detect(model_path, video_path, output_path, logger)
        logger.info("Detection completed successfully")
    except Exception as e:
        logger.error(f"Execution error: {e}")
        sys.exit(1)
