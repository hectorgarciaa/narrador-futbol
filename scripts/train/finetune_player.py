import sys
from pathlib import Path
from ultralytics import YOLO

from football_ai.core import get_config, get_logger, Logger

def finetuning(model_path, data_yaml_path, epochs, batch, imgsz, output_dir, logger):
    """Performs fine-tuning of a YOLO model for football detection."""
    try:
        logger.info(f"Loading base model: {model_path}")
        model = YOLO(model_path)
        
        logger.info(f"Starting fine-tuning with dataset: {data_yaml_path}")
        logger.info(f"Parameters: epochs={epochs}, batch={batch}, imgsz={imgsz}")
        
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name="finetuning",
            project=output_dir,
        )
        
        logger.info(f"Fine-tuning completed. Model saved to: {output_dir}/finetuning")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Load configuration
    config = get_config()
    
    # Configure logging
    Logger.setup_from_config(config)
    logger = get_logger(__name__)
    
    logger.info("Starting fine-tuning process")
    
    try:
        model_path = str(config.get_path('paths', 'models', 'yolo_v11_m'))
        data_yaml_path = str(config.get_path('paths', 'data', 'dataset_football_ai'))
        
        # Build output directory
        output_base = config.get_path('paths', 'models', 'finetuned_player').parent
        output_dir = str(output_base)
        
        # Fine-tuning parameters from config
        epochs = config.get('finetuning', 'epochs')
        batch = config.get('finetuning', 'batch')
        imgsz = config.get('finetuning', 'imgsz')
        
        finetuning(model_path, data_yaml_path, epochs, batch, imgsz, output_dir, logger)
        logger.info("Fine-tuning completed successfully")
    except Exception as e:
        logger.error(f"Execution error: {e}")
        sys.exit(1)
