import sys
import os
from pathlib import Path
from ultralytics import YOLO

from football_ai.core import get_config, get_logger, Logger

def finetune_phases(base_model_path, phases, project_dir, logger):
    """Performs multi-phase fine-tuning of a YOLO model."""
    current_model_path = base_model_path
    
    for idx, phase in enumerate(phases):
        phase_name = phase.get('name', f'phase_{idx}')
        if not phase.get('enabled', True):
            logger.info(f"Skipping phase: {phase_name} (disabled)")
            continue
            
        data_yaml = phase.get('dataset')
        epochs = phase.get('epochs', 50)
        batch = phase.get('batch', 16)
        imgsz = phase.get('imgsz', 640)
        output_name = phase.get('output_name', phase_name)
        
        logger.info(f"=== Starting Phase: {phase_name} ===")
        logger.info(f"Using model: {current_model_path}")
        logger.info(f"Dataset: {data_yaml}")
        logger.info(f"Params: epochs={epochs}, batch={batch}, imgsz={imgsz}")
        logger.info(f"Project Dir: {project_dir}, Run Name: {output_name}")
        
        try:
            model = YOLO(current_model_path)
            
            model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                name=output_name,
                project=project_dir,
                exist_ok=True # Permite reanudar o escribir en el mismo run name
            )
            
            # Calculate the path of the newly trained best.pt
            best_model_path = Path(project_dir) / output_name / "weights" / "best.pt"
            
            if not best_model_path.exists():
                logger.warning(f"Expected model weights not found at {best_model_path}.")
                logger.warning("If training crashed, next phase will use the previous model weights.")
            else:
                current_model_path = str(best_model_path)
                logger.info(f"Phase {phase_name} completed. Next phase will use weights from: {current_model_path}")
                
        except Exception as e:
            logger.error(f"Error during phase {phase_name}: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    # Load configuration
    config = get_config()
    
    # Configure logging
    Logger.setup_from_config(config)
    logger = get_logger(__name__)
    
    logger.info("Starting multi-phase fine-tuning process")
    
    try:
        model_path = str(config.get_path('paths', 'models', 'yolo_v11_m'))
        phases = config.get('finetuning', 'phases')
        
        if not phases:
            logger.error("No phases defined in config.yaml under 'finetuning: phases:'. Check your configuration.")
            sys.exit(1)
            
        # Determine base directory for all fine-tuning runs
        project_dir = str(Path("models/finetuning").resolve())
        Path(project_dir).mkdir(parents=True, exist_ok=True)
        
        finetune_phases(model_path, phases, project_dir, logger)
        logger.info("All enabled fine-tuning phases completed successfully")
    except Exception as e:
        logger.error(f"Execution error: {e}")
        sys.exit(1)
