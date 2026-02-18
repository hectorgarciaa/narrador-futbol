import sys
from pathlib import Path
from ultralytics import YOLO

from football_ai.core import get_config, get_logger, Logger

def finetuning(ruta_modelo, ruta_data_yaml, epochs, batch, imgsz, output_dir, logger):
    """Realiza fine-tuning de modelo YOLO para detección de fútbol."""
    try:
        logger.info(f"Cargando modelo base: {ruta_modelo}")
        model = YOLO(ruta_modelo)
        
        logger.info(f"Iniciando fine-tuning con dataset: {ruta_data_yaml}")
        logger.info(f"Parámetros: epochs={epochs}, batch={batch}, imgsz={imgsz}")
        
        model.train(
            data=ruta_data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name="finetuning",
            project=output_dir,
        )
        
        logger.info(f"Fine-tuning completado. Modelo guardado en: {output_dir}/finetuning")
    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado: {e}")
        raise
    except Exception as e:
        logger.error(f"Error durante el fine-tuning: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Cargar configuración
    config = get_config()
    
    # Configurar logging
    Logger.setup_from_config(config)
    logger = get_logger(__name__)
    
    logger.info("Iniciando proceso de fine-tuning")
    
    try:
        ruta_modelo = str(config.get_path('paths', 'models', 'yolo_v11_m'))
        ruta_data_yaml = str(config.get_path('paths', 'data', 'dataset_football_ai'))
        
        # Construir directorio de salida
        output_base = config.get_path('paths', 'models', 'finetuned_player').parent.parent
        output_dir = str(output_base)
        
        # Parámetros de fine-tuning desde config
        epochs = config.get('finetuning', 'epochs')
        batch = config.get('finetuning', 'batch')
        imgsz = config.get('finetuning', 'imgsz')
        
        finetuning(ruta_modelo, ruta_data_yaml, epochs, batch, imgsz, output_dir, logger)
        logger.info("Fine-tuning completado exitosamente")
    except Exception as e:
        logger.error(f"Error en la ejecución: {e}")
        sys.exit(1)
