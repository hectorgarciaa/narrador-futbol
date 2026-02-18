import sys
from pathlib import Path
from ultralytics import YOLO

from football_ai.core import get_config, get_logger, Logger
from football_ai.detection import DetectR8

def detect(ruta_modelo, ruta_partido, ruta_salida, logger):
    """Detecta balón en un video usando modelo fine-tuned con DetectR8."""
    try:
        logger.info(f"Cargando modelo desde: {ruta_modelo}")
        model = YOLO(ruta_modelo)
        model.model[-1] = DetectR8(nc=model.model[-1].nc, ch=model.model[-1].ch)
        
        logger.info(f"Procesando video: {ruta_partido}")
        results = model(ruta_partido, save=True, project=ruta_salida, name="partido_prueba_balon.mp4", exist_ok=True)
        
        logger.info(f"Resultados guardados en: {ruta_salida}")
        return results
    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado: {e}")
        raise
    except Exception as e:
        logger.error(f"Error durante la detección: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Cargar configuración
    config = get_config()
    
    # Configurar logging
    Logger.setup_from_config(config)
    logger = get_logger(__name__)
    
    logger.info("Iniciando detección de balón")
    
    try:
        ruta_modelo = str(config.get_path('paths', 'models', 'finetuned_ball'))
        ruta_partido = str(config.get_path('paths', 'data', 'video_prueba_medio'))
        ruta_salida = str(config.get_path('paths', 'output', 'prueba_finetuning', create_if_missing=True) / "pruebaDeteccionBalon")
        
        detect(ruta_modelo, ruta_partido, ruta_salida, logger)
        logger.info("Detección completada exitosamente")
    except Exception as e:
        logger.error(f"Error en la ejecución: {e}")
        sys.exit(1)
