import sys
from ultralytics import YOLO

from football_ai.core import get_config, get_logger, Logger


def detect(ruta_modelo, ruta_partido, ruta_salida, logger):
    """Realiza detección usando modelo YOLO finetuneado."""
    try:
        logger.info(f"Cargando modelo finetuneado desde: {ruta_modelo}")
        model = YOLO(ruta_modelo)

        logger.info(f"Ejecutando detección en video: {ruta_partido}")
        results = model(ruta_partido, save=True, project=ruta_salida, name="pruebaFinetuning", exist_ok=True)

        logger.info(f"Resultados guardados en: {ruta_salida}/pruebaFinetuning")
        return results
    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado: {e}")
        raise
    except Exception as e:
        logger.error(f"Error durante la detección: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    config = get_config()
    Logger.setup_from_config(config)
    logger = get_logger(__name__)

    logger.info("Iniciando detección con modelo finetuneado")

    try:
        ruta_modelo = str(config.get_path('paths', 'models', 'finetuned_player'))
        ruta_partido = str(config.get_path('paths', 'data', 'video_08fd33'))
        ruta_salida = str(config.get_path('paths', 'output', 'prueba_finetuning', create_if_missing=True))

        detect(ruta_modelo, ruta_partido, ruta_salida, logger)
        logger.info("Detección completada exitosamente")
    except Exception as e:
        logger.error(f"Error en la ejecución: {e}")
        sys.exit(1)
