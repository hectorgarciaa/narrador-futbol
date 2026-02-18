import json
import sys
from pathlib import Path
import numpy as np

from football_ai.tracking import Tracker
from football_ai.evaluation import Evaluator, MetricsVisualizer
from football_ai.visualization import Drawer
from football_ai.core import get_config, get_logger, Logger, convert_to_serializable

def save_result(tracks, output_path, logger):
    """Guarda los tracks en formato JSON con manejo de errores."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(tracks), f, indent=4, ensure_ascii=False, sort_keys=True)
        logger.info(f"Tracks guardados en: {output_path}")
    except IOError as e:
        logger.error(f"Error al guardar tracks en {output_path}: {e}")
    except Exception as e:
        logger.error(f"Error inesperado al guardar tracks: {e}")


if __name__ == "__main__":
    # Cargar configuración
    config = get_config()
    
    # Configurar logging
    Logger.setup_from_config(config)
    logger = get_logger(__name__)
    
    logger.info("Iniciando sistema de tracking de fútbol")
    
    try:
        # Obtener rutas y parámetros desde config
        MODEL_PATH = str(config.get_path('paths', 'models', 'finetuned_player'))
        VIDEO_PATH = str(config.get_path('paths', 'data', 'video_prueba'))
        OUTPUT = str(config.get_path('paths', 'output', 'prueba_tracker', create_if_missing=True) / "nueva_prueba.mp4")
        OUTPUT_PATH = str(config.get_path('paths', 'output', 'tracks_json', create_if_missing=True) / "tracker" / "tracks.json")
        
        # Parámetros de configuración
        CONF = config.get('detection', 'conf_threshold')
        BALL_MIN_CONF = config.get('detection', 'ball_min_conf')
        SHOWKMEANS = config.get('visualization', 'show_kmeans')
        SHOW_OUTPUT = config.get('visualization', 'show_output')
        
        # Configuración de tracking
        tracking_cfg = config.tracking
        TRACKER_CONF = {
            "track_thresh": tracking_cfg['track_thresh'],
            "track_buffer": tracking_cfg['track_buffer'],
            "match_thresh": tracking_cfg['match_thresh'],
            "frame_rate": tracking_cfg['frame_rate'],
            "minimum_consecutive_frames": tracking_cfg['minimum_consecutive_frames']
        }
        
        # Colores de equipos
        TEAM_COLORS = config.get_team_colors()
        
        logger.info(f"Modelo: {MODEL_PATH}")
        logger.info(f"Video: {VIDEO_PATH}")
        logger.info(f"Configuración de tracking: {TRACKER_CONF}")
        
        # Ejecutar tracking
        tracker = Tracker(MODEL_PATH, CONF, TRACKER_CONF, TEAM_COLORS, ball_min_conf=BALL_MIN_CONF)
        logger.info("Extrayendo tracks del video...")
        tracks = tracker.get_tracks(VIDEO_PATH, SHOWKMEANS)
        
        # Dibujar tracks
        colors = config.get_visualization_colors()
        drawer = Drawer(colors=colors)
        logger.info("Dibujando tracks en video...")
        drawer.draw_tracks(tracks, VIDEO_PATH, OUTPUT, show=SHOW_OUTPUT)
        logger.info(f"Video con tracks guardado en: {OUTPUT}")
        
        # Evaluación
        logger.info("Evaluando tracks...")
        evaluator = Evaluator()
        evaluation = evaluator.evaluate(["player"], tracks)
        metrics = evaluation["player"]["metrics"]
        summary = evaluation["player"]["summary"]
        
        # Guardar resultados (comentado por defecto)
        # save_result(tracks, OUTPUT_PATH, logger)
        
        # Mostrar resumen
        logger.info("Resumen de evaluación:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        print("\n=== RESUMEN DE TRACKING ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        
    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Proceso completado exitosamente")
'''
    vis = MetricsVisualizer()
    # --- SPEED (por frame) ---
    speed_events = vis.collect_speed_events(metrics)
    vis.plot_speed_events_scatter(speed_events)  # scatter interactivo speed vs frame

    # --- COVERAGE (1 punto por track) ---
    cov_events = vis.collect_metric_events(metrics, "coverage")
    vis.plot_metric_events_scatter(cov_events, "coverage")

    # --- mean_speed por track ---
    mean_speed_events = vis.collect_metric_events(metrics, "mean_speed")
    vis.plot_metric_events_scatter(mean_speed_events, "mean_speed")

    # --- color_diff por track ---
    color_diff_events = vis.collect_metric_events(metrics, "color_diff")
    vis.plot_metric_events_scatter(color_diff_events, "color_diff")
'''
    

    
