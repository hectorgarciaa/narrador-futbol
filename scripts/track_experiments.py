import json
import sys
import time
import numpy as np

from football_ai.tracking import Tracker
from football_ai.visualization import Drawer
from football_ai.core import get_config, get_logger, Logger, convert_to_serializable

if __name__ == "__main__":
    config = get_config()
    Logger.setup_from_config(config)
    logger = get_logger(__name__)

    MODEL_PATH = str(config.get_path('paths', 'models', 'finetuned_player'))
    VIDEO = str(config.get_path('paths', 'data', 'video_08fd33_medio'))
    OUTPUT = str(config.get_path('paths', 'output', 'prueba_tracker', create_if_missing=True))
    SHOWKMEANS = config.get('visualization', 'show_kmeans')
    TEAM_COLORS = config.get_team_colors()

    # Colores de visualización
    vis_colors = config.get_visualization_colors()

    confs = [0.1]
    tts = [0.5]
    mts = [0.945]
    mcfs = [5]

    prueba_id = 0
    total_pruebas = len(confs) * len(tts) * len(mts) * len(mcfs)
    tracks_todos = []
    for conf in confs:
        for tt in tts:
            for mt in mts:
                for mcf in mcfs:
                    inicio = time.time()
                    tracker_conf = {
                        "track_thresh": tt,
                        "track_buffer": config.get('tracking', 'track_buffer'),
                        "match_thresh": mt,
                        "frame_rate": config.get('tracking', 'frame_rate'),
                        "minimum_consecutive_frames": mcf,
                    }
                    output = OUTPUT + "/" + str(prueba_id) + ".mp4"
                    prueba_id += 1

                    tracker = Tracker(MODEL_PATH, conf, tracker_conf, TEAM_COLORS)
                    tracks = tracker.get_tracks(VIDEO, SHOWKMEANS)

                    tracks_todos.append({"conf": conf, "tt": tt, "mt": mt, "mcf": mcf, "track": tracks})

                    drawer = Drawer(colors=vis_colors)
                    drawer.draw_tracks(tracks, VIDEO, output)

                    if prueba_id == 1:
                        with open("./tracks_prueba.json", "w", encoding="utf-8") as f:
                            json.dump(convert_to_serializable(tracks), f, indent=4, ensure_ascii=False, sort_keys=True)
                    fin = time.time()
                    logger.info(f"Prueba {prueba_id}/{total_pruebas} - {int(fin - inicio)}s")

            with open("./tracks.json", "w", encoding="utf-8") as f:
                json.dump(convert_to_serializable(tracks_todos), f, indent=4, ensure_ascii=False, sort_keys=True)
