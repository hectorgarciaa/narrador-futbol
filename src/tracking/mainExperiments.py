import json
import numpy as np

from tracking import Tracker
from drawer import Drawer

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj

if __name__ == "__main__":
    import time

    MODEL_PATH = "../../models/finetuning/v11/yolov11m/weights/best.pt"
    VIDEO = "../../data/partidoPrueba/08fd33_4_medio.mp4"
    OUTPUT = "../../output/pruebaTracker/experimentos/"
    SHOWKMEANS = False
    TEAM_COLORS = { "Real Madrid": np.array([255, 127, 127]),       "Wolfsburgo":  np.array([224, 77, 196]) }

    confs = [0.1]
    tts = [0.5]
    mts = [0.945]
    mcfs = [5]
    # mts = [0.933, 0.966, 1]
    # tts = [0.3, 0.4, 0.5]
    # mts = [0.93, 0.945, 0.96]
    # mcfs = [4, 5, 6]
    
    prueba_id = 0
    total_pruebas = len(confs) * len(tts) * len(mts) * len(mcfs)
    tracks_todos = []
    for conf in confs:
        for tt in tts:
            for mt in mts:
                for mcf in mcfs:
                    inicio = time.time()
                    tracker_conf = { "track_thresh": tt, "track_buffer": 90, "match_thresh": mt, "frame_rate": 25, "minimum_consecutive_frames": mcf }
                    output = OUTPUT + str(prueba_id) + ".mp4"
                    prueba_id += 1

                    tracker = Tracker(MODEL_PATH, conf, tracker_conf, TEAM_COLORS)
                    tracks = tracker.get_tracks(VIDEO, SHOWKMEANS)
                
                    tracks_todos.append({"conf": conf, "tt": tt, "mt": mt, "mcf": mcf, "track": tracks})

                    drawer = Drawer(colors = { "player": (0, 255, 0), "goalkeeper": (0, 255, 255), "referee": (255, 0, 0), "ball": (0, 0, 255) })
                    drawer.draw_tracks(tracks, VIDEO, output)
                    
                    if prueba_id == 0:
                        with open("./tracks_prueba.json", "w", encoding="utf-8") as f:
                            json.dump(convert_to_serializable(tracks), f, indent=4, ensure_ascii=False, sort_keys=True)
                    fin = time.time()
                    print("Prueba: ", prueba_id, "/", total_pruebas, end="   ")
                    print(f"{int(fin - inicio)}", end="\n\n")

            with open("./tracks.json", "w", encoding="utf-8") as f:
                json.dump(convert_to_serializable(tracks_todos), f, indent=4, ensure_ascii=False, sort_keys=True)
