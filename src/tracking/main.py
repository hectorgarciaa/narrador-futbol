import json
import numpy as np

from trackers import Tracker
from evaluators import Evaluator

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

    MODEL_PATH = "../../models/finetuning/v11/yolov11m/weights/best.pt"
    VIDEO = "../../data/partidoPrueba/08fd33_4_medio.mp4"
    OUTPUT = "../../output/tracks_json/tracker/tracks.json"
    
    SHOWKMEANS = False
    TEAM_COLORS = { "Real Madrid": np.array([255, 127, 127]),       "Wolfsburgo":  np.array([224, 77, 196]) }

    CONF = 0.1
    TT = 0.5
    MT = 0.945
    MCF = 5
    
    TRACKER_CONF = { "track_thresh": TT, "track_buffer": 90, "match_thresh": MT, "frame_rate": 25, "minimum_consecutive_frames": MCF }

    tracker = Tracker(MODEL_PATH, CONF, TRACKER_CONF, TEAM_COLORS)
    tracks = tracker.get_tracks(VIDEO, SHOWKMEANS)

    evaluator = Evaluator()
    metrics, metrics_list, summary, n_frames = evaluator.evaluate(tracks)
    
    with open("OUTPUT", "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(tracks), f, indent=4, ensure_ascii=False, sort_keys=True)