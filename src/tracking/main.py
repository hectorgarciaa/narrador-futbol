import json
import numpy as np

from trackers import Tracker
from evaluators import Evaluator
from evaluators.MetricsVisualizer import MetricsVisualizer  

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
    VIDEO = "../../data/partidoPrueba/partido_medio.mp4"
    OUTPUT = "../../output/tracks_json/tracks.json"
    SHOWKMEANS = False
    TEAM_COLORS = { "Real Madrid": np.array([255, 127, 127]),       "Wolfsburgo":  np.array([224, 77, 196]) }

    #CONF = 0.1
    CONF = 0.01
    TT = 0.5
    MT = 0.945
    MCF = 5
    
    TRACKER_CONF = { "track_thresh": TT, "track_buffer": 90, "match_thresh": MT, "frame_rate": 25, "minimum_consecutive_frames": MCF }

    # tracker = Tracker(MODEL_PATH, CONF, TRACKER_CONF, TEAM_COLORS)
    # tracks = tracker.get_tracks(VIDEO, SHOWKMEANS)

    with open("./tracks_m_vmedio.json", "r", encoding="utf-8") as f:
        tracks = json.load(f, )

    evaluator = Evaluator()
    evaluation = evaluator.evaluate(["player"], tracks)
    metrics, metrics_list, summary, n_frames = evaluation["player"]["metrics"], evaluation["player"]["metrics_list"], evaluation["player"]["summary"], evaluation["player"]["n_frames"], 
    
    vis = MetricsVisualizer()

    print(summary)

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

    
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(tracks), f, indent=4, ensure_ascii=False, sort_keys=True)