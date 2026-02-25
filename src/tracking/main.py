import json
import numpy as np

from trackers import Tracker
from evaluators import Evaluator
from evaluators.MetricsVisualizer import MetricsVisualizer  
from drawer.drawer import Drawer

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
    
def saveResult(tracks):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(tracks), f, indent=4, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":

    MODEL_PATH  = "../../models/finetuning/v11/yolov11m/weights/best.pt"
    VIDEO_PATH  = "../../data/partidoPrueba/partido_ajustado.mp4"
    TRACK_PATH  = "../../output/tracks_json/tracker/tracks_m.json"
    OUTPUT_PATH = "../../output/tracks_json/tracker/tracks.json"
    SHOWKMEANS = False
    TEAM_COLORS = { "Real Madrid": np.array([255, 127, 127]),       "Wolfsburgo":  np.array([224, 77, 196]) }

    OUTPUT = "../../output/pruebaTracker/nueva_prueba.mp4"
    SHOW_OUTPUT = True

    #CONF = 0.1
    CONF = 0.01
    TT = 0.5
    MT = 0.945
    MCF = 5
    BALL_MIN_CONF = 0.01
    MAX_TRACKS_PER_CLASS = {
        "player": 22,
        "ball": 1,
        "referee": 3,
    }
    
    TRACKER_CONF = {
        "track_thresh": TT,
        "track_buffer": 90,
        "match_thresh": MT,
        "frame_rate": 25,
        "minimum_consecutive_frames": MCF,
        "max_total_tracks": 25,
        "enforce_internal_class_limits": False,
        "team_mismatch_penalty": 1000.0,
        "second_match_threshold": 0.9,
        "unconfirmed_match_threshold": 0.8,
        "reassign_motion_factor": 1.0,
        "reassign_min_distance": 25.0,
        "reassign_min_samples": 3,
        "referee_recovery_max_lost_frames": 3,
        "referee_recovery_max_distance": 45.0,
    }

    tracker = Tracker(
        MODEL_PATH,
        CONF,
        TRACKER_CONF,
        TEAM_COLORS,
        ball_min_conf=BALL_MIN_CONF,
        max_tracks_per_class=MAX_TRACKS_PER_CLASS,
    )
    tracks = tracker.get_tracks(VIDEO_PATH, SHOWKMEANS)
    drawer = Drawer(colors = { "player": (0, 255, 0), "goalkeeper": (0, 255, 255), "referee": (255, 0, 0), "ball": (0, 0, 255) })
    drawer.draw_tracks(tracks, VIDEO_PATH, OUTPUT, show=SHOW_OUTPUT)

    #with open(TRACK_PATH, "r", encoding="utf-8") as f:
    #   tracks = json.load(f, )

    evaluator = Evaluator()
    evaluation = evaluator.evaluate(["player"], tracks)
    metrics, metrics_list, summary, n_frames = evaluation["player"]["metrics"], evaluation["player"]["metrics_list"], evaluation["player"]["summary"], evaluation["player"]["n_frames"], 
    
    
    #saveResult(tracks)
    print(summary)
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
    

    
