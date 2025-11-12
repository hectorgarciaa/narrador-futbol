import os
import json
import pickle
import numpy as np
import pandas as pd
import supervision as sv

from detectors import Detector, TeamDetector
from trackers import ByteTrack
from drawer import Drawer

class Tracker:
    def __init__(self, model_path, conf, traker_conf, team_colors):
        self.model = Detector(model_path, conf)
        self.teamDetector = TeamDetector(team_colors)
        self.tracker = ByteTrack(traker_conf["track_thresh"], traker_conf["track_buffer"], traker_conf["match_thresh"], traker_conf["frame_rate"], tracker_conf["minimum_consecutive_frames"])
    
    def get_tracks(self, video, showKMeans):
        model_detections = self.model.detect(video)
        tracks = {"player": [], "goalkeeper": [], "referee": [], "ball": [] }
        for n_frame, detections in enumerate(model_detections):
            detections_sv = sv.Detections.from_ultralytics(detections)

            teams_of_detected_objects = self.teamDetector.detectTeams(detections, showKMeans)
            teams_labels = [dicc["team"] for dicc in teams_of_detected_objects]
            
            tracks_detection = self.tracker.update_with_detections(detections_sv, teams_labels)
            
            for key in tracks.keys():
                tracks[key].append({})

            for i, object_detected in enumerate(tracks_detection):
                bbox, _, confidence, _, tracker_id, class_name = object_detected
                class_name = class_name["class_name"]

                tracks[class_name][n_frame][tracker_id] = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "team": teams_of_detected_objects[i]["team"],
                    "distances": teams_of_detected_objects[i]["distances"],
                    "shirt_color": teams_of_detected_objects[i]["shirt_color"],
                    "bbox_size": teams_of_detected_objects[i]["bbox_size"],
                }
        
        return tracks
    
"""
if __name__ == "__main__":
    MODEL_PATH = "../../models/finetuning/v11/yolov11x/weights/best.pt"
    VIDEO = "../../data/partidoPrueba/08fd33_4_medio.mp4"
    OUTPUT = "../../output/pruebaTracker/08fd33_4_.mp4"
    CONF = 0.05
    SHOWKMEANS = False
    TEAM_COLORS = {     "Real Madrid": np.array([0, 0, 245]),       "Wolfsburgo":  np.array([70, 150, 150])     }
    TRACKER_CONF = {
        "track_thresh": CONF,
        "track_buffer": 90,
        "match_thresh": 1,
        "frame_rate": 25
    }
    
    tracker = Tracker(MODEL_PATH, CONF, TRACKER_CONF, TEAM_COLORS)
    tracks = tracker.get_tracks(VIDEO, SHOWKMEANS)
    
    drawer = Drawer(colors = { "player": (0, 255, 0), "goalkeeper": (0, 255, 255), "referee": (255, 0, 0), "ball": (0, 0, 255) })
    drawer.draw_tracks(tracks, VIDEO, OUTPUT)
"""

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
    OUTPUT = "../../output/pruebaTracker/08fd33_4_pruebas_"
    SHOWKMEANS = False
    TEAM_COLORS = { "Real Madrid": np.array([255, 127, 127]),       "Wolfsburgo":  np.array([224, 77, 196]) }

    # confs = [0.01, 0.05, 0.1, 0.2]
    # tts = [0.1, 0.35, 0.7]
    # mts = [0.933, 0.966, 1]
    # mcfs = [1, 5, 10]

    confs = [0.1]
    tts = [0.35]
    mts = [0.99]
    mcfs = [1]
    
    prueba_id = 0
    total_pruebas = len(confs) * len(tts) * len(mts) * len(mcfs)
    tracks_todos = []
    for conf in confs:
        for tt in tts:
            for mt in mts:
                for mcf in mcfs:
                    print("\n\n\nPrueba: ", prueba_id, "/", total_pruebas)
                    print("#"*100, end="\n\n")
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

            try:                             
                with open("./tracks.json", "w", encoding="utf-8") as f:
                    json.dump(convert_to_serializable(tracks_todos), f, indent=4, ensure_ascii=False, sort_keys=True)
            except:
                with open("./tracks.pkl", "rb") as f:
                    tracks_todos = pickle.load(f)

    print(tracker.teamDetector.team_colors)
    print(tracker.teamDetector.confirmed_teams)
    print(tracker.teamDetector.color_samples)
