import os
import json
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
                    "bbox": bbox,
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

def convert_keys_to_int(obj):
        if isinstance(obj, dict):
            return {int(k) if isinstance(k, (np.integer,)) else k: convert_keys_to_int(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys_to_int(i) for i in obj]
        else:
            return obj

if __name__ == "__main__":
    MODEL_PATH = "../../models/finetuning/v11/yolov11m/weights/best.pt"
    VIDEO = "../../data/partidoPrueba/08fd33_4_medio.mp4"
    OUTPUT = "../../output/pruebaTracker/08fd33_4_pruebas_"
    SHOWKMEANS = False
    TEAM_COLORS = {     "Real Madrid": np.array([0, 0, 245]),       "Wolfsburgo":  np.array([70, 150, 150])     }

    confs = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2]
    tts = [0.1, 0.25, 0.5, 0.7]
    mts = [0.9, 0.925, 0.95, 0.975, 0.99]
    mcfs = [1, 3, 5, 10]
    
    prueba_id = 0
    total_pruebas = len(confs) * len(tts) * len(mts)
    resultados, tracks_todos = [], []
    for conf in confs:
        for tt in tts:
            for mt in mts:
                for mcf in mcfs:
                    print("\n\n\nPrueba: ", prueba_id, "/", total_pruebas)
                    print("#"*100, end="\n\n")
                    tracker_conf = { "track_thresh": tt, "track_buffer": 90, "match_thresh": mt, "frame_rate": 25, "minimum_consecutive_frames": mcf }
                    output = OUTPUT + str(prueba_id) + ".mp4"
                    prueba_id += 1
                    try: 
                        tracker = Tracker(MODEL_PATH, conf, tracker_conf, TEAM_COLORS)
                        tracks = tracker.get_tracks(VIDEO, SHOWKMEANS)
                        
                        tracks_todos.append({"conf": conf, "tt": tt, "mt": mt, "track": tracks})
                        tracks_todos.append({"conf": conf, "tt": tt, "mt": mt, "mcf": mcf, "track": tracks})

                        metricas = {}
                        for clase in tracks:
                            # detecciones, confianza, frames vacíos
                            metricas[f"{clase}_detecciones_media"] = np.mean([len(f) for f in tracks[clase]]) if tracks[clase] else 0
                            metricas[f"{clase}_confianza_media"] = np.mean([d["confidence"] for f in tracks[clase] for d in f.values()]) if tracks[clase] else 0
                            metricas[f"{clase}_frames_vacios"] = sum(1 for f in tracks[clase] if len(f) == 0)
                            metricas[f"{clase}_max_track_id"] = max([tid for f in tracks[clase] for tid in f.keys()], default=-1)
                            # duración de tracks y reapariciones
                            duraciones = {}
                            for i, f in enumerate(tracks[clase]):
                                for tid in f.keys():
                                    duraciones.setdefault(tid, []).append(i)
                            metricas[f"{clase}_duracion_media"] = np.mean([len(v) for v in duraciones.values()]) if duraciones else 0
                            metricas[f"{clase}_reapariciones"] = sum(np.sum(np.diff(v) > 1) for v in duraciones.values())
                            # cambios de equipo
                            historial = {}
                            cambios_equipo = 0
                            for f in tracks[clase]:
                                for tid, d in f.items():
                                    t = d["team"]
                                    if tid in historial and historial[tid] != t:
                                        cambios_equipo += 1
                                    historial[tid] = t
                            metricas[f"{clase}_cambios_equipo"] = cambios_equipo

                            bbox_areas = [d["bbox_size"] for f in tracks[clase] for d in f.values() if "bbox_size" in d]
                            metricas[f"{clase}_bbox_area_media"] = np.mean(bbox_areas) if bbox_areas else 0
                            metricas[f"{clase}_bbox_area_std"] = np.std(bbox_areas) if bbox_areas else 0

                            # Color de camisetas
                            shirt_colors = [d["shirt_color"] for f in tracks[clase] for d in f.values() if "shirt_color" in d]
                            if shirt_colors:
                                shirt_colors = np.array(shirt_colors)
                                metricas[f"{clase}_shirt_color_promedio"] = shirt_colors.mean(axis=0).tolist()
                                metricas[f"{clase}_shirt_color_std"] = shirt_colors.std(axis=0).tolist()
                            else:
                                metricas[f"{clase}_shirt_color_promedio"] = [0, 0, 0]
                                metricas[f"{clase}_shirt_color_std"] = [0, 0, 0]
                        
                        metricas.update({ "prueba_id": prueba_id, "conf": conf, "track_thresh": tt, "match_thresh": mt, "mcf": mcf })
                        resultados.append(metricas)
                        drawer = Drawer(colors = { "player": (0, 255, 0), "goalkeeper": (0, 255, 255), "referee": (255, 0, 0), "ball": (0, 0, 255) })
                        drawer.draw_tracks(tracks, VIDEO, output)
                        
                        if prueba_id == 0:
                            with open("./tracks_prueba.json", "w", encoding="utf-8") as f:
                                json.dump(convert_keys_to_int(tracks), f, indent=4, ensure_ascii=False, sort_keys=True)      # Ordena las claves alfabéticamente
                    except Exception:
                        metricas = {}
                        metricas.update({ "prueba_id": prueba_id, "conf": conf, "track_thresh": tt, "match_thresh": mt, "mcf": mcf })
                        resultados.append(metricas)
                        raise


    try:
        with open("./tracks.json", "w", encoding="utf-8") as f:
            json.dump(convert_keys_to_int(tracks), f, indent=4, ensure_ascii=False, sort_keys=True)      # Ordena las claves alfabéticamente
    except:
        pass

    pd.DataFrame(resultados).to_csv("./resultados_metricas.csv", index=False)
