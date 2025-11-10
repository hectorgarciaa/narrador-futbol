import numpy as np
import supervision as sv

from detector import Detector
from teamDetector import TeamDetector
from drawer import Drawer
from byteTracker import ByteTrack

class Tracker:
    def __init__(self, model_path, conf, traker_conf, team_colors):
        self.model = Detector(model_path, conf)
        self.teamDetector = TeamDetector(team_colors)
        # self.tracker = sv.ByteTrack(traker_conf["track_thresh"], traker_conf["track_buffer"], traker_conf["match_thresh"], traker_conf["frame_rate"])
        self.tracker = ByteTrack(traker_conf["track_thresh"], traker_conf["track_buffer"], traker_conf["match_thresh"], traker_conf["frame_rate"])
    
    def get_tracks(self, video, showKMeans):
        model_detections = self.model.detect(video)
        tracks = {"player": [], "goalkeeper": [], "referee": [], "ball": [] }
        for n_frame, detections in enumerate(model_detections):
            detections_sv = sv.Detections.from_ultralytics(detections)

            teams_of_detected_objects = self.teamDetector.detectTeams(detections, showKMeans)
            teams_labels = [dicc["team"] for dicc in teams_of_detected_objects]
            
            tracks_detection = self.tracker.update_with_detections(detections_sv, teams_labels)
            # tracks_detection = self.tracker.update_with_detections(detections_sv)
            
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
                }
        
        return tracks
    

if __name__ == "__main__":
    MODEL_PATH = "../../../models/finetuning/v11/yolov11x/weights/best.pt"
    VIDEO = "../../../data/partidoPrueba/08fd33_4.mp4"
    OUTPUT = "../../../output/pruebaTracker/08fd33_4_7.mp4"
    TEAM_COLORS = {
        "Real Madrid": np.array([0, 0, 245]),      # blanco (HSV)
        "Wolfsburgo":  np.array([70, 150, 150])    # verde
    }

    CONF = 0.02
    
    TRACKER_CONF = {
        "track_thresh": CONF,
        "track_buffer": 90,
        "match_thresh": 1,
        "frame_rate": 25
    }

    SHOWKMEANS = False
    tracker = Tracker(MODEL_PATH, CONF, TRACKER_CONF, TEAM_COLORS)
    tracks = tracker.get_tracks(VIDEO, SHOWKMEANS)
    
    drawer = Drawer(colors = { "player": (0, 255, 0), "goalkeeper": (0, 255, 255), "referee": (255, 0, 0), "ball": (0, 0, 255) })
    drawer.draw_tracks(tracks, VIDEO, OUTPUT)
