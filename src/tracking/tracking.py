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
    def __init__(self, model_path, conf, tracker_conf, team_colors):
        self.model = Detector(model_path, conf)
        self.teamDetector = TeamDetector(team_colors)
        self.tracker = ByteTrack(tracker_conf["track_thresh"], tracker_conf["track_buffer"], tracker_conf["match_thresh"], tracker_conf["frame_rate"], tracker_conf["minimum_consecutive_frames"])
    
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