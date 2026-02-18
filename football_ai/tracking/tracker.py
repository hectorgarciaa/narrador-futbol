import numpy as np
import supervision as sv

from football_ai.detection import Detector
from football_ai.identification import TeamDetector
from football_ai.tracking.byte_tracker import ByteTrack

class Tracker:
    def __init__(self, model_path, conf, tracker_conf, team_colors, ball_min_conf=0.01):
        self.model = Detector(model_path, conf)
        self.team_detector = TeamDetector(team_colors)
        self.tracker = ByteTrack(
            tracker_conf["track_thresh"],
            tracker_conf["track_buffer"],
            tracker_conf["match_thresh"],
            tracker_conf["frame_rate"],
            tracker_conf["minimum_consecutive_frames"],
            team_penalty=tracker_conf.get("team_penalty", 1000),
            team_switch_threshold=tracker_conf.get("team_switch_threshold", 5),
        )
        self.ball_min_conf = ball_min_conf
    
    def get_tracks(self, video, show_kmeans=False):
        model_detections = self.model.detect(video)
        tracks = {"player": [], "goalkeeper": [], "referee": [], "ball": [] }
        fallback_id_counter = 0

        for n_frame, detections in enumerate(model_detections):
            detections_sv = sv.Detections.from_ultralytics(detections)

            teams_of_detected_objects = self.team_detector.detect_teams(detections, show_kmeans)
            teams_labels = [dicc["team"] for dicc in teams_of_detected_objects]

            # Save original index to align after tracker filtering
            detections_sv.data['original_idx'] = np.arange(len(detections_sv))

            tracks_detection = self.tracker.update_with_detections(detections_sv, teams_labels)
            
            for key in tracks.keys():
                tracks[key].append({})

            has_tracked_ball = False
            for object_detected in tracks_detection:
                bbox, _, confidence, _, tracker_id, data = object_detected
                class_name = data["class_name"]
                original_idx = int(data["original_idx"])

                if class_name == "ball":
                    has_tracked_ball = True

                tracks[class_name][n_frame][tracker_id] = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "team": teams_of_detected_objects[original_idx]["team"],
                    "distances": teams_of_detected_objects[original_idx]["distances"],
                    "shirt_color": teams_of_detected_objects[original_idx]["shirt_color"],
                    "bbox_size": teams_of_detected_objects[original_idx]["bbox_size"],
                }

            # If the tracker did not activate the ball, use raw detections (without tracking)
            if not has_tracked_ball and detections.boxes is not None and len(detections.boxes) > 0:
                boxes = detections.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                
                for bbox, score, cid in zip(xyxy, conf, cls):
                    class_name = detections.names[cid]
                    if class_name != "ball" or score < self.ball_min_conf:
                        continue
                    x1, y1, x2, y2 = bbox.tolist()
                    tracks["ball"][n_frame][f"fallback_{fallback_id_counter}"] = {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(score),
                        "team": None,
                        "distances": None,
                        "shirt_color": None,
                        "bbox_size": float((x2 - x1) * (y2 - y1)),
                    }
                    fallback_id_counter += 1
        
        return tracks
