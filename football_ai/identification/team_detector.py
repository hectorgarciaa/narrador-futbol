import cv2
import numpy as np
from time import perf_counter

from football_ai.identification.shirt_detector import ShirtDetector


class TeamDetector:
    def __init__(self, team_colors_refs, confirmation_threshold=3, color_tolerance=25):
        self.team_colors_refs = team_colors_refs
        self.team_colors = {team: team_colors_refs[team] for team in team_colors_refs.keys()}
        
        self.color_samples = {team: {} for team in team_colors_refs.keys()}

        self.confirmation_threshold = confirmation_threshold
        self.color_tolerance = color_tolerance
        self.confirmed_teams = set()

        self.shirt_detector = ShirtDetector()
        self.last_timing_detail = {}

    def update_team_colors(self, shirt_color):
        """Updates confirmed team colors based on accumulated samples."""
        if len(self.confirmed_teams) == len(self.team_colors):
            return
        
        distances = {
            team: np.linalg.norm(shirt_color - color) if color is not None
            else np.linalg.norm(shirt_color - self.team_colors_refs[team])
            for team, color in self.team_colors.items()
        }
        closest_team = min(distances, key=distances.get)

        if closest_team in self.confirmed_teams:
            return
        
        found_similar = False
        for shirt_colors_sample in self.color_samples[closest_team].keys():
            if np.linalg.norm(shirt_color - np.array(shirt_colors_sample)) < self.color_tolerance:
                self.color_samples[closest_team][shirt_colors_sample] += 1
                if self.color_samples[closest_team][shirt_colors_sample] >= self.confirmation_threshold:
                    self.confirmed_teams.add(closest_team)
                    self.team_colors[closest_team] = np.array(shirt_colors_sample)
                found_similar = True
                break

        if not found_similar:
            self.color_samples[closest_team][tuple(shirt_color)] = 1

    def assign_team(self, shirt_color):
        """Assigns the closest team by color distance."""
        distances = {team: np.linalg.norm(shirt_color - color) for team, color in self.team_colors.items()}
        return min(distances, key=distances.get), distances
    
    def get_team_of_players(self, frame, shirts, bbox, return_timing=False):
        """Detects a player's team from their shirt crop."""
        total_start = perf_counter()
        x1, y1, x2, y2 = map(int, bbox[0])
        crop_player_start = perf_counter()
        player_pixels = frame[y1:y2, x1:x2]
        crop_player_s = perf_counter() - crop_player_start
        bbox_area = float((x2 - x1) * (y2 - y1))
        if player_pixels.size > 0:
            crop_shirt_start = perf_counter()
            h = player_pixels.shape[0]
            shirt = player_pixels[:int(0.5*h), :]
            crop_shirt_s = perf_counter() - crop_shirt_start
            shirts.append(shirt)

            kmeans_start = perf_counter()
            shirt_color, kmeans_timing = self.shirt_detector.get_color_kmeans(
                shirt, return_timing=True
            )
            kmeans_total_s = perf_counter() - kmeans_start

            update_color_start = perf_counter()
            self.update_team_colors(shirt_color)
            update_color_s = perf_counter() - update_color_start

            assign_start = perf_counter()
            team, distances = self.assign_team(shirt_color)
            assign_s = perf_counter() - assign_start

            timing = {
                "crop_player_s": float(crop_player_s),
                "crop_shirt_s": float(crop_shirt_s),
                "kmeans_total_s": float(kmeans_total_s),
                "kmeans_fit_s": float(kmeans_timing.get("kmeans_fit_s", 0.0)),
                "kmeans_preprocess_color_convert_s": float(
                    kmeans_timing.get("kmeans_preprocess_color_convert_s", 0.0)
                ),
                "kmeans_preprocess_reshape_s": float(
                    kmeans_timing.get("kmeans_preprocess_reshape_s", 0.0)
                ),
                "kmeans_cluster_selection_s": float(
                    kmeans_timing.get("kmeans_cluster_selection_s", 0.0)
                ),
                "team_color_update_s": float(update_color_s),
                "team_assign_s": float(assign_s),
                "total_s": float(perf_counter() - total_start),
            }
            if return_timing:
                return team, distances, shirt_color, bbox_area, timing
            return team, distances, shirt_color, bbox_area

        timing = {
            "crop_player_s": float(crop_player_s),
            "crop_shirt_s": 0.0,
            "kmeans_total_s": 0.0,
            "kmeans_fit_s": 0.0,
            "kmeans_preprocess_color_convert_s": 0.0,
            "kmeans_preprocess_reshape_s": 0.0,
            "kmeans_cluster_selection_s": 0.0,
            "team_color_update_s": 0.0,
            "team_assign_s": 0.0,
            "total_s": float(perf_counter() - total_start),
        }
        if return_timing:
            return None, None, None, bbox_area, timing
        return None, None, None, bbox_area
    
    def detect_teams(self, frame_detections, show_plot=False):
        """Detects the team for each detected object in the frame."""
        total_start = perf_counter()
        loop_start = perf_counter()
        shirts = []
        teams_of_detected_objects = []
        total_crop_player_s = 0.0
        total_crop_shirt_s = 0.0
        total_kmeans_s = 0.0
        total_kmeans_fit_s = 0.0
        total_kmeans_preprocess_color_convert_s = 0.0
        total_kmeans_preprocess_reshape_s = 0.0
        total_kmeans_cluster_selection_s = 0.0
        total_team_color_update_s = 0.0
        total_team_assign_s = 0.0
        objects_with_valid_crop = 0

        for object_detected in frame_detections:
            bbox = object_detected.boxes.xyxy
            class_name = object_detected.names[object_detected.boxes.cls.item()]
            team, distances, shirt_color, bbox_size, timing = self.get_team_of_players(
                frame_detections.orig_img, shirts, bbox, return_timing=True
            )
            total_crop_player_s += float(timing["crop_player_s"])
            total_crop_shirt_s += float(timing["crop_shirt_s"])
            total_kmeans_s += float(timing["kmeans_total_s"])
            total_kmeans_fit_s += float(timing["kmeans_fit_s"])
            total_kmeans_preprocess_color_convert_s += float(
                timing["kmeans_preprocess_color_convert_s"]
            )
            total_kmeans_preprocess_reshape_s += float(
                timing["kmeans_preprocess_reshape_s"]
            )
            total_kmeans_cluster_selection_s += float(
                timing["kmeans_cluster_selection_s"]
            )
            total_team_color_update_s += float(timing["team_color_update_s"])
            total_team_assign_s += float(timing["team_assign_s"])
            if shirt_color is not None:
                objects_with_valid_crop += 1
            teams_of_detected_objects.append({
                "class": class_name,
                "team": team,
                "distances": distances,
                "shirt_color": shirt_color,
                "bbox_size": bbox_size,
            })
        loop_s = perf_counter() - loop_start

        plot_s = 0.0
        if show_plot and len(shirts) > 0:
            plot_start = perf_counter()
            from football_ai.evaluation.cluster_visualizer import visualize_shirt_clusters
            visualize_shirt_clusters(shirts)
            plot_s = perf_counter() - plot_start

        self.last_timing_detail = {
            "team_detection_total_s": float(perf_counter() - total_start),
            "team_detection_loop_s": float(loop_s),
            "team_plot_kmeans_s": float(plot_s),
            "team_crop_player_s": float(total_crop_player_s),
            "team_crop_shirt_s": float(total_crop_shirt_s),
            "team_kmeans_s": float(total_kmeans_s),
            "team_kmeans_fit_s": float(total_kmeans_fit_s),
            "team_kmeans_preprocess_color_convert_s": float(
                total_kmeans_preprocess_color_convert_s
            ),
            "team_kmeans_preprocess_reshape_s": float(
                total_kmeans_preprocess_reshape_s
            ),
            "team_kmeans_cluster_selection_s": float(
                total_kmeans_cluster_selection_s
            ),
            "team_clustering_s": float(total_kmeans_s),
            "team_color_update_s": float(total_team_color_update_s),
            "team_assign_s": float(total_team_assign_s),
            "team_objects_count": float(len(teams_of_detected_objects)),
            "team_objects_valid_crop_count": float(objects_with_valid_crop),
        }

        return teams_of_detected_objects
