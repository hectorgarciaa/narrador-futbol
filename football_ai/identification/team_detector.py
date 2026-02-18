import cv2
import numpy as np

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
    
    def get_team_of_players(self, frame, shirts, bbox):
        """Detects a player's team from their shirt crop."""
        x1, y1, x2, y2 = map(int, bbox[0])
        player_pixels = frame[y1:y2, x1:x2]
        bbox_area = float((x2 - x1) * (y2 - y1))
        if player_pixels.size > 0:
            h = player_pixels.shape[0]
            shirt = player_pixels[:int(0.5*h), :]
            shirts.append(shirt)
            shirt_color = self.shirt_detector.get_color_kmeans(shirt)
            self.update_team_colors(shirt_color)
            team, distances = self.assign_team(shirt_color)
            return team, distances, shirt_color, bbox_area
        return None, None, None, bbox_area
    
    def detect_teams(self, frame_detections, show_plot=False):
        """Detects the team for each detected object in the frame."""
        shirts = []
        teams_of_detected_objects = []

        for object_detected in frame_detections:
            bbox = object_detected.boxes.xyxy
            class_name = object_detected.names[object_detected.boxes.cls.item()]
            team, distances, shirt_color, bbox_size = self.get_team_of_players(
                frame_detections.orig_img, shirts, bbox
            )
            teams_of_detected_objects.append({
                "class": class_name,
                "team": team,
                "distances": distances,
                "shirt_color": shirt_color,
                "bbox_size": bbox_size,
            })

        if show_plot and len(shirts) > 0:
            from football_ai.evaluation.cluster_visualizer import visualize_shirt_clusters
            visualize_shirt_clusters(shirts)

        return teams_of_detected_objects
