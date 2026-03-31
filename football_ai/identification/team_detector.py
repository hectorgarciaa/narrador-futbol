import numpy as np
from scipy.stats import trim_mean
from sklearn.cluster import KMeans

from football_ai.identification.shirt_detector import ShirtDetector
from football_ai.evaluation.cluster_visualizer import visualize_shirt_clusters

class TeamDetector:
    def __init__(self, team_colors=None, min_samples={"player": 60, "referee": 15}, min_size_cluster=4, shirt_detector_conf={}):
        self.min_samples = min_samples
        self.min_size_cluster = min_size_cluster
        self.with_ref = team_colors is not None
        if team_colors is None:
            self.team_colors = {team: None for team in ["Equipo 1", "Equipo 2"]}
        else:
            self.team_colors = {team: np.asarray(color, dtype=np.float32) for team, color in team_colors.items()}
        self.team_colors["referee"] = None

        self.n_teams = 2
        self.shirt_detector = ShirtDetector(**shirt_detector_conf)
        self.candidate_classes = [c_name for c_name in ["player", "referee"]]
        self.updated = {class_name: False for class_name in self.candidate_classes}
        self.class_samples = {class_name: [] for class_name in self.candidate_classes}
        
        self.contador = 0

    def detect_teams(self, frame_detections, show_plot=False):
        shirts = []
        teams_of_detected_objects = []
        
        for object_detected in frame_detections:
            class_name = object_detected.names[object_detected.boxes.cls.item()]
            bbox_size = self._extract_bbox_size(object_detected)
            if class_name in self.candidate_classes:
                shirt, shirt_color = self._get_shirt_color(object_detected)
                if show_plot:
                    shirts.append(shirt)
                if shirt_color is not None and not self.updated[class_name]:
                    self.class_samples[class_name].append(shirt_color)

                if len(self.class_samples[class_name]) > self.min_samples[class_name] and not self.updated[class_name]:
                    self.updated[class_name] = self._update_class_colors(class_name)
                
                aux = class_name
                class_name, team, distances = self._reassign_class(object_detected, shirt_color)
                if aux != class_name:
                    self.contador += 1


                teams_of_detected_objects.append(
                    {
                        "class": class_name,
                        "team": team,
                        "shirt_color": self._serialize_color(shirt_color),
                        "distances": distances,
                        "bbox_size": float(bbox_size),
                    }
                )

            else:
                teams_of_detected_objects.append(
                    {
                        "class": class_name,
                        "team": None,
                        "shirt_color": None,
                        "distances": None,
                        "bbox_size": float(bbox_size),
                    }
                )
        
        if show_plot and len(shirts) > 0:
            visualize_shirt_clusters(shirts)

        return teams_of_detected_objects

    def _serialize_color(self, shirt_color):
        if shirt_color is None:
            return None
        return [float(channel) for channel in np.asarray(shirt_color, dtype=np.float32).reshape(-1).tolist()]

    @staticmethod
    def _extract_bbox_size(object_detected):
        try:
            x1, y1, x2, y2 = map(int, object_detected.boxes.xyxy[0])
        except Exception:
            return 0.0
        return max(0, x2 - x1) * max(0, y2 - y1)

    
    def _get_shirt_color(self, object_detected):
        x1, y1, x2, y2 = map(int, object_detected.boxes.xyxy[0])
        player_pixels = object_detected.orig_img[y1:y2, x1:x2]
        if player_pixels.size > 0:
            h = player_pixels.shape[0]
            shirt = player_pixels[:int(0.5*h), :]
            shirt_color = self.shirt_detector.get_color_kmeans(shirt)
            return shirt, shirt_color
        return None, None
    
    def _update_class_colors(self, class_name):
        if class_name == "player":
            return self._update_team_colors(np.asarray(self.class_samples[class_name], dtype=np.float32))
        elif class_name == "referee":
            return self._update_referee_colors(np.asarray(self.class_samples[class_name], dtype=np.float32))

    def _update_team_colors(self, class_samples):
        km = KMeans(n_clusters=self.n_teams, init="k-means++", n_init=5, random_state=0)
        km.fit(class_samples)
        km, class_samples = self._check_clusters_sizes(km, class_samples)
        if km is None:
            return False

        cluster_refs = []
        for cluster_id in range(self.n_teams):
            samples_cluster = class_samples[km.labels_ == cluster_id]
            cluster_refs.append(np.median(samples_cluster, axis=0))
        cluster_refs = np.asarray(cluster_refs, dtype=np.float32)
        sorted_centers = sorted(cluster_refs, key=self._center_hue_key)
        self._update_reference_team_colors(sorted_centers)
        return True
    
    def _update_referee_colors(self, class_samples): 
        self.team_colors["referee"] = np.array([trim_mean(class_samples[:, i], proportiontocut=0.15) for i in range(class_samples.shape[1])]).astype(np.float32)
        return True

    def _check_clusters_sizes(self, km, samples):
        sizes = np.array([int(np.sum(km.labels_ == idx)) for idx in range(self.n_teams)], dtype=int)
        i = 0
        large_cluster_samples = samples
        while np.min(sizes) < self.min_size_cluster:
            largest_cluster_id = np.argmax(sizes)
            large_cluster_samples = large_cluster_samples[km.labels_ == largest_cluster_id]
            if len(large_cluster_samples) < max(self.min_samples["player"], self.n_teams * self.min_size_cluster):
                return None, None
            km.fit(large_cluster_samples)
            sizes = np.array([int(np.sum(km.labels_ == idx)) for idx in range(self.n_teams)], dtype=int)
            
            i += 1
            if i > 5:
                return None, None
        
        return km, large_cluster_samples
    
    def _center_hue_key(self, lab_color):
        lab_color = np.asarray(lab_color, dtype=np.float32).reshape(-1)
        if lab_color.size < 3:
            return (0.0, 0.0, 0.0)
        a_channel = float(lab_color[1]) - 128.0
        b_channel = float(lab_color[2]) - 128.0
        hue = (np.degrees(np.arctan2(b_channel, a_channel)) + 360.0) % 360.0
        chroma = float((a_channel ** 2 + b_channel ** 2) ** 0.5)
        lightness = float(lab_color[0])
        return (hue, -chroma, -lightness)

    def _update_reference_team_colors(self, centers):
        if self.with_ref:
            team_names = [team for team in self.team_colors.keys() if team != "referee"]

            dist = np.array([
                [np.linalg.norm(center - self.team_colors[team_name]) for center in centers]
                for team_name in team_names])

            cost_a = dist[0, 0] + dist[1, 1]
            cost_b = dist[0, 1] + dist[1, 0]

            assignment = [0, 1] if cost_a <= cost_b else [1, 0]

            for idx, team in enumerate(team_names):
                self.team_colors[team] = centers[assignment[idx]]

        else:
            for idx, center in enumerate(centers, start=1):
                self.team_colors[f"Equipo {idx}"] = center.astype(np.float32)

    def _reassign_class(self, object_detected, shirt_color):
        team, distances = self._assign_team(shirt_color)
        if team == "referee":
            return "referee", None, distances
        elif team in self.team_colors:
            return "player", team, distances
        else:
            return object_detected.names[object_detected.boxes.cls.item()], team, distances
        
    
    def _assign_team(self, shirt_color):
        if shirt_color is None:
            return None, None
        distances = {team: None if team_color is None else float(np.linalg.norm(shirt_color - team_color))
                     for team, team_color in self.team_colors.items()}
        valid_distances = {team: dist for team, dist in distances.items() if dist is not None}

        if not valid_distances:
            return None, None

        team_idx = np.argmin([d for d in valid_distances.values()])
        team_name = list(valid_distances.keys())[team_idx]
        return team_name, distances
