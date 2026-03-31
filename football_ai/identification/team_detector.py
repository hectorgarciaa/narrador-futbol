import numpy as np
from sklearn.cluster import KMeans

from football_ai.identification.shirt_detector import ShirtDetector
from football_ai.evaluation.cluster_visualizer import visualize_shirt_clusters

class TeamDetector:
    @staticmethod
    def _normalize_class_name(class_name):
        token = str(class_name or "").strip().lower()
        aliases = {
            "player": "player",
            "players": "player",
            "goalkeeper": "goalkeeper",
            "gk": "goalkeeper",
            "keeper": "goalkeeper",
            "referee": "referee",
            "ref": "referee",
            "refs": "referee",
            "ball": "ball",
            "balls": "ball",
        }
        return aliases.get(token, token)

    def __init__(self, n_teams=2, with_ref=False, team_colors=None, min_samples=60, min_size_cluster=4,
                 candidate_classes=["player"]):
        self.n_teams = n_teams
        self.samples = []
        self.with_ref = with_ref
        self.min_samples = min_samples
        self.min_size_cluster = min_size_cluster
        self.shirt_detector = ShirtDetector()
        self.candidate_classes = [
            self._normalize_class_name(class_name)
            for class_name in candidate_classes
        ]
        self.team_colors = { team: np.asarray(team_colors[team], dtype=np.float32) for team in team_colors.keys() } if with_ref else {}

        self.updated = False

    def detect_teams(self, frame_detections, show_plot=False):
        shirts = []
        teams_of_detected_objects = []
        
        for object_detected in frame_detections:
            class_name = self._normalize_class_name(
                object_detected.names[object_detected.boxes.cls.item()]
            )
            bbox_size = self._extract_bbox_size(object_detected)
            if class_name in self.candidate_classes:
                shirt, shirt_color = self.get_shirt_color(object_detected)
                if show_plot:
                    shirts.append(shirt)
                if shirt_color is not None:
                    self.samples.append(shirt_color)

                if len(self.samples) > self.min_samples and not self.updated:
                    self.updated = self.update_team_colors()
                
                team, distances = self.assign_team(shirt_color)
                teams_of_detected_objects.append(
                    {
                        "class": class_name,
                        "team": team,
                        "shirt_color": self.serialize_color(shirt_color),
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

    def serialize_color(self, shirt_color):
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

    
    def get_shirt_color(self, object_detected):
        x1, y1, x2, y2 = map(int, object_detected.boxes.xyxy[0])
        player_pixels = object_detected.orig_img[y1:y2, x1:x2]
        if player_pixels.size > 0:
            h = player_pixels.shape[0]
            shirt = player_pixels[:int(0.5*h), :]
            shirt_color = self.shirt_detector.get_color_kmeans(shirt)
            return shirt, shirt_color
        return None, None
    
    def update_team_colors(self):
        samples = np.asarray(self.samples, dtype=np.float32)

        km = KMeans(n_clusters=self.n_teams, init="k-means++", n_init=5, random_state=0)
        km.fit(samples)
        km, samples = self.check_clusters_sizes(km, samples)
        if km is None:
            return False

        cluster_refs = []
        for cluster_id in range(self.n_teams):
            samples_cluster = samples[km.labels_ == cluster_id]
            cluster_refs.append(np.median(samples_cluster, axis=0))
        cluster_refs = np.asarray(cluster_refs, dtype=np.float32)
        sorted_centers = sorted(cluster_refs, key=self.center_hue_key)
        self.update_refs_colors(sorted_centers)
        return True

    def check_clusters_sizes(self, km, samples):
        sizes = np.array([int(np.sum(km.labels_ == idx)) for idx in range(self.n_teams)], dtype=int)
        i = 0
        large_cluster_samples = samples
        while np.min(sizes) < self.min_size_cluster:
            largest_cluster_id = np.argmax(sizes)
            large_cluster_samples = large_cluster_samples[km.labels_ == largest_cluster_id]
            if len(large_cluster_samples) < max(self.min_samples, self.n_teams * self.min_size_cluster):
                return None, None
            km.fit(large_cluster_samples)
            sizes = np.array([int(np.sum(km.labels_ == idx)) for idx in range(self.n_teams)], dtype=int)
            
            i += 1
            if i > 5:
                return None, None
        
        return km, large_cluster_samples
    
    def center_hue_key(self, lab_color):
        lab_color = np.asarray(lab_color, dtype=np.float32).reshape(-1)
        if lab_color.size < 3:
            return (0.0, 0.0, 0.0)
        a_channel = float(lab_color[1]) - 128.0
        b_channel = float(lab_color[2]) - 128.0
        hue = (np.degrees(np.arctan2(b_channel, a_channel)) + 360.0) % 360.0
        chroma = float((a_channel ** 2 + b_channel ** 2) ** 0.5)
        lightness = float(lab_color[0])
        return (hue, -chroma, -lightness)

    def update_refs_colors(self, centers):
        if self.with_ref:
            team_names = list(self.team_colors.keys())
            team_ref_colors = [self.team_colors[team] for team in team_names]

            dist = np.array([
                [np.linalg.norm(center - team_color) for center in centers]
                for team_color in team_ref_colors])

            cost_a = dist[0, 0] + dist[1, 1]
            cost_b = dist[0, 1] + dist[1, 0]

            assignment = [0, 1] if cost_a <= cost_b else [1, 0]

            new_team_colors = {team: centers[assignment[idx]] for idx, team in enumerate(team_names)}
            self.team_colors = new_team_colors

        else:
            self.team_colors = {}
            for idx, center in enumerate(centers, start=1):
                team_name = f"Equipo {idx}".strip()
                self.team_colors[team_name] = center.astype(np.float32)

    def assign_team(self, shirt_color):
        if shirt_color is None:
            return None, None
        if self.updated or self.with_ref:
            distances = {team: float(np.linalg.norm(shirt_color - team_color)) for team, team_color in self.team_colors.items()}
            team_idx = np.argmin([d for d in distances.values()])
            team_name = list(self.team_colors.keys())[team_idx]
            return team_name, distances
        return None, None
