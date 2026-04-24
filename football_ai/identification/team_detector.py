import numpy as np
from collections import deque
from scipy.stats import trim_mean
from sklearn.cluster import KMeans

from football_ai.identification.shirt_detector import ShirtDetector
from football_ai.evaluation.cluster_visualizer import visualize_shirt_clusters

class TeamDetector:
    def __init__(
        self,
        team_colors=None,
        min_samples={"player": 60, "referee": 15},
        min_size_cluster=4,
        min_conf={"player": 0.8, "referee": 0.7},
        max_samples_per_class=500,
        referee_bootstrap_margin=10,
        shirt_detector_conf={},
    ):
        self.min_samples = min_samples
        self.min_size_cluster = min_size_cluster
        self.min_conf = min_conf
        self.max_samples_per_class = max_samples_per_class
        self.referee_bootstrap_margin = float(referee_bootstrap_margin)
        self.with_ref = team_colors is not None
        if team_colors is None:
            self.team_colors = {team: None for team in ["Equipo 1", "Equipo 2"]}
        else:
            self.team_colors = {team: np.asarray(color, dtype=np.float32) for team, color in team_colors.items()}
        self.team_colors["referee"] = None

        self.N_TEAMS = 2
        self.shirt_detector = ShirtDetector(**shirt_detector_conf)
        self.candidate_classes = [ "player", "goalkeeper", "referee" ]
        self.updated = { "player": False, "referee": False }
        self.class_samples = { "player": deque(maxlen=self.max_samples_per_class), "referee": deque(maxlen=self.max_samples_per_class) }
        self.outfield_team_distance_stats = {}
        
        self.GOALKEEPER_OUTLIER_IQR_FACTOR = 1.5

        self.n_frame = -1

    def detect_teams(self, frame_detections, field_positions, yolo_class_labels, field_width_m, sideline_band_distance_m, show_plot=False):
        self.n_frame += 1
        shirts = []
        teams_of_detected_objects = []
        x_positions = self._get_k_positions(field_positions, yolo_class_labels, 0)
        y_positions = self._get_k_positions(field_positions, yolo_class_labels, 1)
        x_positions.sort()
        y_positions.sort()

        shirts_by_detection_idx = {}
        colors_by_detection_idx = {}
        color_candidate_indices = []
        color_candidate_shirts = []

        for det_idx, object_detected in enumerate(frame_detections):
            class_name = object_detected.names[object_detected.boxes.cls.item()]
            if class_name not in self.candidate_classes:
                continue

            shirt = self._extract_shirt_crop(object_detected)
            shirts_by_detection_idx[det_idx] = shirt
            if show_plot and shirt is not None:
                shirts.append(shirt)
            if shirt is None:
                continue

            color_candidate_indices.append(det_idx)
            color_candidate_shirts.append(shirt)

        if color_candidate_shirts:
            shirt_colors = self.shirt_detector.get_color_kmeans_batch(color_candidate_shirts)
            for det_idx, shirt_color in zip(color_candidate_indices, shirt_colors):
                colors_by_detection_idx[det_idx] = shirt_color
        
        for det_idx, (object_detected, field_position) in enumerate(zip(frame_detections, field_positions)):
            field_position = self._field_position_to_tuple(field_position)
            class_name = object_detected.names[object_detected.boxes.cls.item()]
            bbox_size = self._extract_bbox_size(object_detected)
            if class_name in self.candidate_classes:
                shirt = shirts_by_detection_idx.get(det_idx)
                shirt_color = colors_by_detection_idx.get(det_idx)
                
                sample_bucket = class_name
                effective_class = class_name
                if field_position is not None:
                    if shirt_color is not None and sample_bucket in self.updated:
                        if self.updated[sample_bucket]:
                            sample_bucket, effective_class = self._check_possible_new_class(shirt_color, field_position, field_width_m, sideline_band_distance_m, x_positions, y_positions)
                            sample_bucket = sample_bucket if sample_bucket is not None else class_name
                            effective_class = effective_class if effective_class is not None else class_name
                        
                        if sample_bucket in self.updated:
                            conf = float(object_detected.boxes.conf.item()) if object_detected.boxes.conf is not None else 0
                            if conf >= self.min_conf[sample_bucket] or (not self.updated[sample_bucket] and self.n_frame > 200):
                                self.class_samples[sample_bucket].append(shirt_color)

                                if len(self.class_samples[sample_bucket]) > self.min_samples[sample_bucket] and self.n_frame % 5 == 0:
                                    self.updated[sample_bucket] = self._update_class_colors(sample_bucket)

                class_name_aux, team, distances = self._reassign_class(shirt_color, effective_class, x_positions, y_positions, field_position, field_width_m, sideline_band_distance_m)
                class_name = class_name_aux if field_position is not None else class_name

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
    
    
    def _get_k_positions(self, field_positions, yolo_class_labels, k):
        x_positions = []
        for det_idx, class_name in enumerate(yolo_class_labels):
            if class_name not in {"player", "goalkeeper"}:
                continue

            field_position_aux = (field_positions[det_idx] if det_idx < len(field_positions) else None)
            field_position_aux = self._field_position_to_tuple(field_position_aux)
            if field_position_aux is None:
                continue
            x_positions.append(float(field_position_aux[k]))
        return x_positions

    @staticmethod
    def _field_position_to_tuple(field_position):
        if field_position is None:
            return None
        try:
            x_pos, y_pos = field_position
            if not np.isfinite(x_pos) or not np.isfinite(y_pos):
                return None
            return float(x_pos), float(y_pos)
        except (TypeError, ValueError):
            return None

    def _extract_shirt_crop(self, object_detected):
        x1, y1, x2, y2 = map(int, object_detected.boxes.xyxy[0])
        player_pixels = object_detected.orig_img[y1:y2, x1:x2]
        if player_pixels.size > 0:
            h = player_pixels.shape[0]
            shirt = player_pixels[:int(0.5*h), :]
            return shirt
        return None
    
    @staticmethod
    def _extract_bbox_size(object_detected):
        try:
            x1, y1, x2, y2 = map(int, object_detected.boxes.xyxy[0])
        except Exception:
            return 0.0
        return max(0, x2 - x1) * max(0, y2 - y1)
    
    def _serialize_color(self, shirt_color):
        if shirt_color is None:
            return None
        return [float(channel) for channel in np.asarray(shirt_color, dtype=np.float32).reshape(-1).tolist()]
    
    def _update_class_colors(self, class_name):
        if class_name == "player":
            return self._update_team_colors(np.asarray(self.class_samples[class_name], dtype=np.float32))
        elif class_name == "referee":
            return self._update_referee_colors(np.asarray(self.class_samples[class_name], dtype=np.float32))
    
    def _check_possible_new_class(self, shirt_color, field_position, field_width_m, sideline_band_distance_m, x_positions, y_positions):
        if not self.updated["player"] or len(self.outfield_team_distance_stats) < 2:
            return None, None
        
        distances = {team: np.linalg.norm(shirt_color - team_color) for team, team_color in self.team_colors.items() if team_color is not None}
        if any([distance < self.outfield_team_distance_stats[team]["upper_bound"] for team, distance in distances.items() if team != "referee"]):
            return "player", "player"
        
        else:
            can_be_goalkeeper = self._check_goalkeeper_pos(x_positions, field_position, field_width_m, sideline_band_distance_m)
            if can_be_goalkeeper:
                return "goalkeeper", "goalkeeper"
            
            else:
                if self.updated["referee"]:
                    team_idx = np.argmin([d for d in distances.values()])
                    team_name = list(distances.keys())[team_idx]
                    if team_name != "referee":
                        return "player", "player"
                    
                    else:
                        _, can_be_middle_ref = self._check_ref_pos(x_positions, y_positions, field_position, field_width_m, sideline_band_distance_m)
                        if can_be_middle_ref:
                            return "referee", "referee"
                        else:
                            return None, None
                else:
                    _, can_be_middle_ref = self._check_ref_pos(x_positions, y_positions, field_position, field_width_m, sideline_band_distance_m)
                    if can_be_middle_ref:
                        margins = [d - self.outfield_team_distance_stats[team]["upper_bound"] for team, d in distances.items() if team != "referee"]
                        if min(margins) > self.referee_bootstrap_margin:
                            return "referee", "player"
                        else:
                            return None, None
                    else:
                        return None, None

    def _update_team_colors(self, class_samples):
        km = KMeans(n_clusters=self.N_TEAMS, init="k-means++", n_init=3, random_state=0)
        km.fit(class_samples)
        km, class_samples = self._check_clusters_sizes(km, class_samples)
        if km is None:
            return False

        cluster_entries = []
        for cluster_id in range(self.N_TEAMS):
            samples_cluster = class_samples[km.labels_ == cluster_id]
            cluster_entries.append(
                {
                    "center": np.median(samples_cluster, axis=0).astype(np.float32),
                    "samples": samples_cluster.astype(np.float32),
                }
            )
        cluster_entries = sorted(cluster_entries, key=lambda entry: self._center_hue_key(entry["center"]))
        self._update_reference_team_colors(cluster_entries)
        return True
    
    def _update_referee_colors(self, class_samples): 
        self.team_colors["referee"] = np.array([trim_mean(class_samples[:, i], proportiontocut=0.15) for i in range(class_samples.shape[1])]).astype(np.float32)
        return True

    def _check_clusters_sizes(self, km, samples):
        sizes = np.array([int(np.sum(km.labels_ == idx)) for idx in range(self.N_TEAMS)], dtype=int)
        i = 0
        large_cluster_samples = samples
        while np.min(sizes) < self.min_size_cluster:
            largest_cluster_id = np.argmax(sizes)
            large_cluster_samples = large_cluster_samples[km.labels_ == largest_cluster_id]
            if len(large_cluster_samples) < max(self.min_samples["player"], self.N_TEAMS * self.min_size_cluster):
                return None, None
            km.fit(large_cluster_samples)
            sizes = np.array([int(np.sum(km.labels_ == idx)) for idx in range(self.N_TEAMS)], dtype=int)
            
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

    def _update_reference_team_colors(self, cluster_entries):
        centers = np.asarray([entry["center"] for entry in cluster_entries], dtype=np.float32)
        if self.with_ref:
            team_names = [team for team in self.team_colors.keys() if team != "referee"]

            dist = np.array([
                [np.linalg.norm(center - self.team_colors[team_name]) for center in centers]
                for team_name in team_names])

            cost_a = dist[0, 0] + dist[1, 1]
            cost_b = dist[0, 1] + dist[1, 0]

            assignment = [0, 1] if cost_a <= cost_b else [1, 0]

            for idx, team in enumerate(team_names):
                cluster_entry = cluster_entries[assignment[idx]]
                self.team_colors[team] = cluster_entry["center"]
                self.outfield_team_distance_stats[team] = self._build_outfield_distance_stats(
                    cluster_entry["samples"],
                    cluster_entry["center"],
                )

        else:
            for idx, cluster_entry in enumerate(cluster_entries, start=1):
                team_name = f"Equipo {idx}"
                self.team_colors[team_name] = cluster_entry["center"].astype(np.float32)
                self.outfield_team_distance_stats[team_name] = self._build_outfield_distance_stats(
                    cluster_entry["samples"],
                    cluster_entry["center"],
                )

    def _build_outfield_distance_stats(self, samples, center):
        samples = np.asarray(samples, dtype=np.float32)
        center = np.asarray(center, dtype=np.float32).reshape(-1)
        if samples.ndim != 2 or samples.shape[0] <= 0 or center.size < 3:
            return None
        distances = np.linalg.norm(samples - center[: samples.shape[1]], axis=1)
        distances = np.asarray(distances, dtype=np.float32).reshape(-1)
        if distances.size <= 0:
            return None
        q1, median, q3 = np.quantile(distances, [0.25, 0.5, 0.75])
        iqr = max(0.0, float(q3 - q1))
        upper_bound = float(q3 + (self.GOALKEEPER_OUTLIER_IQR_FACTOR * iqr))
        return {
            "sample_count": int(distances.size),
            "median": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "upper_bound": upper_bound,
        }

    def _reassign_class(self, shirt_color, class_name, x_positions, y_positions, field_position, field_width_m, sideline_band_distance_m):
        team, distances = self._assign_team(shirt_color)
        if team == "referee":
            if class_name == "referee":
                return "referee", None, distances
            if field_position is None:
                return class_name, self._nearest_outfield_team_from_distances(distances), distances
            
            can_be_ref, _ = self._check_ref_pos(x_positions, y_positions, field_position, field_width_m, sideline_band_distance_m)
            if can_be_ref:
                return "referee", None, distances
            can_be_gk = self._check_goalkeeper_pos(x_positions, field_position, field_width_m, sideline_band_distance_m)
            if can_be_gk:
                return "goalkeeper", None, distances
            else:
                return class_name, self._nearest_outfield_team_from_distances(distances), distances   
        elif team in self.team_colors:
            can_be_gk = self._check_goalkeeper_pos(x_positions, field_position, field_width_m, sideline_band_distance_m)
            if class_name == "goalkeeper" and (field_position is None or can_be_gk):
                return "goalkeeper", None, distances
            
            if self.updated["player"] or len(self.outfield_team_distance_stats) >= 2:
                if any([distance < self.outfield_team_distance_stats[team]["upper_bound"] for team, distance in distances.items() if team != "referee" ]):
                    return "player", team, distances
                elif field_position is None or can_be_gk:
                    return "goalkeeper", None, distances
        return class_name, team, distances
        
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
    
    def _check_ref_pos(self, x_positions, y_positions, field_position, field_width_m, sideline_band_distance_m):
        if field_position is None:
            return False

        is_in_middle = False
        is_middle_ref = False
        if len(x_positions) >= 8:
            left_x_bound = float(x_positions[3])
            right_x_bound = float(x_positions[-4])
            up_y_bound = float(y_positions[3])
            down_y_bound = float(y_positions[-4])
            is_in_middle = left_x_bound <= float(field_position[0]) <= right_x_bound
            is_middle_ref = is_in_middle and up_y_bound <= float(field_position[1]) <= down_y_bound

        lower_sideline_limit = float(sideline_band_distance_m)
        upper_sideline_limit = float(field_width_m) - float(sideline_band_distance_m)
        is_near_sidelines = float(field_position[1]) <= lower_sideline_limit or float(field_position[1]) >= upper_sideline_limit
        
        return is_near_sidelines or is_in_middle, is_middle_ref
    
    def _check_goalkeeper_pos(self, x_positions, field_position, field_width_m, sideline_band_distance_m):
        if field_position is None:
            return False
        if len(x_positions) < 6:
            return False
        left_x_bound = float(x_positions[2])
        right_x_bound = float(x_positions[-3])

        outside_x_bounds = field_position[0] < left_x_bound or field_position[0] > right_x_bound
        lower_sideline_limit = float(sideline_band_distance_m)
        upper_sideline_limit = float(field_width_m) - float(sideline_band_distance_m)
        far_from_sidelines = field_position[1] > lower_sideline_limit and field_position[1] < upper_sideline_limit
        
        return outside_x_bounds and far_from_sidelines

    @staticmethod
    def _nearest_outfield_team_from_distances(distances):
        if not isinstance(distances, dict):
            return None
        valid = [
            (str(team_name), float(distance_value))
            for team_name, distance_value in distances.items()
            if team_name != "referee" and distance_value is not None
        ]
        if not valid:
            return None
        return min(valid, key=lambda item: item[1])[0]
