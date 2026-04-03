import numpy as np
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
        shirt_detector_conf={},
    ):
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
        self.candidate_classes = [c_name for c_name in ["player", "goalkeeper", "referee"]]
        self.updated = {class_name: False for class_name in ["player", "referee"]}
        self.class_samples = {class_name: [] for class_name in ["player", "referee"]}
        self.outfield_team_distance_stats = {}
        self.goalkeeper_outlier_iqr_factor = 1.5
        self.goalkeeper_sideline_min_distance_m = 3.0

        self.contador = 0

    def detect_teams(self, frame_detections, field_positions, yolo_class_labels, field_width_m, sideline_band_distance_m, show_plot=False):
        shirts = []
        teams_of_detected_objects = []
        x_positions = self._get_x_positions(field_positions, yolo_class_labels)
        
        for object_detected, field_position in zip(frame_detections, field_positions):
            class_name = object_detected.names[object_detected.boxes.cls.item()]
            sample_bucket = "player" if class_name == "goalkeeper" else class_name
            bbox_size = self._extract_bbox_size(object_detected)
            if class_name in self.candidate_classes:
                shirt, shirt_color = self._get_shirt_color(object_detected)
                if show_plot:
                    shirts.append(shirt)

                if shirt_color is not None and sample_bucket in self.updated and self.updated[sample_bucket]:
                    new_possible_class = self._check_possible_new_class(shirt_color, field_position, field_width_m, sideline_band_distance_m, x_positions)
                    sample_bucket = new_possible_class if new_possible_class is not None else sample_bucket
                
                if shirt_color is not None and sample_bucket in self.updated and not self.updated[sample_bucket]:
                    self.class_samples[sample_bucket].append(shirt_color)

                if (
                    sample_bucket in self.updated
                    and len(self.class_samples[sample_bucket]) > self.min_samples[sample_bucket]
                    and not self.updated[sample_bucket]
                ):
                    self.updated[sample_bucket] = self._update_class_colors(sample_bucket)
                
                class_name, team, distances = self._reassign_class(shirt_color, sample_bucket)

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
    
    def _check_possible_new_class(self, shirt_color, field_position, field_width_m, sideline_band_distance_m, x_positions):
        if not self.updated["player"] or len(self.outfield_team_distance_stats) < 2:
            return None
        
        distances = {team: np.linalg.norm(shirt_color - team_color) for team, team_color in self.team_colors.items() if team != "referee"}
        if any([distance < self.outfield_team_distance_stats[team]["upper_bound"] for team, distance in distances.items() ]):
            return "player"
        
        else:
            field_position = self._field_position_to_tuple(field_position)

            is_in_middle = False
            if len(x_positions) >= 8:
                x_positions.sort()
                left_x_bound = float(x_positions[3])
                right_x_bound = float(x_positions[-4])
                is_in_middle = left_x_bound <= float(field_position[0]) <= right_x_bound

            lower_sideline_limit = float(sideline_band_distance_m)
            upper_sideline_limit = float(field_width_m) - float(sideline_band_distance_m)
            is_near_sidelines = float(field_position[1]) <= lower_sideline_limit or float(field_position[1]) >= upper_sideline_limit
            
            if is_in_middle or is_near_sidelines:
                return "referee"
            
            else:
                left_x_bound = float(x_positions[2])
                right_x_bound = float(x_positions[-3])

                outside_x_bounds = field_position[0] < left_x_bound or field_position[0] > right_x_bound
                far_from_sidelines = field_position[1] > lower_sideline_limit and field_position[1] < upper_sideline_limit
                if outside_x_bounds and far_from_sidelines:
                    return "goalkeeper"            
        return None
    
    def _get_x_positions(self, field_positions, yolo_class_labels):
        x_positions = []
        for det_idx, class_name in enumerate(yolo_class_labels):
            if class_name not in {"player", "goalkeeper"}:
                continue

            field_position_aux = (field_positions[det_idx] if det_idx < len(field_positions) else None)
            field_position_aux = self._field_position_to_tuple(field_position_aux)
            x_positions.append(float(field_position_aux[0]))
        return x_positions

    def _update_team_colors(self, class_samples):
        km = KMeans(n_clusters=self.n_teams, init="k-means++", n_init=5, random_state=0)
        km.fit(class_samples)
        km, class_samples = self._check_clusters_sizes(km, class_samples)
        if km is None:
            return False

        cluster_entries = []
        for cluster_id in range(self.n_teams):
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
        upper_bound = float(q3 + (self.goalkeeper_outlier_iqr_factor * iqr))
        return {
            "sample_count": int(distances.size),
            "median": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "upper_bound": upper_bound,
        }

    def _reassign_class(self, shirt_color, class_name):
        team, distances = self._assign_team(shirt_color)
        if team == "referee":
            return "referee", None, distances
        elif team in self.team_colors:
            if class_name == "goalkeeper":
                return "goalkeeper", team, distances
            return "player", team, distances
        else:
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

    def _goalkeeper_outlier_debug_from_distances(self, distances):
        if not isinstance(distances, dict):
            return None
        team_names = [team_name for team_name in self.team_colors.keys() if team_name != "referee"]
        if len(team_names) < 2:
            return None
        per_team = {}
        all_outlier = True
        for team_name in team_names:
            distance_value = distances.get(team_name)
            stats = self.outfield_team_distance_stats.get(team_name)
            is_outlier = False
            if distance_value is not None and isinstance(stats, dict):
                is_outlier = float(distance_value) > float(stats.get("upper_bound", np.inf))
            else:
                all_outlier = False
            per_team[str(team_name)] = {
                "distance": (None if distance_value is None else float(distance_value)),
                "upper_bound": (
                    None
                    if not isinstance(stats, dict)
                    else float(stats.get("upper_bound", np.inf))
                ),
                "median": (
                    None
                    if not isinstance(stats, dict)
                    else float(stats.get("median", np.nan))
                ),
                "iqr": (
                    None
                    if not isinstance(stats, dict)
                    else float(stats.get("iqr", np.nan))
                ),
                "is_outlier": bool(is_outlier),
            }
            all_outlier = all_outlier and bool(is_outlier)
        return {
            "applied": bool(per_team),
            "all_outfield_teams_outlier": bool(all_outlier),
            "teams": per_team,
            "iqr_factor": float(self.goalkeeper_outlier_iqr_factor),
        }

    def apply_referee_relabel_gate(
        self,
        teams_of_detected_objects,
        yolo_class_labels,
        field_positions,
        field_width_m,
        sideline_band_distance_m,
    ):
        if not teams_of_detected_objects or field_width_m is None:
            return None

        x_positions = []
        for det_idx, class_name in enumerate(yolo_class_labels):
            normalized = str(class_name or "").strip().lower()
            if normalized not in {"player", "goalkeeper"}:
                continue
            field_position = (
                field_positions[det_idx]
                if det_idx < len(field_positions)
                else None
            )
            field_position = self._field_position_to_tuple(field_position)
            if field_position is None:
                continue
            x_positions.append(float(field_position[0]))

        if len(x_positions) < 8:
            return None

        x_positions.sort()
        left_x_bound = float(x_positions[3])
        right_x_bound = float(x_positions[-4])
        lower_sideline_limit = float(sideline_band_distance_m)
        upper_sideline_limit = float(field_width_m) - float(sideline_band_distance_m)

        for det_idx, detection_info in enumerate(teams_of_detected_objects):
            raw_class_name = str(
                yolo_class_labels[det_idx] if det_idx < len(yolo_class_labels) else ""
            ).strip().lower()
            relabeled_class_name = str(detection_info.get("class") or "").strip().lower()
            if raw_class_name not in {"player", "goalkeeper"}:
                continue
            if relabeled_class_name != "referee":
                continue

            field_position = (
                field_positions[det_idx]
                if det_idx < len(field_positions)
                else None
            )
            field_position = self._field_position_to_tuple(field_position)
            allowed = False
            reason = "missing_field_position"
            if field_position is not None:
                x_pos, y_pos = field_position
                within_x_bounds = left_x_bound <= float(x_pos) <= right_x_bound
                within_sideline_band = (
                    float(y_pos) <= lower_sideline_limit
                    or float(y_pos) >= upper_sideline_limit
                )
                allowed = bool(within_x_bounds or within_sideline_band)
                if allowed and within_x_bounds and within_sideline_band:
                    reason = "inside_referee_x_bounds_and_sideline_band"
                elif allowed and within_x_bounds:
                    reason = "inside_referee_x_bounds"
                elif allowed and within_sideline_band:
                    reason = "inside_referee_sideline_band"
                elif not within_x_bounds:
                    reason = "outside_referee_x_bounds"
                else:
                    reason = "outside_referee_sideline_band"

            detection_info["referee_reassign_gate"] = {
                "applied": True,
                "allowed": bool(allowed),
                "reason": reason,
                "left_x_bound": left_x_bound,
                "right_x_bound": right_x_bound,
                "sideline_band_distance_m": float(sideline_band_distance_m),
                "field_position_m": list(field_position) if field_position is not None else None,
            }
            if allowed:
                continue

            detection_info["class"] = raw_class_name
            detection_info["team"] = self._nearest_outfield_team_from_distances(
                detection_info.get("distances")
            )

        return {
            "left_x_bound": left_x_bound,
            "right_x_bound": right_x_bound,
            "sideline_band_distance_m": float(sideline_band_distance_m),
        }

    def apply_goalkeeper_relabel_gate(
        self,
        teams_of_detected_objects,
        current_class_labels,
        field_positions,
        field_width_m,
    ):
        if not teams_of_detected_objects or field_width_m is None:
            return None

        x_positions = []
        for det_idx, class_name in enumerate(current_class_labels):
            normalized = str(class_name or "").strip().lower()
            if normalized not in {"player", "goalkeeper"}:
                continue
            field_position = (
                field_positions[det_idx]
                if det_idx < len(field_positions)
                else None
            )
            field_position = self._field_position_to_tuple(field_position)
            if field_position is None:
                continue
            x_positions.append(float(field_position[0]))

        if len(x_positions) < 6:
            return None

        x_positions.sort()
        left_x_bound = float(x_positions[2])
        right_x_bound = float(x_positions[-3])
        lower_sideline_limit = float(self.goalkeeper_sideline_min_distance_m)
        upper_sideline_limit = float(field_width_m) - float(self.goalkeeper_sideline_min_distance_m)

        for det_idx, detection_info in enumerate(teams_of_detected_objects):
            relabeled_class_name = str(detection_info.get("class") or "").strip().lower()
            if relabeled_class_name not in {"player", "referee"}:
                continue

            field_position = (
                field_positions[det_idx]
                if det_idx < len(field_positions)
                else None
            )
            field_position = self._field_position_to_tuple(field_position)
            outlier_debug = self._goalkeeper_outlier_debug_from_distances(
                detection_info.get("distances")
            )
            color_outlier = bool(
                isinstance(outlier_debug, dict)
                and outlier_debug.get("all_outfield_teams_outlier", False)
            )
            allowed = False
            reason = "missing_field_position"
            if field_position is not None:
                x_pos, y_pos = field_position
                outside_x_bounds = (
                    float(x_pos) < left_x_bound or float(x_pos) > right_x_bound
                )
                far_from_sidelines = (
                    float(y_pos) > lower_sideline_limit
                    and float(y_pos) < upper_sideline_limit
                )
                allowed = bool(color_outlier and outside_x_bounds and far_from_sidelines)
                if allowed:
                    reason = "outside_goalkeeper_x_bounds_and_color_outlier"
                elif not color_outlier:
                    reason = "outfield_team_color_not_outlier"
                elif not outside_x_bounds:
                    reason = "inside_goalkeeper_x_bounds"
                else:
                    reason = "inside_goalkeeper_sideline_exclusion_band"

            detection_info["goalkeeper_reassign_gate"] = {
                "applied": True,
                "allowed": bool(allowed),
                "reason": reason,
                "left_x_bound": left_x_bound,
                "right_x_bound": right_x_bound,
                "sideline_min_distance_m": float(self.goalkeeper_sideline_min_distance_m),
                "field_position_m": list(field_position) if field_position is not None else None,
                "color_outlier_debug": outlier_debug,
            }
            if not allowed:
                continue

            detection_info["class"] = "goalkeeper"
            detection_info["team"] = self._nearest_outfield_team_from_distances(
                detection_info.get("distances")
            )

        return {
            "left_x_bound": left_x_bound,
            "right_x_bound": right_x_bound,
            "sideline_min_distance_m": float(self.goalkeeper_sideline_min_distance_m),
        }
