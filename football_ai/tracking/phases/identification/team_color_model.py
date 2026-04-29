from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import trim_mean
from sklearn.cluster import KMeans

from .team_detector_utils import nearest_outfield_team, serialize_color


class TeamColorModel:
    def __init__(
        self,
        team_colors=None,
        min_samples={"player": 60, "referee": 15},
        min_size_cluster=4,
        min_conf={"player": 0.8, "referee": 0.7},
        max_samples_per_class=500,
        referee_bootstrap_margin=10,
    ):
        self.min_samples = {"player": 60, "referee": 15, **dict(min_samples or {})}
        self.min_conf = {"player": 0.8, "referee": 0.7, **dict(min_conf or {})}
        self.min_size_cluster = int(min_size_cluster)
        self.max_samples_per_class = int(max_samples_per_class)
        self.referee_bootstrap_margin = float(referee_bootstrap_margin)
        self.goalkeeper_outlier_iqr_factor = 1.5
        self.n_teams = 2
        self.with_reference_colors = team_colors is not None

        base_colors = (
            {"Equipo 1": None, "Equipo 2": None}
            if team_colors is None
            else {team: np.asarray(color, dtype=np.float32) for team, color in team_colors.items()}
        )
        base_colors["referee"] = None
        self.team_colors = base_colors
        self.updated = {"player": False, "referee": False}
        self.class_samples = {
            "player": deque(maxlen=self.max_samples_per_class),
            "referee": deque(maxlen=self.max_samples_per_class),
        }
        self._last_cluster_update_frame = {
            "player": -1,
            "referee": -1,
        }
        self.outfield_team_distance_stats = {}

    def trace_snapshot(self, cluster_events):
        return {
            "events": list(cluster_events),
            "updated_flags": dict(self.updated),
            "class_sample_counts": {
                class_name: len(samples)
                for class_name, samples in self.class_samples.items()
            },
            "team_colors": {
                str(team_name): serialize_color(color)
                for team_name, color in self.team_colors.items()
            },
            "outfield_team_distance_stats": {
                str(team_name): (
                    {key: float(value) for key, value in stats.items()}
                    if stats is not None
                    else None
                )
                for team_name, stats in self.outfield_team_distance_stats.items()
            },
        }

    def decide_sample_class(self, shirt_color, can_be_goalkeeper, can_be_middle_ref):
        if not self.updated["player"] or len(self.outfield_team_distance_stats) < 2:
            return None, None, "outfield_clusters_not_ready"

        distances = self._distances(shirt_color)
        if self.matches_outfield_cluster(distances):
            return "player", "player", "inside_outfield_cluster"
        if can_be_goalkeeper:
            return "goalkeeper", "goalkeeper", "goalkeeper_position_gate"

        if self.updated["referee"]:
            nearest_team = self._nearest_team(distances)
            if nearest_team == "referee" and can_be_middle_ref:
                return "referee", "referee", "referee_color_and_position_gate"
            if nearest_team != "referee":
                return "player", "player", "nearest_color_is_outfield_team"
            return None, None, "referee_color_without_valid_position"

        if not can_be_middle_ref:
            return None, None, "no_position_gate_matched"

        margins = [
            float(distance) - self.outfield_team_distance_stats[team_name]["upper_bound"]
            for team_name, distance in distances.items()
            if team_name != "referee" and distance is not None
        ]
        if margins and min(margins) > self.referee_bootstrap_margin:
            return "referee", "player", "bootstrap_referee_margin_passed"
        return None, None, "bootstrap_referee_margin_too_small"

    def maybe_add_sample(self, sample_bucket, shirt_color, confidence, frame_index):
        if sample_bucket not in self.updated:
            return None, None

        accepted = confidence >= self.min_conf[sample_bucket] or (
            not self.updated[sample_bucket] and frame_index > 200
        )
        trace = {
            "bucket": sample_bucket,
            "confidence": float(confidence),
            "accepted": bool(accepted),
        }
        if not accepted:
            return trace, None

        self.class_samples[sample_bucket].append(shirt_color)
        trace["sample_count_after_append"] = len(self.class_samples[sample_bucket])
        if len(self.class_samples[sample_bucket]) <= self.min_samples[sample_bucket] or frame_index % 5 != 0:
            return trace, None
        if self._last_cluster_update_frame[sample_bucket] == frame_index:
            trace["cluster_update_skipped"] = "already_updated_this_frame"
            return trace, None

        updated, cluster_event = (
            self._update_player_colors()
            if sample_bucket == "player"
            else self._update_referee_color()
        )
        self._last_cluster_update_frame[sample_bucket] = frame_index
        self.updated[sample_bucket] = updated
        if cluster_event is not None:
            trace["cluster_update"] = {
                "bucket": sample_bucket,
                "success": bool(updated),
                "event": cluster_event,
            }
        return trace, cluster_event

    def assign_team(self, shirt_color):
        distances = self._distances(shirt_color)
        return self._nearest_team(distances), distances

    def matches_outfield_cluster(self, distances):
        if not distances:
            return False
        for team_name, distance in distances.items():
            if team_name == "referee" or distance is None:
                continue
            stats = self.outfield_team_distance_stats.get(team_name)
            if stats is not None and float(distance) < stats["upper_bound"]:
                return True
        return False

    @staticmethod
    def nearest_outfield_team(distances):
        return nearest_outfield_team(distances)

    def _distances(self, shirt_color):
        if shirt_color is None:
            return None
        return {
            team_name: (
                None if color is None else float(np.linalg.norm(shirt_color - color))
            )
            for team_name, color in self.team_colors.items()
        }

    @staticmethod
    def _nearest_team(distances):
        if not distances:
            return None
        valid = [(team_name, distance) for team_name, distance in distances.items() if distance is not None]
        return min(valid, key=lambda item: item[1])[0] if valid else None

    def _update_player_colors(self):
        samples = np.asarray(self.class_samples["player"], dtype=np.float32)
        minimum = max(self.min_samples["player"], self.n_teams)
        if len(samples) < minimum:
            return False, {
                "event_type": "player_cluster_update",
                "success": False,
                "reason": "insufficient_samples",
                "sample_count": len(samples),
            }

        km = KMeans(n_clusters=self.n_teams, init="k-means++", n_init=3, random_state=0)
        km.fit(samples)
        km, samples = self._shrink_small_clusters(km, samples)
        if km is None:
            return False, {
                "event_type": "player_cluster_update",
                "success": False,
                "reason": "clusters_too_small",
                "sample_count": len(samples),
            }

        cluster_entries = []
        for cluster_id in range(self.n_teams):
            cluster_samples = samples[km.labels_ == cluster_id]
            cluster_entries.append({
                "center": np.median(cluster_samples, axis=0).astype(np.float32),
                "samples": cluster_samples.astype(np.float32),
            })
        cluster_entries.sort(key=lambda entry: self._center_hue_key(entry["center"]))
        assignment = self._apply_cluster_assignment(cluster_entries)
        return True, {
            "event_type": "player_cluster_update",
            "success": True,
            "sample_count": len(samples),
            "clusters": [
                {"center": serialize_color(entry["center"]), "sample_count": len(entry["samples"])}
                for entry in cluster_entries
            ],
            "assignment": assignment,
        }

    def _update_referee_color(self):
        samples = np.asarray(self.class_samples["referee"], dtype=np.float32)
        if len(samples) == 0:
            return False, {
                "event_type": "referee_color_update",
                "success": False,
                "reason": "insufficient_samples",
                "sample_count": 0,
            }

        self.team_colors["referee"] = np.array(
            [trim_mean(samples[:, channel], proportiontocut=0.15) for channel in range(samples.shape[1])],
            dtype=np.float32,
        )
        return True, {
            "event_type": "referee_color_update",
            "success": True,
            "sample_count": len(samples),
            "center": serialize_color(self.team_colors["referee"]),
        }

    def _shrink_small_clusters(self, km, samples):
        for _ in range(6):
            sizes = np.array([np.sum(km.labels_ == cluster_id) for cluster_id in range(self.n_teams)])
            if int(np.min(sizes)) >= self.min_size_cluster:
                return km, samples
            samples = samples[km.labels_ == int(np.argmax(sizes))]
            if len(samples) < max(self.min_samples["player"], self.n_teams * self.min_size_cluster):
                return None, samples
            km.fit(samples)
        return None, samples

    @staticmethod
    def _center_hue_key(lab_color):
        color = np.asarray(lab_color, dtype=np.float32).reshape(-1)
        if color.size < 3:
            return 0.0, 0.0, 0.0
        a_channel = float(color[1]) - 128.0
        b_channel = float(color[2]) - 128.0
        hue = (np.degrees(np.arctan2(b_channel, a_channel)) + 360.0) % 360.0
        chroma = float(np.hypot(a_channel, b_channel))
        return hue, -chroma, -float(color[0])

    def _apply_cluster_assignment(self, cluster_entries):
        team_names = [team_name for team_name in self.team_colors if team_name != "referee"]
        centers = np.asarray([entry["center"] for entry in cluster_entries], dtype=np.float32)

        if self.with_reference_colors:
            reference_distances = np.array([
                [np.linalg.norm(center - self.team_colors[team_name]) for center in centers]
                for team_name in team_names
            ])
            assignment = [0, 1] if reference_distances[0, 0] + reference_distances[1, 1] <= reference_distances[0, 1] + reference_distances[1, 0] else [1, 0]
        else:
            assignment = list(range(len(team_names)))

        trace = {"with_reference_colors": self.with_reference_colors, "teams": {}}
        for team_index, team_name in enumerate(team_names):
            cluster_index = assignment[team_index]
            if not self.with_reference_colors:
                team_name = f"Equipo {team_index + 1}"
            cluster_entry = cluster_entries[cluster_index]
            self.team_colors[team_name] = cluster_entry["center"].astype(np.float32)
            self.outfield_team_distance_stats[team_name] = self._distance_stats(
                cluster_entry["samples"],
                cluster_entry["center"],
            )
            trace["teams"][str(team_name)] = {
                "cluster_index": int(cluster_index),
                "center": serialize_color(cluster_entry["center"]),
                "distance_stats": self.outfield_team_distance_stats[team_name],
            }
        return trace

    def _distance_stats(self, samples, center):
        distances = np.linalg.norm(np.asarray(samples, dtype=np.float32) - np.asarray(center, dtype=np.float32), axis=1)
        if distances.size == 0:
            return None
        q1, median, q3 = np.quantile(distances, [0.25, 0.5, 0.75])
        iqr = max(0.0, float(q3 - q1))
        return {
            "sample_count": int(distances.size),
            "median": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "upper_bound": float(q3 + self.goalkeeper_outlier_iqr_factor * iqr),
        }
