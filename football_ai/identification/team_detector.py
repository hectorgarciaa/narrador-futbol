import logging

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from football_ai.identification.shirt_detector import ShirtDetector

logger = logging.getLogger(__name__)


class TeamDetector:
    def __init__(
        self,
        team_colors_refs=None,
        confirmation_threshold=3,
        color_tolerance=25,
        assignment_mode="reference",
        auto_bootstrap_frames=1,
        auto_bootstrap_min_samples=12,
        auto_num_teams=2,
        auto_min_cluster_samples=4,
        auto_team_name_prefix="Equipo",
        auto_label_by_reference_colors=False,
        team_candidate_classes=None,
        shirt_detector_kwargs=None,
        team_colors=None,
        candidate_classes=None,
        n_teams=None,
        with_ref=None,
        min_samples=None,
        min_size_cluster=None,
    ):
        # Compatibilidad con la API antigua del detector.
        if team_colors_refs is None:
            team_colors_refs = team_colors
        if team_colors_refs is None:
            team_colors_refs = {}
        if team_candidate_classes is None and candidate_classes is not None:
            team_candidate_classes = candidate_classes
        if n_teams is not None:
            try:
                auto_num_teams = max(2, int(n_teams))
            except (TypeError, ValueError):
                pass
        if min_samples is not None:
            try:
                auto_bootstrap_min_samples = max(2, int(min_samples))
            except (TypeError, ValueError):
                pass
        if min_size_cluster is not None:
            try:
                auto_min_cluster_samples = max(1, int(min_size_cluster))
            except (TypeError, ValueError):
                pass
        if with_ref is False and str(assignment_mode or "reference").strip().lower() == "reference":
            assignment_mode = "auto-bootstrap"

        self.team_colors_refs = {
            str(team): np.asarray(color, dtype=np.float32)
            for team, color in dict(team_colors_refs).items()
        }
        self.reference_team_colors_refs = {
            team: color.copy() for team, color in self.team_colors_refs.items()
        }
        self.team_colors = {
            team: color.copy() for team, color in self.team_colors_refs.items()
        }
        self.color_samples = {team: {} for team in self.team_colors_refs.keys()}

        self.confirmation_threshold = max(1, int(confirmation_threshold))
        self.color_tolerance = float(color_tolerance)
        self.confirmed_teams = set()

        normalized_mode = str(assignment_mode or "reference").strip().lower()
        if normalized_mode not in {"reference", "auto-bootstrap"}:
            logger.warning(
                "assignment_mode '%s' no soportado. Se usará 'reference'.",
                assignment_mode,
            )
            normalized_mode = "reference"
        self.assignment_mode = normalized_mode
        self.auto_bootstrap_frames = max(1, int(auto_bootstrap_frames))
        self.auto_bootstrap_min_samples = max(2, int(auto_bootstrap_min_samples))
        self.auto_num_teams = max(2, int(auto_num_teams))
        self.auto_min_cluster_samples = max(1, int(auto_min_cluster_samples))
        self.auto_team_name_prefix = str(auto_team_name_prefix or "Equipo").strip() or "Equipo"
        self.auto_label_by_reference_colors = bool(auto_label_by_reference_colors)
        self.team_candidate_classes = set(
            team_candidate_classes or ["player", "goalkeeper"]
        )
        self.bootstrap_samples = []
        self.current_frame_idx = -1
        self.bootstrap_ready = self.assignment_mode == "reference"

        if shirt_detector_kwargs is None:
            shirt_detector_kwargs = {}
        self.shirt_detector = ShirtDetector(**shirt_detector_kwargs)

    @staticmethod
    def _center_hue_key(lab_color):
        lab_color = np.asarray(lab_color, dtype=np.float32).reshape(-1)
        if lab_color.size < 3:
            return (0.0, 0.0, 0.0)
        a_channel = float(lab_color[1]) - 128.0
        b_channel = float(lab_color[2]) - 128.0
        hue = (np.degrees(np.arctan2(b_channel, a_channel)) + 360.0) % 360.0
        chroma = float((a_channel**2 + b_channel**2) ** 0.5)
        lightness = float(lab_color[0])
        return (hue, -chroma, -lightness)

    def _build_auto_team_refs(self, centers):
        centers = np.asarray(centers, dtype=np.float32)
        if centers.ndim != 2 or centers.shape[0] <= 0:
            return {}

        if (
            self.auto_label_by_reference_colors
            and len(self.reference_team_colors_refs) == centers.shape[0]
        ):
            ref_names = list(self.reference_team_colors_refs.keys())
            ref_colors = np.asarray(
                [self.reference_team_colors_refs[name] for name in ref_names],
                dtype=np.float32,
            )
            cost_matrix = np.linalg.norm(
                centers[:, None, :] - ref_colors[None, :, :],
                axis=2,
            )
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return {
                str(ref_names[col_pos]): centers[row_pos].astype(np.float32)
                for row_pos, col_pos in zip(row_ind.tolist(), col_ind.tolist())
            }

        sorted_centers = sorted(centers, key=self._center_hue_key)
        team_refs = {}
        for idx, center in enumerate(sorted_centers, start=1):
            team_name = f"{self.auto_team_name_prefix} {idx}".strip()
            team_refs[team_name] = center.astype(np.float32)
        return team_refs

    def _activate_auto_bootstrap_if_possible(self, force=False):
        if self.assignment_mode != "auto-bootstrap" or self.bootstrap_ready:
            return

        sample_count = len(self.bootstrap_samples)
        if sample_count < self.auto_num_teams:
            if force:
                logger.warning(
                    "Auto bootstrap sin muestras suficientes: %d (< %d). Se usará modo reference.",
                    sample_count,
                    self.auto_num_teams,
                )
                self.assignment_mode = "reference"
                self.bootstrap_ready = True
            return

        if not force and sample_count < self.auto_bootstrap_min_samples:
            return

        def fit_kmeans(input_samples):
            km = KMeans(
                n_clusters=self.auto_num_teams,
                init="k-means++",
                n_init=10,
                random_state=0,
            )
            km.fit(input_samples)
            return km

        def cluster_sizes(labels):
            return np.array(
                [int(np.sum(labels == idx)) for idx in range(self.auto_num_teams)],
                dtype=int,
            )

        samples = np.asarray(self.bootstrap_samples, dtype=np.float32)
        working_samples = samples
        km = fit_kmeans(working_samples)
        labels = km.labels_
        sizes = cluster_sizes(labels)

        if (
            self.auto_num_teams == 2
            and sizes.size == 2
            and int(np.min(sizes)) < self.auto_min_cluster_samples
        ):
            small_size = int(np.min(sizes))
            large_cluster_idx = int(np.argmax(sizes))
            large_cluster_samples = working_samples[labels == large_cluster_idx]
            if large_cluster_samples.shape[0] >= self.auto_num_teams:
                logger.info(
                    "Auto bootstrap: cluster pequeno detectado (%d muestras). "
                    "Se re-clusteriza sobre el cluster mayor (%d muestras).",
                    small_size,
                    int(large_cluster_samples.shape[0]),
                )
                working_samples = large_cluster_samples
                km = fit_kmeans(working_samples)
                labels = km.labels_
                sizes = cluster_sizes(labels)

        if sizes.size > 0 and int(np.min(sizes)) < self.auto_min_cluster_samples:
            logger.info(
                "Auto bootstrap pendiente: clusters con pocas muestras %s "
                "(minimo requerido=%d). Se esperan mas detecciones.",
                sizes.tolist(),
                self.auto_min_cluster_samples,
            )
            return

        cluster_refs = []
        for cluster_idx in range(self.auto_num_teams):
            cluster_points = working_samples[labels == cluster_idx]
            if cluster_points.size == 0:
                cluster_ref = np.asarray(km.cluster_centers_[cluster_idx], dtype=np.float32)
            else:
                cluster_ref = np.median(cluster_points, axis=0).astype(np.float32)
            cluster_refs.append(cluster_ref)

        auto_refs = self._build_auto_team_refs(cluster_refs)
        if len(auto_refs) < 2:
            logger.warning(
                "Auto bootstrap no pudo construir suficientes equipos (%d). Se usara modo reference.",
                len(auto_refs),
            )
            self.assignment_mode = "reference"
            self.bootstrap_ready = True
            return

        self.team_colors_refs = auto_refs
        self.team_colors = {team: color.copy() for team, color in auto_refs.items()}
        self.color_samples = {team: {} for team in auto_refs.keys()}
        self.confirmed_teams = set(auto_refs.keys())
        self.bootstrap_ready = True
        logger.info(
            "Auto bootstrap de equipos completado: frames_bootstrap=%d, samples=%d, teams=%s",
            self.current_frame_idx + 1,
            sample_count,
            sorted(self.team_colors.keys()),
        )

    def _is_team_candidate_class(self, class_name):
        return class_name in self.team_candidate_classes

    def update_team_colors(self, shirt_color):
        if self.assignment_mode != "reference" or not self.team_colors:
            return
        shirt_color = np.asarray(shirt_color, dtype=np.float32)
        if len(self.confirmed_teams) == len(self.team_colors):
            return

        distances = {
            team: np.linalg.norm(shirt_color - color)
            if color is not None
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
        if shirt_color is None or not self.team_colors:
            return None, None
        shirt_color = np.asarray(shirt_color, dtype=np.float32)
        distances = {
            team: float(np.linalg.norm(shirt_color - color))
            for team, color in self.team_colors.items()
        }
        return min(distances, key=distances.get), distances

    def _extract_shirt_color(self, frame, shirts, bbox, show_plot=False):
        x1, y1, x2, y2 = map(int, bbox[0])
        player_pixels = frame[y1:y2, x1:x2]
        bbox_area = float((x2 - x1) * (y2 - y1))
        if player_pixels.size > 0:
            h = player_pixels.shape[0]
            shirt = player_pixels[: int(0.5 * h), :]
            if show_plot:
                shirts.append(shirt)
            shirt_color = self.shirt_detector.get_color_kmeans(shirt)
            return shirt_color, bbox_area
        return None, bbox_area

    def _assign_team_from_shirt_color(self, shirt_color):
        if shirt_color is None:
            return None, None
        shirt_color = np.asarray(shirt_color, dtype=np.float32)
        if self.assignment_mode == "auto-bootstrap" and not self.bootstrap_ready:
            return None, None
        self.update_team_colors(shirt_color)
        return self.assign_team(shirt_color)

    def detect_teams(self, frame_detections, show_plot=False):
        self.current_frame_idx += 1
        shirts = []
        teams_of_detected_objects = []
        detections_buffer = []

        for object_detected in frame_detections:
            bbox = object_detected.boxes.xyxy
            class_name = object_detected.names[int(object_detected.boxes.cls.item())]
            is_candidate = self._is_team_candidate_class(class_name)
            shirt_color, bbox_size = (
                self._extract_shirt_color(
                    frame_detections.orig_img,
                    shirts,
                    bbox,
                    show_plot=show_plot,
                )
                if is_candidate
                else (
                    None,
                    float(
                        max(0, int(bbox[0][2]) - int(bbox[0][0]))
                        * max(0, int(bbox[0][3]) - int(bbox[0][1]))
                    ),
                )
            )
            detections_buffer.append(
                {
                    "class": class_name,
                    "bbox_size": bbox_size,
                    "shirt_color": shirt_color,
                    "is_candidate": is_candidate,
                }
            )
            if (
                self.assignment_mode == "auto-bootstrap"
                and not self.bootstrap_ready
                and is_candidate
                and shirt_color is not None
            ):
                self.bootstrap_samples.append(np.asarray(shirt_color, dtype=np.float32))

        if self.assignment_mode == "auto-bootstrap" and not self.bootstrap_ready:
            force_bootstrap = (self.current_frame_idx + 1) >= self.auto_bootstrap_frames
            self._activate_auto_bootstrap_if_possible(force=force_bootstrap)

        for det_info in detections_buffer:
            team = None
            distances = None
            if det_info["is_candidate"]:
                team, distances = self._assign_team_from_shirt_color(det_info["shirt_color"])
            serializable_shirt_color = None
            if det_info["shirt_color"] is not None:
                serializable_shirt_color = [
                    float(channel)
                    for channel in np.asarray(det_info["shirt_color"], dtype=np.float32).tolist()
                ]

            teams_of_detected_objects.append(
                {
                    "class": det_info["class"],
                    "team": team,
                    "distances": distances,
                    "shirt_color": serializable_shirt_color,
                    "bbox_size": det_info["bbox_size"],
                }
            )

        if show_plot and shirts:
            from football_ai.evaluation.cluster_visualizer import visualize_shirt_clusters

            visualize_shirt_clusters(shirts)

        return teams_of_detected_objects
