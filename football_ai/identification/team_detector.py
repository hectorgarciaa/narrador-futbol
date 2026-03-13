import cv2
import logging
import numpy as np
from sklearn.cluster import KMeans

from football_ai.identification.shirt_detector import ShirtDetector

logger = logging.getLogger(__name__)


AUTO_COLOR_PALETTE_RGB = {
    "Rojo": np.array([220, 20, 60], dtype=np.float32),
    "Azul": np.array([0, 102, 255], dtype=np.float32),
    "Verde": np.array([0, 170, 0], dtype=np.float32),
    "Amarillo": np.array([255, 221, 0], dtype=np.float32),
    "Naranja": np.array([255, 140, 0], dtype=np.float32),
    "Morado": np.array([128, 0, 128], dtype=np.float32),
    "Rosa": np.array([255, 105, 180], dtype=np.float32),
    "Cian": np.array([0, 180, 200], dtype=np.float32),
    "Blanco": np.array([245, 245, 245], dtype=np.float32),
    "Negro": np.array([15, 15, 15], dtype=np.float32),
    "Gris": np.array([128, 128, 128], dtype=np.float32),
}


class TeamDetector:
    def __init__(
        self,
        team_colors_refs,
        confirmation_threshold=3,
        color_tolerance=25,
        assignment_mode="reference",
        auto_bootstrap_frames=1,
        auto_bootstrap_min_samples=12,
        auto_num_teams=2,
        auto_team_name_prefix="Equipo",
        team_candidate_classes=None,
        shirt_detector_kwargs=None,
    ):
        self.team_colors_refs = team_colors_refs
        self.team_colors = {
            team: np.asarray(team_colors_refs[team], dtype=np.float32)
            for team in team_colors_refs.keys()
        }

        self.color_samples = {team: {} for team in team_colors_refs.keys()}

        self.confirmation_threshold = confirmation_threshold
        self.color_tolerance = color_tolerance
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
        self.auto_team_name_prefix = str(auto_team_name_prefix or "Equipo").strip() or "Equipo"
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
    def _lab_to_rgb_color(lab_color):
        clipped = np.clip(np.asarray(lab_color, dtype=np.float32), 0.0, 255.0)
        lab_pixel = np.array([[clipped]], dtype=np.uint8)
        rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)[0, 0]
        return rgb_pixel.astype(np.float32)

    @staticmethod
    def _center_hue_key(lab_color):
        lab_color = np.asarray(lab_color, dtype=np.float32).reshape(-1)
        if lab_color.size < 3:
            return (0.0, 0.0, 0.0)
        a_channel = float(lab_color[1]) - 128.0
        b_channel = float(lab_color[2]) - 128.0
        hue = (np.degrees(np.arctan2(b_channel, a_channel)) + 360.0) % 360.0
        chroma = float((a_channel ** 2 + b_channel ** 2) ** 0.5)
        lightness = float(lab_color[0])
        return (hue, -chroma, -lightness)

    def _closest_basic_color_name(self, lab_color):
        rgb_color = self._lab_to_rgb_color(lab_color)
        best_name = "Color"
        best_distance = None
        for color_name, reference_rgb in AUTO_COLOR_PALETTE_RGB.items():
            distance = float(np.linalg.norm(rgb_color - reference_rgb))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_name = color_name
        return best_name

    def _build_auto_team_refs(self, centers):
        centers = np.asarray(centers, dtype=np.float32)
        if centers.ndim != 2 or centers.shape[0] <= 0:
            return {}
        sorted_centers = sorted(centers, key=self._center_hue_key)

        team_refs = {}
        used_names = set()
        for idx, center in enumerate(sorted_centers, start=1):
            color_name = self._closest_basic_color_name(center)
            team_name = f"{self.auto_team_name_prefix} {color_name}".strip()
            if team_name in used_names:
                team_name = f"{team_name} {idx}"
            used_names.add(team_name)
            team_refs[team_name] = center.astype(np.float32)
        return team_refs

    def _activate_auto_bootstrap_if_possible(self, force=False):
        if self.assignment_mode != "auto-bootstrap" or self.bootstrap_ready:
            return

        sample_count = len(self.bootstrap_samples)
        if sample_count < self.auto_num_teams:
            if force:
                logger.warning(
                    "Auto bootstrap sin muestras suficientes: %d (< %d). "
                    "Se usará modo reference.",
                    sample_count,
                    self.auto_num_teams,
                )
                self.assignment_mode = "reference"
                self.bootstrap_ready = True
            return

        if not force and sample_count < self.auto_bootstrap_min_samples:
            return

        samples = np.asarray(self.bootstrap_samples, dtype=np.float32)
        km = KMeans(
            n_clusters=self.auto_num_teams,
            init="k-means++",
            n_init=10,
            random_state=0,
        )
        km.fit(samples)
        centers = km.cluster_centers_
        auto_refs = self._build_auto_team_refs(centers)
        if len(auto_refs) < 2:
            logger.warning(
                "Auto bootstrap no pudo construir suficientes equipos (%d). "
                "Se usará modo reference.",
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
        """Updates confirmed team colors based on accumulated samples."""
        if self.assignment_mode != "reference":
            return
        shirt_color = np.asarray(shirt_color, dtype=np.float32)
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
        if not self.team_colors:
            return None, None
        shirt_color = np.asarray(shirt_color, dtype=np.float32)
        distances = {
            team: float(np.linalg.norm(shirt_color - color))
            for team, color in self.team_colors.items()
        }
        return min(distances, key=distances.get), distances

    def _extract_shirt_color(self, frame, shirts, bbox):
        """Extract shirt color from upper bbox area."""
        x1, y1, x2, y2 = map(int, bbox[0])
        player_pixels = frame[y1:y2, x1:x2]
        bbox_area = float((x2 - x1) * (y2 - y1))
        if player_pixels.size > 0:
            h = player_pixels.shape[0]
            shirt = player_pixels[:int(0.5*h), :]
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
        """Detects the team for each detected object in the frame."""
        self.current_frame_idx += 1
        shirts = []
        teams_of_detected_objects = []
        detections_buffer = []

        for object_detected in frame_detections:
            bbox = object_detected.boxes.xyxy
            class_name = object_detected.names[object_detected.boxes.cls.item()]
            is_candidate = self._is_team_candidate_class(class_name)
            shirt_color, bbox_size = (
                self._extract_shirt_color(frame_detections.orig_img, shirts, bbox)
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

            teams_of_detected_objects.append({
                "class": det_info["class"],
                "team": team,
                "distances": distances,
                "shirt_color": serializable_shirt_color,
                "bbox_size": det_info["bbox_size"],
            })

        if show_plot and len(shirts) > 0:
            from football_ai.evaluation.cluster_visualizer import visualize_shirt_clusters
            visualize_shirt_clusters(shirts)

        return teams_of_detected_objects
