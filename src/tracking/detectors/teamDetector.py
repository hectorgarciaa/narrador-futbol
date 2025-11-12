import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from .shirtDetector import ShirtDetector

def visualize_shirt_clusters(shirt_crops, max_players=20):
    """
    Muestra hasta 20 crops de camisetas en una cuadrícula.
    Para cada crop:
      - izquierda: crop original
      - derecha: crop coloreado por cluster (KMeans k=2)
    """
    # nos quedamos con los primeros N
    shirt_crops = shirt_crops[:max_players]
    n = len(shirt_crops)
    if n == 0:
        print("No hay crops para visualizar.")
        return

    # cada jugador tendrá 2 subplots → original y cluster
    # vamos a hacer una cuadrícula de 4 filas x 5 jugadores (o lo que toque)
    cols_players = 5
    rows_players = int(np.ceil(n / cols_players))

    # cada jugador ocupa 2 columnas visuales → 5 jugadores = 10 columnas
    fig, axes = plt.subplots(rows_players, cols_players*2, figsize=(cols_players*4, rows_players*3))
    if rows_players == 1:
        axes = np.expand_dims(axes, 0)  # para indexar igual

    for idx, crop in enumerate(shirt_crops):
        row = idx // cols_players
        col = idx % cols_players

        # aseguramos que esté en BGR → HSV
        if crop is None or crop.size == 0:
            continue

        # --- KMEANS SOBRE LA CAMISETA ---
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)

        # kmeans local
        km = KMeans(n_clusters=2, random_state=0, n_init=5).fit(pixels)
        labels = km.labels_.reshape(hsv.shape[:2])
        centers = km.cluster_centers_

        # vamos a colorear cada cluster con su centro convertido a BGR para que sea intuitivo
        cluster_img = np.zeros_like(crop)
        for k in range(2):
            mask = (labels == k)
            # centro HSV → BGR
            center_hsv = np.uint8([[centers[k]]])
            center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0][0]
            cluster_img[mask] = center_bgr

        # subplot 1: original
        ax_orig = axes[row, col*2]
        ax_orig.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax_orig.set_title(f"Jugador {idx+1} original")
        ax_orig.axis("off")

        # subplot 2: cluster
        ax_cluster = axes[row, col*2 + 1]
        ax_cluster.imshow(cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB))
        ax_cluster.set_title("Clusters (camiseta vs césped)")
        ax_cluster.axis("off")

    plt.tight_layout()
    plt.show()

class TeamDetector:
    def __init__(self, team_colors_refs, confirmation_threshold=3, color_tolerance=25):
        self.team_colors_refs = team_colors_refs
        self.team_colors = {team: team_colors_refs[team] for team in team_colors_refs.keys()}
        
        self.color_samples = {team: {} for team in team_colors_refs.keys()}

        self.confirmation_threshold = confirmation_threshold
        self.color_tolerance = color_tolerance
        self.confirmed_teams  = set()

        self.shirtDetector = ShirtDetector()

    def updateTeamColors(self, shirt_color):
        if len(self.confirmed_teams) == len(self.team_colors):
            return
        
        distances = {team: np.linalg.norm(shirt_color - color) if color is not None else np.linalg.norm(shirt_color - self.team_colors_refs[team]) for team, color in self.team_colors.items()}
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
        distances = {team: np.linalg.norm(shirt_color - color) for team, color in self.team_colors.items()}
        return min(distances, key=distances.get), distances
    
    def getTeamOfPlayers(self, frame, shirts, bbox):
        x1, y1, x2, y2 = map(int, bbox[0])
        player_pixels = frame[y1:y2, x1:x2]
        if player_pixels.size > 0:
            h = player_pixels.shape[0]
            shirt = player_pixels[:int(0.5*h), :]
            shirts.append(shirt)
            shirt_color = self.shirtDetector.getColorKMeans(shirt)
            self.updateTeamColors(shirt_color)
            team, distances = self.assign_team(shirt_color)
            return team, distances, shirt_color, player_pixels.size
        return None, None, None, player_pixels.size
    
    def detectTeams(self, frame_detections, showPlot=False):
        shirts = []
        teams_of_detected_objects = []

        for object_detected in frame_detections:
            bbox = object_detected.boxes.xyxy
            class_name = object_detected.names[object_detected.boxes.cls.item()]
            team, distances, shirt_color, bboxSize = self.getTeamOfPlayers(frame_detections.orig_img, shirts, bbox)
            teams_of_detected_objects.append({"class": class_name, "team": team, "distances": distances, "shirt_color": shirt_color, "bbox_size": bboxSize})

        if showPlot and len(shirts) > 0:
            visualize_shirt_clusters(shirts)

        return teams_of_detected_objects