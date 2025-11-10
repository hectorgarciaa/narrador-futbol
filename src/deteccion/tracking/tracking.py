from ultralytics import YOLO
import supervision as sv
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from team_aware_bytetrack import ByteTrack
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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


# === 1️⃣ Visualizar KMeans dentro de una bbox ===
def plot_bbox_kmeans(image_bgr, kmeans_model):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    labels = kmeans_model.labels_
    centers = kmeans_model.cluster_centers_

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # scatter de los píxeles por cluster
    ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2],
            c=labels, cmap='viridis', s=2, alpha=0.5)

    # centroides
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
            c='red', s=200, marker='x')

    ax.set_xlabel("Hue")
    ax.set_ylabel("Saturation")
    ax.set_zlabel("Value")
    plt.title("KMeans por bounding box (camiseta vs césped)")
    plt.show()


# === 2️⃣ Visualizar clusters globales de los equipos ===
def plot_team_clusters(color_samples, kmeans_model):
    """
    Muestra cómo se agrupan los colores promedio de las camisetas
    en los dos clusters globales (equipos).
    """
    labels = kmeans_model.labels_
    centers = kmeans_model.cluster_centers_

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(color_samples[:, 0], color_samples[:, 1], color_samples[:, 2],
            c=labels, cmap='coolwarm', s=10, alpha=0.6)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
            c='black', s=200, marker='X')

    ax.set_xlabel("Hue")
    ax.set_ylabel("Saturation")
    ax.set_zlabel("Value")
    plt.title("KMeans global (colores medios de camisetas → equipos)")
    plt.show()

class Tracker:
    def __init__(self, ruta_modelo, conf_model, track_thresh, track_buffer, match_thresh, frame_rate, team_colors_hsv):
        self.modelo = YOLO(ruta_modelo)
        self.conf_model = conf_model
        self.tracker = ByteTrack(track_thresh, track_buffer, match_thresh, frame_rate)
        #self.tracker = sv.ByteTrack(track_thresh, track_buffer, match_thresh, frame_rate)
        self.team_refs_hsv = team_colors_hsv
        self.player_teams = {}
        self.player_teams_distances = {}

    def detect(self, partido):
        return self.modelo.predict(partido, stream=True, conf=self.conf_model)

    def get_shirt_color_kmeans(self, image_bgr):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)
        if len(pixels) < 2:
            return np.array([0, 0, 0])
        km = KMeans(n_clusters=2, random_state=0, n_init=5).fit(pixels)
        centers = km.cluster_centers_
        #plot_bbox_kmeans(image_bgr, km)
        lower_grass = np.array([35, 40, 40])
        upper_grass = np.array([85, 255, 255])

        def is_grass(c):
            h, s, v = c
            return lower_grass[0] <= h <= upper_grass[0] and \
                   lower_grass[1] <= s <= upper_grass[1] and \
                   lower_grass[2] <= v <= upper_grass[2]

        if is_grass(centers[0]) and not is_grass(centers[1]):
            shirt = centers[1]
        elif is_grass(centers[1]) and not is_grass(centers[0]):
            shirt = centers[0]
        else:
            shirt = centers[np.argmax(centers[:, 1])]
        return shirt

    def assign_team(self, shirt_hsv):
        #print("Shirt HSV:", shirt_hsv)
        #print("Team refs HSV:", self.team_refs_hsv)
        distances = {team: np.linalg.norm(shirt_hsv - ref_hsv)
                     for team, ref_hsv in self.team_refs_hsv.items()}
        #print(distances)
        return min(distances, key=distances.get), distances
    
    def get_tracks(self, partido):

        detections = self.detect(partido)
        tracks = {"player": [], "goalkeeper": [], "referee": [], "ball": []}

        for num_frame, detection_frame in enumerate(detections):
            detection_sv = sv.Detections.from_ultralytics(detection_frame)

            # --- 🔹 Calcular equipo para cada detección del frame ---
            team_labels = []
            team_distances_labels = []
            shirt_crops = []

            for bbox in detection_sv.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                crop = detection_frame.orig_img[y1:y2, x1:x2]

                if crop.size > 0:
                    h, w = crop.shape[:2]
                    shirt = crop[int(0.1 * h):int(0.5 * h), int(0.2 * w):int(0.8 * w)]
                    shirt_crops.append(shirt)
                    shirt_hsv = self.get_shirt_color_kmeans(shirt)
                    #print('Num_frame: ',num_frame)
                    team, distances = self.assign_team(shirt_hsv)
                    team_labels.append(team)
                    team_distances_labels.append(distances)
                else:
                    team_labels.append("Unknown")

            # --- 🔹 Pasar detecciones + equipos al tracker ---
            #[print(label) for label in team_labels]
            track = self.tracker.update_with_detections(detection_sv, team_labels=team_labels)

            # --- 🔹 Inicializar diccionarios por clase ---
            for key in tracks.keys():
                tracks[key].append({})

            # --- 🔹 Procesar cada track detectado ---
            for object_detected, det_team, det_distance in zip(track, team_labels, team_distances_labels):
                bbox, _, confidence, class_id, tracker_id, class_name = object_detected
                class_name = class_name["class_name"]

                # 🔸 Solo asignamos equipo si aún no lo tiene
                if class_name in ["player", "goalkeeper"]:
                    if tracker_id not in self.player_teams:
                        self.player_teams[tracker_id] = det_team
                    self.player_teams_distances[tracker_id] = det_distance

                # Guardar info del track
                tracks[class_name][num_frame][tracker_id] = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "team": self.player_teams.get(tracker_id, "Unknown"),
                    "distance": self.player_teams_distances.get(tracker_id, {})
                }

            # --- 🔹 Visualizar los clusters de camisetas ---
            '''
            if len(shirt_crops) > 0:
                visualize_shirt_clusters(shirt_crops)
            '''
        return tracks

    
    def draw_tracks(self, partido, tracks, output_path):
        cap = cv2.VideoCapture(partido)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_idx = 0
        colors = { "player": (0, 255, 0), "goalkeeper": (0, 255, 255), "referee": (255, 0, 0), "ball": (0, 0, 255) }

        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(tracks["player"]):
                break
            
            # Dibujar cajas para cada clase
            for class_name, frames_data in tracks.items():
                frame_data = frames_data[frame_idx]
                color = colors.get(class_name, (255, 255, 255))
                
                for track_id, data in frame_data.items():
                    #print(data)
                    x1, y1, x2, y2 = map(int, data["bbox"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    cv2.putText(frame, f"{class_name}-{data['team']}\n {[d.round(2) for d in data['distance'].values()]} #{track_id}", 
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, color, 1, cv2.LINE_AA)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"✅ Video anotado guardado en: {output_path}")
    

ruta_modelo = "../../../models/finetuning/v11/yolov11m/weights/bestx.pt"
partido = "../../../data/partidoPrueba/08fd33_4.mp4"
output = "../../../output/pruebaTracker/08fd33_4_3.mp4"

if __name__ == "__main__":
    TEAM_COLORS = {
    "Real Madrid": np.array([0, 0, 245]),      # blanco (HSV)
    "Wolfsburgo":  np.array([70, 150, 150])    # verde
}
    t = Tracker(ruta_modelo, 0.05, 0.05, 90, 0.95, 25, TEAM_COLORS)
    tracks = t.get_tracks(partido)
    #print(tracks)
    t.draw_tracks(partido, tracks, output)

