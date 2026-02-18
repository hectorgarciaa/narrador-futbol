import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def visualize_shirt_clusters(shirt_crops, max_players=20):
    """
    Muestra hasta N crops de camisetas en una cuadricula.
    Para cada crop:
      - izquierda: crop original
      - derecha: crop coloreado por cluster (KMeans k=2)
    """
    shirt_crops = shirt_crops[:max_players]
    n = len(shirt_crops)
    if n == 0:
        return

    cols_players = 5
    rows_players = int(np.ceil(n / cols_players))

    fig, axes = plt.subplots(rows_players, cols_players * 2,
                             figsize=(cols_players * 4, rows_players * 3))
    if rows_players == 1:
        axes = np.expand_dims(axes, 0)

    for idx, crop in enumerate(shirt_crops):
        row = idx // cols_players
        col = idx % cols_players

        if crop is None or crop.size == 0:
            continue

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)

        km = KMeans(n_clusters=2, random_state=0, n_init=5).fit(pixels)
        labels = km.labels_.reshape(hsv.shape[:2])
        centers = km.cluster_centers_

        cluster_img = np.zeros_like(crop)
        for k in range(2):
            mask = (labels == k)
            center_hsv = np.uint8([[centers[k]]])
            center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0][0]
            cluster_img[mask] = center_bgr

        ax_orig = axes[row, col * 2]
        ax_orig.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax_orig.set_title(f"Jugador {idx + 1} original")
        ax_orig.axis("off")

        ax_cluster = axes[row, col * 2 + 1]
        ax_cluster.imshow(cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB))
        ax_cluster.set_title("Clusters (camiseta vs cesped)")
        ax_cluster.axis("off")

    plt.tight_layout()
    plt.show()


class ClusterVisualizer:
    def __init__(self, tracks, video_path):
        self.reformed_tracks = self.reformat_track(tracks)
        self.video_path = video_path

    def reformat_track(self, tracks):
        """Reformatea los tracks a un DataFrame plano para analisis."""
        reformed_data = []
        for class_name, class_info in tracks.items():
            for n_frame, frame_tracks in enumerate(class_info):
                for tracker_id, player_dict in frame_tracks.items():
                    if player_dict.get("distances") is None or player_dict.get("shirt_color") is None:
                        continue
                    row = {
                        'class_name': class_name,
                        'frame': n_frame,
                        'tracker_id': tracker_id,
                        "x1": player_dict["bbox"][0],
                        "y1": player_dict["bbox"][1],
                        "w": player_dict["bbox"][2] - player_dict["bbox"][0],
                        "h": player_dict["bbox"][3] - player_dict["bbox"][1],
                        "bbox_size": player_dict["bbox_size"],
                        "confidence": player_dict["confidence"],
                        "L": player_dict["shirt_color"][0],
                        "A": player_dict["shirt_color"][1],
                        "B": player_dict["shirt_color"][2],
                        "team": player_dict["team"],
                    }
                    # Distancias dinamicas por nombre de equipo
                    for team_name, dist in player_dict["distances"].items():
                        row[team_name] = dist
                    reformed_data.append(row)

        return pd.DataFrame(reformed_data)

    def get_data_frame(self):
        """Devuelve el DataFrame reformateado."""
        return self.reformed_tracks

    def show_hist_of_class_name(self, axes, df, class_name, cols, xlabel):
        """Muestra histogramas para una clase especifica."""
        for i, c in enumerate(cols):
            axes[i].hist(df[c], bins=256)
            axes[i].set_title(c)
            axes[i].set_xlabel(xlabel)
            axes[i].grid()
            axes[i].set_ylabel(f"Freq {class_name}")

    def show_hist(self, cols, xlabel, width):
        """Muestra histogramas comparativos por clase."""
        _, axes = plt.subplots(5, len(cols), figsize=(25, width), sharey="row")

        self.show_hist_of_class_name(axes[0], self.reformed_tracks, "Global", cols, xlabel)
        for i, class_name in enumerate(self.reformed_tracks["class_name"].unique()):
            aux = self.reformed_tracks[self.reformed_tracks["class_name"] == class_name]
            self.show_hist_of_class_name(axes[i + 1], aux, class_name, cols, xlabel)

        plt.tight_layout()
        plt.show()

    def get_cols(self):
        """Devuelve las columnas del DataFrame."""
        return list(self.reformed_tracks.columns)

    def show_clusters(self, df, filtro, n_clusters=2, init='k-means++', n_init=10, random_state=0):
        """Muestra clusters visuales de los crops de camisetas."""
        if len(filtro) > 0:
            df = df.drop_duplicates(subset=filtro)

        km = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=random_state)
        bbox_info = []

        cap = cv2.VideoCapture(self.video_path)

        for i, row in df.iterrows():
            frame_idx = row["frame"]
            x1, y1, w, h = [int(row["x1"]), int(row["y1"]), int(row["w"]), int(row["h"])]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bbox_full = frame_rgb[y1:y1 + h, x1:x1 + w]
                if bbox_full.size > 0 and h > 0:
                    bbox_upper = bbox_full[:h // 2, :]
                    bbox_flat = bbox_upper.reshape(-1, 3)
                    km.fit(bbox_flat)
                    predictions = km.predict(bbox_flat)
                    clustered = km.cluster_centers_[predictions].astype(np.uint8)
                    clustered_img = clustered.reshape(h // 2, w, 3)
                    bbox_info.append({
                        'frame_idx': frame_idx, 'bbox_size': row["bbox_size"],
                        'upper_bbox': bbox_upper, 'clustered': clustered_img,
                        'team': row.get('team', 'Unknown')
                    })

        cap.release()

        n_samples = len(bbox_info)
        n_cols = 8
        n_rows = math.ceil(n_samples / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 3 * n_rows))
        axes = np.array(axes).reshape(-1)

        for i in range(n_samples):
            info = bbox_info[i]
            axes[i].imshow(np.hstack((info['upper_bbox'], info['clustered'])))
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def show_clusters_3d(self, class_name, n_frames, team_colors=None):
        """
        Muestra un scatter 3D en espacio LAB por equipo.

        Args:
            class_name: Clase a visualizar (ej: 'player')
            n_frames: Numero maximo de frames a incluir
            team_colors: Dict opcional {nombre_equipo: np.array([L, A, B])}
        """
        df = self.reformed_tracks[self.reformed_tracks["class_name"] == class_name]
        df = df[df["frame"] <= n_frames]

        team_names = df["team"].dropna().unique().tolist()

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')

        plot_colors = plt.cm.tab10(np.linspace(0, 1, max(len(team_names), 2)))

        for idx, team_name in enumerate(team_names):
            mask = df["team"] == team_name
            ax.scatter(df[mask]["L"], df[mask]["A"], df[mask]["B"],
                       c=[plot_colors[idx]], label=team_name, s=10, alpha=0.7)

        if team_colors is not None:
            marker_colors = ['orange', 'yellow', 'cyan', 'magenta']
            for idx, (team_name, color_lab) in enumerate(team_colors.items()):
                mc = marker_colors[idx % len(marker_colors)]
                ax.scatter([color_lab[0]], [color_lab[1]], [color_lab[2]],
                           c=mc, label=f"{team_name} - Ref",
                           s=100, marker='*', edgecolors='black', linewidth=1)

        ax.set_xlabel('L')
        ax.set_ylabel('A')
        ax.set_zlabel('B')
        ax.set_title('Espacio de Color LAB por Equipo')

        plt.tight_layout()
        plt.legend()
        plt.show()
