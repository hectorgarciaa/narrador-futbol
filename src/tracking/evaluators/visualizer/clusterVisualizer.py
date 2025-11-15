import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class ClusterVisualizer:
    def __init__(self, tracks, video_path):
        self.reformedTracks = self.reformatTrack(tracks)
        self.video_path = video_path

    def reformatTrack(self, tracks):
        reformed_data = []
        # for exp in tracks:
        #    base_data = { k:v for k,v in exp.items() if k != "track" }
        for class_name, class_info in tracks.items():
            for n_frame, frame_tracks in enumerate(class_info):
                for tracker_id, player_dict in frame_tracks.items():
                    #row = base_data.copy()
                    # row.update({'class_name': class_name, 'frame': n_frame, 'tracker_id': tracker_id})
                    row= {'class_name': class_name, 'frame': n_frame, 'tracker_id': tracker_id}
                    row.update({
                        "x1": player_dict["bbox"][0],
                        "y1": player_dict["bbox"][1],
                        "w": player_dict["bbox"][2] - player_dict["bbox"][0],
                        "h": player_dict["bbox"][3] - player_dict["bbox"][1],
                        "bbox_size": player_dict["bbox_size"],
                        "confidence": player_dict["confidence"],
                        "Real Madrid": player_dict["distances"]["Real Madrid"],
                        "Wolfsburgo": player_dict["distances"]["Wolfsburgo"],
                        "L": player_dict["shirt_color"][0],
                        "A": player_dict["shirt_color"][1],
                        "B": player_dict["shirt_color"][2],
                        "team": player_dict["team"]
                    } )
                    reformed_data.append(row)

        return pd.DataFrame(reformed_data)
    
    def getDataFrame(self):
        return self.reformedTracks
    
    def showHistOfClassName(self, axes, df, class_name, cols, xlabel):
        for i, c in enumerate(cols):
            axes[i].hist(df[c], bins=256), axes[i].set_title(c), axes[i].set_xlabel(xlabel), axes[i].grid(), axes[i].set_ylabel(f"Freq {class_name}")

    def showHist(self, cols, xlabel, width):
        _, axes = plt.subplots(5, len(cols), figsize=(25, width), sharey="row")

        self.showHistOfClassName(axes[0], self.reformedTracks, "Global", cols, xlabel)
        for i, class_name in enumerate(self.reformedTracks["class_name"].unique()):
            aux = self.reformedTracks[self.reformedTracks["class_name"]==class_name]
            self.showHistOfClassName(axes[i+1], aux, class_name, cols, xlabel)

        plt.tight_layout()
        plt.show()

    def getCols(self):
        return list(self.reformedTracks.columns)
    
    def showClusters(self, df, filtro, n_clusters=2, init='k-means++', n_init=10, random_state=0):
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
                bbox_full = frame_rgb[y1:y1+h, x1:x1+w]
                if bbox_full.size > 0 and h > 0:
                    bbox_upper = bbox_full[:h // 2, :]
                    bbox_flat = bbox_upper.reshape(-1, 3)  # (64*64, 3)
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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 3*n_rows))
        axes = np.array(axes).reshape(-1)
        
        for i in range(n_samples):
            info = bbox_info[i]
            axes[i].imshow(np.hstack((info['upper_bbox'], info['clustered'])))
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

    def showClusters3D(self, class_name, n_frames):
        df = self.reformedTracks[self.reformedTracks["class_name"]==class_name]
        df = df[df["frame"]<=n_frames]

        TEAM_COLORS = { "Madrid": np.array([255, 127, 127]),       "Wolfs":  np.array([224, 77, 196]), 
                "Madrid Ident": np.array([230, 123, 125]),  "Wolfs Ident":  np.array([230, 100, 165]) }

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')

        mask = df["team"]=="Real Madrid"
        ax.scatter(df[mask]["L"], df[mask]["A"], df[mask]["B"],
                   c="blue", label="Real Madrid", s=10, alpha=0.7)
        ax.scatter(df[~mask]["L"], df[~mask]["A"], df[~mask]["B"],
                   c="red", label="Wolfsburgo", s=10, alpha=0.7)

        ax.scatter([TEAM_COLORS["Madrid"][0]], [TEAM_COLORS["Madrid"][1]], [TEAM_COLORS["Madrid"][2]], c="orange", label="Madrid - Color Oficial", s=100, marker='*', edgecolors='black', linewidth=1)
        ax.scatter([TEAM_COLORS["Wolfs"][0]], [TEAM_COLORS["Wolfs"][1]], [TEAM_COLORS["Wolfs"][2]], c="yellow", label="Wolfsb - Color Oficial",  s=100, marker='*', edgecolors='black', linewidth=1)
        ax.scatter([TEAM_COLORS["Madrid Ident"][0]], [TEAM_COLORS["Madrid Ident"][1]], [TEAM_COLORS["Madrid Ident"][2]], c="orange", label="Madrid Ident", s=150, marker='^', edgecolors='black', linewidth=1)
        ax.scatter([TEAM_COLORS["Wolfs Ident"][0]], [TEAM_COLORS["Wolfs Ident"][1]], [TEAM_COLORS["Wolfs Ident"][2]], c="yellow", label="Wolfsb Ident",  s=150, marker='^', edgecolors='black', linewidth=1)

        ax.set_xlabel('L'), ax.set_ylabel('A'), ax.set_zlabel('B'), ax.set_title('Espacio de Color LAB por Equipo')

        plt.tight_layout(), plt.legend()
        plt.show()
