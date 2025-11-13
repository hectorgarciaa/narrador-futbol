import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open("../tracks2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

exp = data[0]["track"]
players = exp["goalkeeper"]

reformed_data = []
for n_frame, frame_tracks in enumerate(players):
    for tracker_id, player_dict in frame_tracks.items():
        row = {'frame': n_frame, 'tracker_id': tracker_id}
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
    break

df = pd.DataFrame(reformed_data)
df = df.set_index(['frame', 'tracker_id'])

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

mask = df["team"]=="Real Madrid"

TEAM_COLORS = { "Madrid": np.array([255, 127, 127]),       "Wolfs":  np.array([224, 77, 196]), 
                "Madrid Ident": np.array([230, 123, 125]),  "Wolfs Ident":  np.array([230, 100, 165]) }

ax.scatter(df[mask]["L"], df[mask]["A"], df[mask]["B"], c="blue", label="Madrid", s=10, alpha=0.7)
ax.scatter(df[~mask]["L"], df[~mask]["A"], df[~mask]["B"], c="red", label="Wolfs", s=10, alpha=0.7)

ax.scatter([TEAM_COLORS["Madrid"][0]], [TEAM_COLORS["Madrid"][1]], [TEAM_COLORS["Madrid"][2]], c="orange", label="Madrid - Color Oficial", s=200, marker='*', edgecolors='black', linewidth=2)
ax.scatter([TEAM_COLORS["Wolfs"][0]], [TEAM_COLORS["Wolfs"][1]], [TEAM_COLORS["Wolfs"][2]], c="yellow", label="Wolfsb - Color Oficial",  s=200, marker='*', edgecolors='black', linewidth=2)
ax.scatter([TEAM_COLORS["Madrid Ident"][0]], [TEAM_COLORS["Madrid Ident"][1]], [TEAM_COLORS["Madrid Ident"][2]], c="orange", label="Madrid Ident", s=200, marker='^', edgecolors='black', linewidth=2)
ax.scatter([TEAM_COLORS["Wolfs Ident"][0]], [TEAM_COLORS["Wolfs Ident"][1]], [TEAM_COLORS["Wolfs Ident"][2]], c="yellow", label="Wolfsb Ident",  s=200, marker='^', edgecolors='black', linewidth=2)

ax.set_xlabel('Color L'), ax.set_ylabel('Color A'), ax.set_zlabel('Color B'), ax.set_title('Espacio de Color LAB por Equipo')

plt.tight_layout(), plt.legend(), plt.show()