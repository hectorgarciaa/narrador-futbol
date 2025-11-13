import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def createDf(data):
    reformed_data = []
    for exp in data:
        base_data = { k:v for k,v in exp.items() if k != "track" }
        for class_name, class_info in exp["track"].items():
            for n_frame, frame_tracks in enumerate(class_info):
                for tracker_id, player_dict in frame_tracks.items():
                    row = base_data.copy()
                    row.update({'class_name': class_name, 'frame': n_frame, 'tracker_id': tracker_id})
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

def showPlot(df):
    TEAM_COLORS = { "Madrid": np.array([255, 127, 127]),       "Wolfs":  np.array([224, 77, 196]), 
                "Madrid Ident": np.array([230, 123, 125]),  "Wolfs Ident":  np.array([230, 100, 165]) }

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    mask = df["team"]=="Real Madrid"
    ax.scatter(df[mask]["L"], df[mask]["A"], df[mask]["B"], c="blue", label="Real Madrid", s=10, alpha=0.7)
    ax.scatter(df[~mask]["L"], df[~mask]["A"], df[~mask]["B"], c="red", label="Wolfsburgo", s=10, alpha=0.7)

    ax.scatter([TEAM_COLORS["Madrid"][0]], [TEAM_COLORS["Madrid"][1]], [TEAM_COLORS["Madrid"][2]], c="orange", label="Madrid - Color Oficial", s=100, marker='*', edgecolors='black', linewidth=1)
    ax.scatter([TEAM_COLORS["Wolfs"][0]], [TEAM_COLORS["Wolfs"][1]], [TEAM_COLORS["Wolfs"][2]], c="yellow", label="Wolfsb - Color Oficial",  s=100, marker='*', edgecolors='black', linewidth=1)
    ax.scatter([TEAM_COLORS["Madrid Ident"][0]], [TEAM_COLORS["Madrid Ident"][1]], [TEAM_COLORS["Madrid Ident"][2]], c="orange", label="Madrid Ident", s=150, marker='^', edgecolors='black', linewidth=1)
    ax.scatter([TEAM_COLORS["Wolfs Ident"][0]], [TEAM_COLORS["Wolfs Ident"][1]], [TEAM_COLORS["Wolfs Ident"][2]], c="yellow", label="Wolfsb Ident",  s=150, marker='^', edgecolors='black', linewidth=1)

    ax.set_xlabel('L'), ax.set_ylabel('A'), ax.set_zlabel('B'), ax.set_title('Espacio de Color LAB por Equipo')

    plt.tight_layout(), plt.legend()
    plt.show()

with open("../tracks3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = createDf(data)

showPlot(df[df["class_name"]=="goalkeeper"])

showPlot(df[df["class_name"]=="referee"])

showPlot(df[df["class_name"]=="player"])

showPlot(df[df["class_name"]=="ball"])