import logging
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from football_ai.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)

class TrackVisualizer:
    def __init__(self, tracks, class_name):
        self.evaluator = Evaluator()
        _, metrics_list, _, n_frames = self.evaluator.evaluate_class(tracks, class_name)

        self.df_metrics_list = pd.DataFrame(metrics_list).T
        self.n_frames = n_frames
    
    def get_df_metrics_list(self):
        return self.df_metrics_list
    
    def show_all_hists_metrics_list(self):
        columns = [c for c in self.df_metrics_list.columns if c not in ['frames_seen', 'speed_frames']]
        logger.debug(f"Columns for histograms: {columns}")
        ncols = min(7, len(columns))
        nrows = math.ceil(len(columns)/ ncols)
        _, axes = plt.subplots(nrows, ncols, figsize=(25, 4*nrows))
        axes = axes.flatten()
        for i, c in enumerate(columns):
            aux = self.df_metrics_list[c].explode().dropna()
            if aux.dtype == bool or all(isinstance(x, (bool, np.bool_)) for x in aux.dropna()[:10]):
                true_count, false_count = (aux == True).sum(), (aux == False).sum()
                axes[i].bar(['False', 'True'], [false_count, true_count], color=['red', 'blue'])
                axes[i].set_title(f"{c}\n(True: {true_count}, False: {false_count})")
            else:
                axes[i].hist(aux, bins=20)
                axes[i].set_title(c)

        plt.tight_layout()
        plt.show()

    def show_hist(self, nparray, column, bins=30):
        nparray.hist(bins=bins)
        plt.title(f"Histograma de {column}")
        plt.show()

    def show_tracks_evolution(self, y, cov_threshold, tracks_per_row, vertical_offset):
        self.df_metrics_list["mean_coverage"] = self.df_metrics_list["frames_seen"].str.len() / self.n_frames
        aux = self.df_metrics_list[self.df_metrics_list["mean_coverage"] < cov_threshold]

        num_rows = math.ceil(len(aux) / tracks_per_row)

        _, axes = plt.subplots(num_rows, 1, figsize=(30, 3 * num_rows))
        if num_rows == 1:   axes = [axes]
        else:               axes = axes.flatten()

        # Colores distintos para cada TID
        colors = plt.cm.tab10(np.linspace(0, 1, len(aux)))

        for row_idx, ax in enumerate(axes):
            start_tid_idx = row_idx * tracks_per_row
            end_tid_idx = min((row_idx + 1) * tracks_per_row, len(aux))
            
            tids_in_row = aux.index[start_tid_idx:end_tid_idx]
            
            for i, tid in enumerate(tids_in_row):
                row = aux.loc[tid]
                frames_seen = row["frames_seen"]
                confidences = row[y]
                
                # Crear array completo con 0 para frames no vistos
                y_full = np.zeros(self.n_frames)
                for frame, conf in zip(frames_seen, confidences):
                    if frame < self.n_frames:
                        y_full[frame] = conf
                
                # Plot con offset vertical para separar tracks
                v_offset = i * 0.02 if vertical_offset else 0
                ax.plot(range(self.n_frames), y_full + v_offset, 
                        color=colors[start_tid_idx + i], 
                        label=f"TID#{tid}", marker='o', markersize=1, alpha=0.7, linewidth=0.5)
            
            ax.set_ylabel(y)
            ax.set_xlim(0, self.n_frames)
            ax.legend(loc='upper right', fontsize=8)
            if row_idx == num_rows - 1:
                ax.set_xlabel("Frames")
            ax.set_title(f"Tracks {start_tid_idx}-{end_tid_idx-1}")

        plt.suptitle("Confidence vs Frames por TID (Agrupado)", fontsize=16)
        plt.tight_layout()
        plt.show()
