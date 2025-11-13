import math
import pandas as pd
import matplotlib.pyplot as plt

class TrackVisualizator:
    def __init__(self, metrics, metrics_list, summary, n_frames):
        self.metrics = pd.DataFrame(metrics).T
        self.metrics_list = pd.DataFrame(metrics_list)
        self.summary = summary
        self.n_frames = n_frames

    def show_hist(self, column, filter=None, criterio=None, bins=30):
        aux = self.metrics_list[column].explode()
        
        if filter is not None and criterio is not None:
            if criterio == ">":
                aux = aux[aux > filter]
            elif criterio == "<":
                aux = aux[aux < filter]
        
        aux.hist(bins=bins)
        plt.title(f"Histograma de {column} {criterio if criterio else ''} {filter if filter else ''}")
        plt.show()

    def show_tracks(self):
        # Calcular cuántos tracks por fila
        tracks_per_row = 5
        num_rows = math.ceil(len(aux) / tracks_per_row)

        fig, axes = plt.subplots(num_rows, 1, figsize=(30, 3 * num_rows))
        if num_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Colores distintos para cada TID
        colors = plt.cm.tab10(np.linspace(0, 1, len(aux)))

        for row_idx, ax in enumerate(axes):
            start_tid_idx = row_idx * tracks_per_row
            end_tid_idx = min((row_idx + 1) * tracks_per_row, len(aux))
            
            tids_in_row = aux.index[start_tid_idx:end_tid_idx]
            
            for i, tid in enumerate(tids_in_row):
                row = aux.loc[tid]
                frames_seen = row["frames_seen"]
                confidences = row["confs"]
                
                # Crear array completo con 0 para frames no vistos
                y_full = np.zeros(T)
                for frame, conf in zip(frames_seen, confidences):
                    if frame < T:
                        y_full[frame] = conf
                
                # Plot con offset vertical para separar tracks
                vertical_offset = i * 0.1  # Ajusta este valor según necesites
                ax.plot(range(T), y_full + vertical_offset, 
                        color=colors[start_tid_idx + i], 
                        label=f"TID#{tid}", marker='o', markersize=1, alpha=0.7, linewidth=0.5)
            
            ax.set_ylabel("Confidence")
            ax.set_xlim(0, T)
            ax.legend(loc='upper right', fontsize=8)
            if row_idx == num_rows - 1:
                ax.set_xlabel("Frames")
            ax.set_title(f"Tracks {start_tid_idx}-{end_tid_idx-1}")

        plt.suptitle("Confidence vs Frames por TID (Agrupado)", fontsize=16)
        plt.tight_layout()
        plt.show()




