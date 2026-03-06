import cv2
import numpy as np
from sklearn.cluster import KMeans
from time import perf_counter

class ShirtDetector:
    def __init__(self, n_clusters=2, init='k-means++', n_init=10, random_state=0):
        self.km = KMeans(n_clusters=n_clusters, init=init, n_init=n_init,
                         random_state=random_state)
        self.last_timing_detail = {}

    def get_color_kmeans(self, image, return_timing=False):
        """Gets the dominant shirt color using KMeans in LAB color space."""
        total_start = perf_counter()

        color_convert_start = perf_counter()
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        color_convert_s = perf_counter() - color_convert_start

        reshape_start = perf_counter()
        pixels = lab.reshape(-1, 3)
        reshape_s = perf_counter() - reshape_start
        if len(pixels) < 2:
            timing = {
                "kmeans_preprocess_color_convert_s": float(color_convert_s),
                "kmeans_preprocess_reshape_s": float(reshape_s),
                "kmeans_fit_s": 0.0,
                "kmeans_cluster_selection_s": 0.0,
                "kmeans_total_s": float(perf_counter() - total_start),
            }
            self.last_timing_detail = timing
            if return_timing:
                return np.array([0, 0, 0]), timing
            return np.array([0, 0, 0])
        
        kmeans_fit_start = perf_counter()
        self.km.fit(pixels)
        kmeans_fit_s = perf_counter() - kmeans_fit_start
        centers = self.km.cluster_centers_
        
        cluster_selection_start = perf_counter()
        height, width = image.shape[:2]
        references_points = [
            (0, 0), (width-1, 0), (0, height-1), (width-1, height-1),
            (width//2, 0), (0, height//2), (width-1, height//2)
        ]
        references_pixels = np.array([lab[y, x] for x, y in references_points])
        references_labels = self.km.predict(references_pixels)
        unique_labels, counts = np.unique(references_labels, return_counts=True)
        background_label = unique_labels[np.argmax(counts)]
        shirt_label = 1 - background_label
        shirt_color = centers[shirt_label]
        cluster_selection_s = perf_counter() - cluster_selection_start

        timing = {
            "kmeans_preprocess_color_convert_s": float(color_convert_s),
            "kmeans_preprocess_reshape_s": float(reshape_s),
            "kmeans_fit_s": float(kmeans_fit_s),
            "kmeans_cluster_selection_s": float(cluster_selection_s),
            "kmeans_total_s": float(perf_counter() - total_start),
        }
        self.last_timing_detail = timing
        if return_timing:
            return shirt_color, timing
        return shirt_color
