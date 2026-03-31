import cv2
import numpy as np
from sklearn.cluster import KMeans

class ShirtDetector:
    def __init__(
        self,
        n_clusters=2,
        init='k-means++',
        n_init=3,
        random_state=0,
        downsample=1.0,
    ):
        self.downsample = float(downsample)
        if not np.isfinite(self.downsample) or self.downsample <= 0.0:
            self.downsample = 1.0
        self.downsample = min(self.downsample, 1.0)
        self._rng = np.random.default_rng(int(random_state))
        self.km = KMeans(n_clusters=n_clusters, init=init, n_init=n_init,
                         random_state=random_state)

    def get_color_kmeans(self, image):
        """Gets the dominant shirt color using KMeans in LAB color space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3)
        if len(pixels) < 2:
            return np.array([0, 0, 0])

        pixels_to_fit = pixels
        if self.downsample < 1.0:
            sample_size = max(2, int(round(len(pixels) * self.downsample)))
            if sample_size < len(pixels):
                sampled_indexes = self._rng.choice(
                    len(pixels),
                    size=sample_size,
                    replace=False,
                )
                pixels_to_fit = pixels[sampled_indexes]

        self.km.fit(pixels_to_fit)
        centers = self.km.cluster_centers_
        
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
        return np.asarray(shirt_color, dtype=np.float32)
