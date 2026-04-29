import cv2
import numpy as np

from .kmeans_batch_gpu import BatchedKMeansGPU


class ShirtDetector:
    def __init__(
        self,
        pixels_resize=1536,
        batched_kmeans_gpu_conf= {},
    ):
        self.pixels_resize = max(2, int(pixels_resize))
        self.kmeans = BatchedKMeansGPU(**batched_kmeans_gpu_conf)

    def get_color_kmeans_batch(self, images):
        pixels_by_idx = {}
        references_by_idx = {}
        for idx, image in enumerate(images):
            lab = self._prepare_lab_image(image)
            pixels = lab.reshape(-1, 3).astype(np.float32)
            if len(pixels) < self.kmeans.n_clusters:
                continue
            pixels_by_idx[idx] = pixels
            references_by_idx[idx] = self._reference_pixels(lab)

        self.kmeans.fit_batch(pixels_by_idx)
        reference_labels = self.kmeans.predict_batch(references_by_idx)

        colors = []
        for idx in range(len(images)):
            centers = None if self.kmeans.cluster_centers_ is None else self.kmeans.cluster_centers_.get(idx)
            labels = reference_labels.get(idx)
            colors.append(self._shirt_color_from_clusters(centers, labels))
        return colors

    def _prepare_lab_image(self, image):
        resized = self._resize_for_color_clustering(image)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

    def _resize_for_color_clustering(self, image):
        height, width = image.shape[:2]
        aspect_ratio = float(width) / float(height)
        new_height = max(1, int(round(np.sqrt(self.pixels_resize / aspect_ratio))))
        new_width = max(1, int(round(np.sqrt(self.pixels_resize * aspect_ratio))))
        interpolation = cv2.INTER_AREA if new_height <= height and new_width <= width else cv2.INTER_LINEAR
        return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    def _reference_pixels(self, lab):
        height, width = lab.shape[:2]
        reference_points = [
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1),
            (width // 2, 0),
            (0, height // 2),
            (width - 1, height // 2),
        ]
        return np.asarray([lab[y, x] for x, y in reference_points], dtype=np.float32)

    def _shirt_color_from_clusters(self, centers, reference_labels):
        if centers is None:
            return np.array([0, 0, 0], dtype=np.float32)

        centers = np.asarray(centers, dtype=np.float32)
        if centers.ndim == 1:
            centers = centers.reshape(1, -1)
        if len(centers) == 1:
            return centers[0]

        labels = np.asarray(reference_labels, dtype=np.int32).reshape(-1)
        if labels.size <= 0:
            return centers[0]

        unique_labels, counts = np.unique(labels, return_counts=True)
        background_label = int(unique_labels[np.argmax(counts)])
        shirt_label = 1 - background_label
        return np.asarray(centers[shirt_label], dtype=np.float32)
