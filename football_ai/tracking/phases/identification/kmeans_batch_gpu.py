import numpy as np
from sklearn.cluster import KMeans

try:
    import cupy as cp
except Exception:
    cp = None

try:
    from cuml.cluster import KMeans as CuKMeans
except Exception:
    CuKMeans = None


class BatchedKMeansGPU:
    def __init__(
        self,
        n_clusters=2,
        init="k-means++",
        n_init=1,
        use_gpu=True,
        max_iter=12,
        tol=1e-3,
        random_state=0,
    ):
        self.n_clusters = int(n_clusters)
        self.init = str(init or "k-means++")
        self.n_init = max(1, int(n_init))
        self.use_gpu = bool(use_gpu and cp is not None)
        self.max_iter = max(1, int(max_iter))
        self.tol = max(0.0, float(tol))
        self.random_state = int(random_state)
        self._cp = cp if self.use_gpu else None
        self._km_gpu = None
        self._km_cpu = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        if self.use_gpu and CuKMeans is not None:
            try:
                self._km_gpu = CuKMeans(
                    n_clusters=self.n_clusters,
                    init=self.init,
                    n_init=self.n_init,
                    random_state=self.random_state,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
            except Exception:
                self._km_gpu = None
                self.use_gpu = False
        else:
            self.use_gpu = False

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit_batch(self, samples):
        if not samples:
            self.cluster_centers_ = {}
            self.labels_ = {}
            self.inertia_ = {}
            return self

        if self.use_gpu and self._cp is not None:
            return self._fit_batch_gpu(samples)
        return self._fit_batch_cpu(samples)

    def predict_batch(self, samples):
        if not samples:
            return {}

        if self.use_gpu and self._cp is not None and isinstance(self.cluster_centers_, dict):
            return self._predict_batch_gpu(samples)

        predictions = {}
        for sample_id, x in samples.items():
            centers = np.asarray(self.cluster_centers_[sample_id], dtype=np.float32)
            points = np.asarray(x, dtype=np.float32)
            distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            predictions[sample_id] = np.argmin(distances, axis=1).astype(np.int32)
        return predictions

    def _fit_batch_cpu(self, samples):
        self.cluster_centers_ = {}
        self.labels_ = {}
        self.inertia_ = {}

        for sample_id, x in samples.items():
            km = KMeans(
                n_clusters=self.n_clusters,
                init=self.init,
                n_init=self.n_init,
                random_state=self.random_state,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            points = np.asarray(x, dtype=np.float32)
            km.fit(points)
            self.cluster_centers_[sample_id] = np.asarray(km.cluster_centers_, dtype=np.float32)
            self.labels_[sample_id] = np.asarray(km.labels_, dtype=np.int32)
            self.inertia_[sample_id] = float(km.inertia_)
        return self

    def _fit_batch_gpu(self, samples):
        sample_ids = list(samples.keys())
        n_samples = len(sample_ids)
        max_points = max(len(np.asarray(samples[sample_id])) for sample_id in sample_ids)

        x = self._cp.zeros((n_samples, max_points, 3), dtype=self._cp.float32)
        valid_mask = self._cp.zeros((n_samples, max_points), dtype=self._cp.bool_)

        for row_idx, sample_id in enumerate(sample_ids):
            points = np.asarray(samples[sample_id], dtype=np.float32)
            num_points = len(points)
            x[row_idx, :num_points, :] = self._cp.asarray(points, dtype=self._cp.float32)
            valid_mask[row_idx, :num_points] = True

        centers, labels, inertia = self._run_batch_gpu(x, valid_mask)
        centers_np = self._cp.asnumpy(centers)
        labels_np = self._cp.asnumpy(labels)
        inertia_np = self._cp.asnumpy(inertia)

        self.cluster_centers_ = {}
        self.labels_ = {}
        self.inertia_ = {}
        for row_idx, sample_id in enumerate(sample_ids):
            num_points = len(np.asarray(samples[sample_id]))
            self.cluster_centers_[sample_id] = np.asarray(centers_np[row_idx], dtype=np.float32)
            self.labels_[sample_id] = np.asarray(labels_np[row_idx, :num_points], dtype=np.int32)
            self.inertia_[sample_id] = float(inertia_np[row_idx])
        return self

    def _predict_batch_gpu(self, samples):
        sample_ids = list(samples.keys())
        n_samples = len(sample_ids)
        max_points = max(len(np.asarray(samples[sample_id])) for sample_id in sample_ids)

        x = self._cp.zeros((n_samples, max_points, 3), dtype=self._cp.float32)
        valid_mask = self._cp.zeros((n_samples, max_points), dtype=self._cp.bool_)
        centers = self._cp.zeros((n_samples, self.n_clusters, 3), dtype=self._cp.float32)

        for row_idx, sample_id in enumerate(sample_ids):
            points = np.asarray(samples[sample_id], dtype=np.float32)
            num_points = len(points)
            x[row_idx, :num_points, :] = self._cp.asarray(points, dtype=self._cp.float32)
            valid_mask[row_idx, :num_points] = True
            centers[row_idx] = self._cp.asarray(self.cluster_centers_[sample_id], dtype=self._cp.float32)

        inf = self._cp.asarray(np.inf, dtype=self._cp.float32)
        distances = self._cp.sum((x[:, :, None, :] - centers[:, None, :, :]) ** 2, axis=3)
        distances = self._cp.where(valid_mask[:, :, None], distances, inf)
        labels = self._cp.argmin(distances, axis=2)
        labels_np = self._cp.asnumpy(labels)

        predictions = {}
        for row_idx, sample_id in enumerate(sample_ids):
            num_points = len(np.asarray(samples[sample_id]))
            predictions[sample_id] = np.asarray(labels_np[row_idx, :num_points], dtype=np.int32)
        return predictions

    def _run_batch_gpu(self, x, valid_mask):
        n_samples, max_points, _ = x.shape
        inf = self._cp.asarray(np.inf, dtype=self._cp.float32)
        eps = self._cp.asarray(1e-6, dtype=self._cp.float32)
        best_centers = None
        best_labels = None
        best_inertia = None

        for init_idx in range(self.n_init):
            centers = self._initialize_centers_gpu(x, valid_mask, init_idx)
            for _ in range(self.max_iter):
                distances = self._cp.sum((x[:, :, None, :] - centers[:, None, :, :]) ** 2, axis=3)
                distances = self._cp.where(valid_mask[:, :, None], distances, inf)
                labels = self._cp.argmin(distances, axis=2)
                new_centers = self._update_centers_gpu(x, valid_mask, labels, centers, eps)
                delta = self._cp.max(self._cp.linalg.norm(new_centers - centers, axis=2))
                centers = new_centers
                if float(self._cp.asnumpy(delta)) <= self.tol:
                    break

            distances = self._cp.sum((x[:, :, None, :] - centers[:, None, :, :]) ** 2, axis=3)
            distances = self._cp.where(valid_mask[:, :, None], distances, inf)
            labels = self._cp.argmin(distances, axis=2)
            row_indices = self._cp.arange(n_samples)[:, None]
            point_indices = self._cp.arange(max_points)[None, :]
            point_inertia = distances[row_indices, point_indices, labels]
            inertia = self._cp.sum(
                self._cp.where(valid_mask, point_inertia, self._cp.float32(0.0)),
                axis=1,
            )

            if best_inertia is None:
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_inertia = inertia.copy()
                continue

            improved = inertia < best_inertia
            if self._cp.any(improved):
                best_centers[improved] = centers[improved]
                best_labels[improved] = labels[improved]
                best_inertia[improved] = inertia[improved]

        return best_centers, best_labels, best_inertia

    def _update_centers_gpu(self, x, valid_mask, labels, centers, eps):
        new_centers = self._cp.empty_like(centers)
        for cluster_idx in range(self.n_clusters):
            cluster_mask = valid_mask & (labels == cluster_idx)
            counts = self._cp.sum(cluster_mask, axis=1)
            cluster_sum = self._cp.sum(x * cluster_mask[:, :, None], axis=1)
            new_centers[:, cluster_idx, :] = cluster_sum / self._cp.maximum(counts[:, None], eps)
            empty_mask = counts <= 0
            if self._cp.any(empty_mask):
                new_centers[empty_mask, cluster_idx, :] = centers[empty_mask, cluster_idx, :]
        return new_centers

    def _initialize_centers_gpu(self, x, valid_mask, init_idx):
        n_samples = int(x.shape[0])
        centers = self._cp.zeros((n_samples, self.n_clusters, 3), dtype=self._cp.float32)

        for row_idx in range(n_samples):
            valid_idx = self._cp.where(valid_mask[row_idx])[0]
            points = self._cp.asnumpy(x[row_idx, valid_idx, :])
            chosen = self._choose_initial_indices(points, init_idx, row_idx)
            centers[row_idx] = self._cp.asarray(points[chosen], dtype=self._cp.float32)
        return centers

    def _choose_initial_indices(self, points, init_idx, row_idx):
        seed = self.random_state + (init_idx * 1009) + row_idx
        rng = np.random.default_rng(seed)

        if self.init == "random":
            return rng.choice(len(points), size=self.n_clusters, replace=len(points) < self.n_clusters)

        first_idx = int(rng.integers(len(points)))
        chosen = [first_idx]
        min_dist_sq = np.sum((points - points[first_idx]) ** 2, axis=1)

        while len(chosen) < self.n_clusters:
            total = float(min_dist_sq.sum())
            if total <= 0.0:
                next_idx = int(rng.integers(len(points)))
            else:
                next_idx = int(rng.choice(len(points), p=min_dist_sq / total))
            chosen.append(next_idx)
            dist_sq = np.sum((points - points[next_idx]) ** 2, axis=1)
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)

        return np.asarray(chosen, dtype=np.int64)
