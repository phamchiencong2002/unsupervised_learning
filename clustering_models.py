from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

class ClusteringModel(ABC):
    """Abstract base class for clustering models."""

    def __init__(self):
        self.model = None
        self.labels_ = None

    @abstractmethod
    def fit_predict(self, pixels):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

class KMeansModel(ClusteringModel):
    def __init__(self, n_clusters=5):
        super().__init__()
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)

    def fit_predict(self, pixels):
        self.labels_ = self.model.fit_predict(pixels)
        return self.labels_
    
    def get_parameters(self):
        return {"n_clusters": self.n_clusters}
    
    def set_n_clusters(self, n_clusters):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)

class GMMModel(ClusteringModel):
    def __init__(self, n_components=5):
        super().__init__()
        self.n_components = n_components
        self.model = GaussianMixture(n_components=self.n_components, random_state=42)

    def fit_predict(self, pixels):
        self.labels_ = self.model.fit_predict(pixels)
        return self.labels_
    
    def get_parameters(self):
        return {"n_components": self.n_components}
    
    def set_n_components(self, n_components):
        self.n_components = n_components
        self.model = GaussianMixture(n_components=self.n_components, random_state=42)
     
class DBSCANModel(ClusteringModel):
    def __init__(self, eps=0.15, min_samples=50, max_fit_pixels=20000):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.max_fit_pixels = max_fit_pixels
    
    def fit_predict(self, pixels):
        n_pixels = len(pixels)
        pixels_norm = pixels.astype(np.float64) / 255.0

        if n_pixels <= self.max_fit_pixels:
            # Small image -> fit DBSCAN directly on all pixels
            labels = self._fit_direct(pixels_norm)
        else:
            # Large image -> subsample strategy
            labels = self.fit_with_subsample(pixels_norm)
        
        # Clean up: remap noise (-1) and make labels contiguous
        labels = self._handle_noise(labels, pixels_norm)
        self.labels_ = labels
        return self.labels_
    
    def _fit_direct(self, pixels_norm):
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return model.fit_predict(pixels_norm)
    
    def _fit_with_subsample(self, pixels_norm):
        # Fit on subsample, then assign ALL pixels to nearest cluster
        n_pixels = len(pixels_norm)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_pixels, self.max_fit_pixels, replace=False)
        sample = pixels_norm[sample_idx]
        
        # Run DBSCAN on the subsample
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        sample_labels = model.fit_predict(sample)

        # Compute center of each cluster (ignoring noise = -1)
        cluster_ids = sorted(set(sample_labels) - {-1})
        if len(cluster_ids) == 0:
            return np.zeros(n_pixels, dtype=int)  # All noise case
        
        centers = np.array([sample[sample_labels == c].mean(axis=0) for c in cluster_ids])

        # Assign all pixels to nearest cluster center
        labels = np.empty(n_pixels, dtype=int)
        batch = 50000
        for i in range(0, n_pixels, batch):
            chunk = pixels_norm[i:i+batch]
            dists = np.linalg.norm(chunk[:, None, :] - centers[None, :, :], axis=2)
            labels[i:i+batch] = np.argmin(dists, axis=1)
        
        return labels
    
    def _handle_noise(self, labels, pixels_norm):
        noise_mark = labels == -1
        n_noise = noise_mark.sum()

        if n_noise == 0 or n_noise == len(labels):
            unique = np.unique(labels)
            mapping = {old: new for new, old in enumerate(unique)}
            return np.array([mapping[l] for l in labels])
        cluster_ids = sorted(set(labels) - {-1})
        centers = np.array([pixels_norm[labels == c].mean(axis=0) for c in cluster_ids])

        noise_pixels = pixels_norm[noise_mark]
        dists = np.linalg.norm(noise_pixels[:, None, :] - centers[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        labels[noise_mark] = [cluster_ids[n] for n in nearest]

        unique = np.unique(labels)
        mapping = {old: new for new, old in enumerate(unique)}
        return np.array([mapping[l] for l in labels])
    
    def get_parameters(self):
        return {"eps": self.eps, "min_samples": self.min_samples, "max_fit_pixels": self.max_fit_pixels}
    
    def set_eps(self, eps):
        self.eps = eps
    
    def set_min_samples(self, min_samples):
        self.min_samples = min_samples
