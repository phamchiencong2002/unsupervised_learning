from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift

class ClusteringModel(ABC):
    """Abstract base class for clustering models."""

    def __init__(self):
        self.model = None
        self.labels_ = None

    @abstractmethod
    def fit_predict(self, pixels):
        """Apply clustering and return labels."""
        pass

    @abstractmethod
    def get_parameters(self):
        """Return dict of adjustable parameters."""
        pass

class KMeansModel(ClusteringModel):
    """K-Means clustering model."""

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

class MeanShiftModel(ClusteringModel):
    """Mean Shift clustering models."""

    def __init__(self, bandwidth=50):
        super().__init__()
        self.bandwidth = bandwidth
        self.model = MeanShift(bandwidth=self.bandwidth)

    def fit_predict(self, pixels):
        self.labels_ = self.model.fit_predict(pixels)
        return self.labels_
    
    def get_parameters(self):
        return {"bandwidth": self.bandwidth}
    
    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        self.model = MeanShift(bandwidth=self.bandwidth)

class DBSCANModel(ClusteringModel):
    """DBSCAN clustering model"""
    
    def __init__(self, eps=10, min_samples=5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
    
    def fit_predict(self, pixels):
        """Apply DBSCAN clustering"""
        self.labels_ = self.model.fit_predict(pixels)
        return self.labels_
    
    def get_parameters(self):
        return {"eps": self.eps, "min_samples": self.min_samples}
    
    def set_parameters(self, eps, min_samples):
        """Update DBSCAN parameters"""
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)