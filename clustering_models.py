from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

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
     
class AgglomerativeModel(ClusteringModel):
    def __init__(self, n_clusters=5):
        super().__init__()
        self.n_clusters = n_clusters
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters)

    def fit_predict(self, pixels):
        self.labels_ = self.model.fit_predict(pixels)
        return self.labels_

    def get_parameters(self):
        return {"n_clusters": self.n_clusters}

    def set_n_clusters(self, n_clusters):
        self.n_clusters = n_clusters
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters)