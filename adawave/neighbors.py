"""Module 2: Neighborhood search.

KDTree-based kNN with index caching.
"""

import numpy as np
from scipy.spatial import cKDTree


class NeighborQuery:
    """Reusable kNN query object with cached KDTree."""

    def __init__(self, xyz: np.ndarray):
        self.xyz = np.ascontiguousarray(xyz, dtype=np.float64)
        self.tree = cKDTree(self.xyz)
        self._cache = {}

    def query_knn(self, k: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """Query k nearest neighbors for every point.

        Parameters
        ----------
        k : int
            Number of neighbors (excluding self).

        Returns
        -------
        distances : (N, k) float64
        indices   : (N, k) int
        """
        if k in self._cache:
            return self._cache[k]

        dists, idxs = self.tree.query(self.xyz, k=k + 1)
        # Exclude self (first column)
        dists = dists[:, 1:]
        idxs = idxs[:, 1:]

        self._cache[k] = (dists, idxs)
        return dists, idxs

    def invalidate_cache(self):
        """Clear cache (call after point positions are updated)."""
        self._cache.clear()

    def rebuild(self, xyz: np.ndarray):
        """Rebuild KDTree with new point positions."""
        self.xyz = np.ascontiguousarray(xyz, dtype=np.float64)
        self.tree = cKDTree(self.xyz)
        self._cache.clear()
