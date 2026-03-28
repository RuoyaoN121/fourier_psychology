"""
Spectral Sabermetrics — Part 3a: Spectral Team Matching
========================================================
Groups developers into strictly balanced teams of 4 using a two-phase
approach:

1. **Spectral embedding** — Compute a Laplacian spectral embedding of the
   RBF affinity matrix to capture non-linear rhythm similarity.
2. **Greedy constrained allocation** — Instead of unconstrained k-means
   (which produces wildly imbalanced clusters), iteratively assign each
   unallocated developer to the nearest cluster centroid that still has
   capacity, enforcing exactly ``team_size`` members per team.

Features are pre-processed with ``StandardScaler`` followed by L2
normalisation so that all frequency bands contribute equally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.csgraph import laplacian

logger = logging.getLogger(__name__)


@dataclass
class TeamAssignment:
    """Result of the spectral clustering."""
    labels: np.ndarray               # cluster label per developer
    teams: Dict[int, List[str]]      # cluster_id → list of developer names
    n_clusters: int
    developer_names: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Spectral embedding (manual, so we can reuse the affinity matrix)
# ---------------------------------------------------------------------------

def _spectral_embed(
    X: np.ndarray,
    n_components: int,
    gamma: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (embedding, affinity_matrix) of shape (n, n_components) and (n, n).

    Uses the normalised graph Laplacian (symmetric) and keeps the
    ``n_components`` smallest non-trivial eigenvectors.
    """
    A = rbf_kernel(X, gamma=gamma)           # (n, n) affinity
    L = laplacian(A, normed=True)            # normalised Laplacian

    # Eigen-decomposition (L is symmetric → use eigh for stability)
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Skip the first eigenvector (constant Fiedler vector for connected graphs)
    # and take the next n_components
    embedding = eigenvectors[:, 1: n_components + 1]

    # Row-normalise the embedding (common in spectral clustering)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embedding = embedding / norms

    return embedding, A


# ---------------------------------------------------------------------------
# Greedy constrained allocation
# ---------------------------------------------------------------------------

def _greedy_balanced_assign(
    embedding: np.ndarray,
    n_clusters: int,
    team_size: int,
    random_state: int = 42,
) -> np.ndarray:
    """Assign each point to one of ``n_clusters`` groups with capacity
    ``team_size``, greedily by closest centroid.

    Algorithm
    ---------
    1. Initialise centroids via k-means++ seeding on the spectral embedding.
    2. Repeat until convergence (or max iterations):
       a. **Assignment** — Sort all (point, centroid) distances.  Walk the
          sorted list and assign each point to its closest centroid that
          still has room (< team_size members).
       b. **Update** — Recompute centroids from current assignments.
    3. Return labels array.
    """
    rng = np.random.default_rng(random_state)
    n = len(embedding)

    # --- k-means++ initialisation ---
    centroids = np.empty((n_clusters, embedding.shape[1]))
    first = rng.integers(n)
    centroids[0] = embedding[first]
    for k in range(1, n_clusters):
        dists = np.min(
            np.linalg.norm(embedding[:, None, :] - centroids[None, :k, :], axis=2),
            axis=1,
        )
        probs = dists ** 2
        probs /= probs.sum()
        centroids[k] = embedding[rng.choice(n, p=probs)]

    labels = np.full(n, -1, dtype=int)
    max_iter = 50

    for _ in range(max_iter):
        old_labels = labels.copy()

        # -- Constrained assignment --
        # Distance matrix: (n, n_clusters)
        dist_matrix = np.linalg.norm(
            embedding[:, None, :] - centroids[None, :, :], axis=2
        )
        # Build a flat list of (distance, point_idx, cluster_idx) sorted by distance
        flat = [
            (dist_matrix[i, k], i, k)
            for i in range(n) for k in range(n_clusters)
        ]
        flat.sort(key=lambda t: t[0])

        labels[:] = -1
        capacity = np.full(n_clusters, team_size, dtype=int)
        assigned = np.zeros(n, dtype=bool)

        for _, pt, cl in flat:
            if assigned[pt]:
                continue
            if capacity[cl] <= 0:
                continue
            labels[pt] = cl
            capacity[cl] -= 1
            assigned[pt] = True

        # Any remaining stragglers (shouldn't happen if n_clusters * team_size >= n)
        for pt in np.where(~assigned)[0]:
            best = int(np.argmin(dist_matrix[pt]))
            labels[pt] = best

        # -- Centroid update --
        for k in range(n_clusters):
            members = embedding[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)

        if np.array_equal(labels, old_labels):
            break

    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def spectral_cluster_teams(
    feature_matrix: np.ndarray,
    developer_names: List[str],
    team_size: int = 4,
    random_state: int = 42,
) -> TeamAssignment:
    """Partition developers into strictly balanced teams.

    Pipeline:
    1. StandardScaler → L2 normalise features.
    2. Spectral embedding via normalised graph Laplacian.
    3. Greedy constrained allocation (exactly ``team_size`` per team).

    Parameters
    ----------
    feature_matrix : np.ndarray
        Shape ``(n_developers, n_features)``.
    developer_names : list[str]
        Names corresponding to rows.
    team_size : int
        Exact members per team.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    TeamAssignment
    """
    n = len(developer_names)
    n_clusters = max(1, int(np.ceil(n / team_size)))

    logger.info("Clustering %d developers into %d teams (exact size %d).",
                n, n_clusters, team_size)

    # ── Pre-processing: scale then L2-normalise ──
    X_scaled = StandardScaler().fit_transform(feature_matrix)
    X_normed = normalize(X_scaled, norm="l2")

    # ── Spectral embedding ──
    embedding, _ = _spectral_embed(X_normed, n_components=n_clusters)

    # ── Constrained greedy assignment ──
    labels = _greedy_balanced_assign(
        embedding, n_clusters, team_size, random_state=random_state,
    )

    # ── Build teams dict ──
    teams: Dict[int, List[str]] = {}
    for idx, label in enumerate(labels):
        teams.setdefault(int(label), []).append(developer_names[idx])

    return TeamAssignment(
        labels=labels,
        teams=teams,
        n_clusters=n_clusters,
        developer_names=developer_names,
    )
