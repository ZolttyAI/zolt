"""Tests for KMeansCluster: fit, assign, centroid_distances, save/load."""
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

from z1.probe.cluster import KMeansCluster


DIM = 32
N = 200


def _clustered_data(k: int, n_per: int, dim: int, seed: int = 42) -> torch.Tensor:
    """Generate clearly separable cluster data."""
    torch.manual_seed(seed)
    parts = []
    for c in range(k):
        center = torch.zeros(dim)
        center[c % dim] = 5.0 * (c + 1)
        cluster = center + torch.randn(n_per, dim) * 0.1
        parts.append(cluster)
    return torch.cat(parts, dim=0)


# ── Fit ───────────────────────────────────────────────────────────────────────

def test_fit_returns_self():
    km = KMeansCluster(n_clusters=4, n_init=2, max_iter=20)
    emb = _clustered_data(4, N // 4, DIM)
    result = km.fit(emb)
    assert result is km


def test_fit_produces_centroids():
    km = KMeansCluster(n_clusters=4, n_init=2)
    km.fit(_clustered_data(4, N // 4, DIM))
    assert km.centroids is not None
    assert km.centroids.shape == (4, DIM)


def test_fit_default_k_8():
    km = KMeansCluster()
    assert km.n_clusters == 8
    km.fit(_clustered_data(8, 25, DIM))
    assert km.centroids.shape[0] == 8


def test_inertia_decreases_with_separable_data():
    km_good = KMeansCluster(n_clusters=4, n_init=3, max_iter=50)
    emb = _clustered_data(4, 50, DIM)
    km_good.fit(emb)
    assert km_good.inertia_ < 0.5  # Well-separated clusters have low cosine distance


# ── Assign ────────────────────────────────────────────────────────────────────

def test_assign_returns_correct_shape():
    km = KMeansCluster(n_clusters=4, n_init=2)
    emb = _clustered_data(4, 50, DIM)
    km.fit(emb)
    assignments = km.assign(emb)
    assert assignments.shape == (200,)
    assert assignments.max().item() < 4
    assert assignments.min().item() >= 0


def test_assign_separable_clusters_correct():
    """Fit on 4 well-separated clusters; each point should map to its own cluster."""
    km = KMeansCluster(n_clusters=4, n_init=5, max_iter=100, seed=0)
    emb = _clustered_data(4, 50, DIM, seed=0)
    km.fit(emb)
    assignments = km.assign(emb)
    # All 50 points in each cluster should get the same label
    for c in range(4):
        chunk = assignments[c * 50 : (c + 1) * 50]
        assert chunk.unique().shape[0] == 1, f"Cluster {c} not cleanly assigned"


def test_assign_before_fit_raises():
    km = KMeansCluster(n_clusters=4)
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        km.assign(torch.randn(10, DIM))


# ── Centroid distances ────────────────────────────────────────────────────────

def test_centroid_distances_shape():
    km = KMeansCluster(n_clusters=4, n_init=2)
    km.fit(_clustered_data(4, 50, DIM))
    dists = km.centroid_distances(torch.randn(DIM))
    assert dists.shape == (4,)


def test_centroid_distances_range():
    km = KMeansCluster(n_clusters=4, n_init=2)
    km.fit(_clustered_data(4, 50, DIM))
    dists = km.centroid_distances(torch.randn(DIM))
    assert dists.min().item() >= -0.01  # Cosine distance in [0, 2]
    assert dists.max().item() <= 2.01


def test_centroid_distances_before_fit_raises():
    km = KMeansCluster(n_clusters=4)
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        km.centroid_distances(torch.randn(DIM))


# ── Save/load ─────────────────────────────────────────────────────────────────

def test_save_load_roundtrip():
    km = KMeansCluster(n_clusters=4, n_init=2)
    emb = _clustered_data(4, 50, DIM)
    km.fit(emb)
    assignments_before = km.assign(emb)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "kmeans.pt"
        km.save(path)
        km2 = KMeansCluster.load(path)

    assignments_after = km2.assign(emb)
    assert torch.equal(assignments_before, assignments_after)
    assert km2.n_clusters == 4
