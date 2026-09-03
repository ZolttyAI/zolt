"""
Mini-batch K-means clustering over zolt hidden-state embeddings.

Pure PyTorch/NumPy implementation with no sklearn dependency.
Default k=8. Centroids are L2-normalized before distance computation
to operate on cosine geometry, matching how zolt embeddings are used elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class KMeansCluster:
    """
    Mini-batch K-means operating on L2-normalized embedding vectors.

    Cosine similarity is computed as dot product after normalization,
    which is appropriate for LLM embedding spaces.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 5,
        max_iter: int = 100,
        tol: float = 1e-4,
        batch_size: int = 512,
        seed: int = 42,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.seed = seed
        self.centroids: torch.Tensor | None = None  # (k, dim), L2-normalized
        self.inertia_: float = float("inf")
        self.n_iter_: int = 0

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        """L2-normalize along the last dimension."""
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    def _init_centroids(self, x: torch.Tensor) -> torch.Tensor:
        """K-means++ initialization for stable convergence."""
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        n = x.shape[0]
        # Pick first centroid randomly
        first = int(torch.randint(n, (1,), generator=gen).item())
        centroids = [x[first]]

        for _ in range(1, self.n_clusters):
            # Distance from each point to nearest centroid
            stack = torch.stack(centroids, dim=0)  # (k_so_far, dim)
            dists = 1.0 - (x @ stack.T)  # cosine distance, (n, k_so_far)
            min_dists = dists.min(dim=1).values  # (n,)
            min_dists = min_dists.clamp(min=0.0)
            probs = min_dists / (min_dists.sum() + 1e-12)
            chosen = int(torch.multinomial(probs, 1, generator=gen).item())
            centroids.append(x[chosen])

        return self._normalize(torch.stack(centroids, dim=0))

    def _assign(self, x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """Assign each embedding to the nearest centroid via cosine similarity."""
        sims = x @ centroids.T  # (n, k)
        return sims.argmax(dim=-1)

    def fit(self, embeddings: torch.Tensor, verbose: bool = False) -> KMeansCluster:
        """
        Fit K-means on the provided embedding matrix (n_samples, dim).
        Runs n_init random restarts and keeps the best inertia.
        """
        x = self._normalize(embeddings.float())
        n, _dim = x.shape
        best_centroids = None
        best_inertia = float("inf")

        for init_idx in range(self.n_init):
            torch.manual_seed(self.seed + init_idx)
            centroids = self._init_centroids(x)
            last_iter = 0

            for iteration in range(self.max_iter):
                last_iter = iteration
                # Mini-batch assignment
                all_assignments = torch.zeros(n, dtype=torch.long)
                for start in range(0, n, self.batch_size):
                    batch = x[start : start + self.batch_size]
                    sims = torch.mm(batch, centroids.T)
                    all_assignments[start : start + self.batch_size] = sims.argmax(dim=-1)

                # Centroid update
                new_centroids = torch.zeros_like(centroids)
                counts = torch.zeros(self.n_clusters, device=x.device)
                for k in range(self.n_clusters):
                    mask = all_assignments == k
                    if mask.any():
                        new_centroids[k] = x[mask].mean(dim=0)
                        counts[k] = mask.sum().float()
                    else:
                        # Dead cluster: reinitialize to a random point
                        dead_idx = int(torch.randint(n, (1,)).item())
                        new_centroids[k] = x[dead_idx]
                new_centroids = self._normalize(new_centroids)

                shift = (new_centroids - centroids).norm(dim=-1).max().item()
                centroids = new_centroids
                if shift < self.tol:
                    break

            # Inertia: mean squared cosine distance to assigned centroid
            assignments = self._assign(x, centroids)
            assigned_centroids = centroids[assignments]
            inertia = (1.0 - (x * assigned_centroids).sum(dim=-1)).mean().item()

            if verbose:
                print(f"[cluster] init {init_idx + 1}/{self.n_init} inertia={inertia:.6f}")

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.clone()
                self.n_iter_ = last_iter + 1

        self.centroids = best_centroids
        self.inertia_ = best_inertia
        return self

    @torch.no_grad()
    def assign(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return cluster index for each embedding. Shape: (n,)."""
        if self.centroids is None:
            raise RuntimeError("Call fit() before assign().")
        x = self._normalize(embeddings.float())
        return self._assign(x, self.centroids.to(x.device))

    @torch.no_grad()
    def centroid_distances(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine distance from a single embedding to all centroids.
        Returns (n_clusters,) tensor of distances in [0, 2].
        Useful for soft-routing and anomaly detection.
        """
        if self.centroids is None:
            raise RuntimeError("Call fit() before centroid_distances().")
        x = self._normalize(embedding.float().unsqueeze(0))
        sims = (x @ self.centroids.T).squeeze(0)
        return 1.0 - sims  # cosine distance

    def save(self, path: str | Path) -> None:
        """Save centroids and configuration to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "centroids": self.centroids,
                "n_clusters": self.n_clusters,
                "inertia": self.inertia_,
                "n_iter": self.n_iter_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> KMeansCluster:
        """Load a fitted KMeansCluster from disk."""
        data = torch.load(path, map_location="cpu")
        obj = cls(n_clusters=data["n_clusters"])
        obj.centroids = data["centroids"]
        obj.inertia_ = data["inertia"]
        obj.n_iter_ = data["n_iter"]
        return obj


def extract_embeddings(
    model: Any,
    token_batches: list[torch.Tensor],
    device: torch.device,
    active_dim: int | None = None,
    pool: str = "last",
) -> torch.Tensor:
    """
    Extract last-token (or pooled) embeddings from ZoltForCausalLM via model.encode().
    The backbone is not modified (encode() runs under no_grad).

    Args:
        model:         ZoltForCausalLM instance in eval mode.
        token_batches: List of (1, T) or (B, T) token id tensors.
        device:        Device to run on.
        active_dim:    Optional MatFormer slice dimension.
        pool:          'last' (default for causal LM) or 'mean'.

    Returns:
        Tensor of shape (n_samples, embedding_dim).
    """
    embeddings = []
    model.eval()
    with torch.no_grad():
        for tokens in token_batches:
            tokens = tokens.to(device)
            emb = model.encode(tokens, active_dim=active_dim, pool=pool)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)
