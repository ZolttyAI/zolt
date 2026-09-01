"""
Intersession memory for z1 inference.

Persistent key-value store mapping embedding vectors to text strings.
Stored as a NumPy .npz archive at ~/.z1/memory/session.npz by default.
Retrieval by cosine similarity with configurable top-k and threshold.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


_DEFAULT_PATH = Path.home() / ".z1" / "memory" / "session.npz"
_FORMAT_VERSION = "1"
_MAX_ENTRIES_WARNING = 50_000


class SessionMemory:
    """
    Cosine-similarity key-value memory with NumPy .npz persistence.

    Keys are L2-normalized float32 embedding vectors (dim,).
    Values are UTF-8 text strings.
    Linear scan is used for retrieval; see scale notes in the docstring.

    Scale note: At N=10,000 entries, retrieval costs ~2ms on CPU.
    At N=50,000 entries, ~10ms. Past 50,000 entries, performance degrades
    and max_entries eviction should be configured.
    """

    def __init__(
        self,
        dim: int,
        path: Union[str, Path] = _DEFAULT_PATH,
        max_entries: int = 10_000,
    ):
        self.dim = dim
        self.path = Path(path)
        self.max_entries = max_entries

        # In-memory store
        self._keys: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._values: List[str] = []
        self._timestamps: List[float] = []

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(norm, 1e-12)

    def add(self, embedding: Union[torch.Tensor, np.ndarray], text: str) -> None:
        """
        Add an (embedding, text) pair to memory.
        If max_entries is reached, the oldest entries are evicted first.
        """
        if isinstance(embedding, torch.Tensor):
            vec = embedding.detach().cpu().float().numpy()
        else:
            vec = np.array(embedding, dtype=np.float32)

        if vec.shape[-1] != self.dim:
            raise ValueError(f"Embedding dim {vec.shape[-1]} != expected {self.dim}.")

        vec = self._normalize(vec.reshape(1, self.dim))

        if len(self._values) >= self.max_entries:
            self._evict_oldest(1)

        self._keys = np.concatenate([self._keys, vec], axis=0)
        self._values.append(text)
        self._timestamps.append(time.time())

        if len(self._values) > _MAX_ENTRIES_WARNING:
            import warnings
            warnings.warn(
                f"[SessionMemory] {len(self._values)} entries exceed the recommended "
                f"limit of {_MAX_ENTRIES_WARNING}. Consider using an approximate index "
                f"(e.g. FAISS) for better retrieval performance.",
                UserWarning,
                stacklevel=2,
            )

    def retrieve(
        self,
        embedding: Union[torch.Tensor, np.ndarray],
        top_k: int = 3,
        threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Retrieve top-k most similar entries by cosine similarity.

        Args:
            embedding:  Query embedding vector.
            top_k:      Maximum number of results to return.
            threshold:  Minimum cosine similarity to include in results.

        Returns:
            List of dicts: {text, similarity, index} sorted by descending similarity.
        """
        if len(self._values) == 0:
            return []

        if isinstance(embedding, torch.Tensor):
            vec = embedding.detach().cpu().float().numpy()
        else:
            vec = np.array(embedding, dtype=np.float32)

        vec = self._normalize(vec.reshape(1, self.dim))
        similarities = (self._keys @ vec.T).squeeze(-1)  # (N,)

        # Top-k indices above threshold
        candidates = np.where(similarities >= threshold)[0]
        if len(candidates) == 0:
            return []

        top_idx = candidates[np.argsort(-similarities[candidates])[:top_k]]
        results = []
        for idx in top_idx:
            results.append({
                "text": self._values[int(idx)],
                "similarity": float(similarities[idx]),
                "index": int(idx),
            })
        return results

    def _evict_oldest(self, n: int = 1) -> None:
        """Remove the n oldest entries by insertion timestamp."""
        n = min(n, len(self._values))
        self._keys = self._keys[n:]
        self._values = self._values[n:]
        self._timestamps = self._timestamps[n:]

    def evict_oldest(self, max_entries: int) -> int:
        """
        Trim memory to at most max_entries entries by removing oldest first.
        Returns the number of entries evicted.
        """
        n_current = len(self._values)
        if n_current <= max_entries:
            return 0
        to_remove = n_current - max_entries
        self._evict_oldest(to_remove)
        return to_remove

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist memory to a .npz file. Returns the path written."""
        save_path = Path(path) if path else self.path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        meta = json.dumps({
            "version": _FORMAT_VERSION,
            "dim": self.dim,
            "entry_count": len(self._values),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        np.savez_compressed(
            save_path,
            keys=self._keys,
            values=np.array(self._values, dtype=object),
            timestamps=np.array(self._timestamps, dtype=np.float64),
            meta=np.array([meta]),
        )
        return save_path

    @classmethod
    def load(cls, path: Union[str, Path], max_entries: int = 10_000) -> "SessionMemory":
        """
        Load a SessionMemory from a .npz file.
        Raises FileNotFoundError if the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Session memory file not found: {path}")

        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"][0]))
        dim = int(meta["dim"])

        obj = cls(dim=dim, path=path, max_entries=max_entries)
        obj._keys = data["keys"].astype(np.float32)
        obj._values = list(data["values"])
        obj._timestamps = list(data["timestamps"].astype(np.float64))
        return obj

    @classmethod
    def load_or_create(
        cls,
        dim: int,
        path: Union[str, Path] = _DEFAULT_PATH,
        max_entries: int = 10_000,
    ) -> "SessionMemory":
        """
        Load existing memory from path, or create a new empty store if the file
        does not exist. The most convenient constructor for typical usage.
        """
        try:
            return cls.load(path, max_entries=max_entries)
        except FileNotFoundError:
            return cls(dim=dim, path=path, max_entries=max_entries)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"SessionMemory(entries={len(self)}, dim={self.dim}, path={self.path})"
