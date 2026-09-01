"""Tests for SessionMemory: add, retrieve, eviction, persistence, load_or_create."""
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

from z1.memory.session import SessionMemory


DIM = 64


def _vec(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    v = torch.randn(DIM)
    return v / v.norm()


# ── Basic operations ──────────────────────────────────────────────────────────

def test_empty_memory_has_zero_len():
    mem = SessionMemory(dim=DIM)
    assert len(mem) == 0


def test_add_single_entry():
    mem = SessionMemory(dim=DIM)
    mem.add(_vec(0), "hello world")
    assert len(mem) == 1


def test_add_numpy_embedding():
    mem = SessionMemory(dim=DIM)
    v = np.random.randn(DIM).astype(np.float32)
    mem.add(v, "numpy entry")
    assert len(mem) == 1


def test_add_wrong_dim_raises():
    mem = SessionMemory(dim=DIM)
    with pytest.raises(ValueError, match="dim"):
        mem.add(torch.randn(DIM + 1), "bad dim")


# ── Retrieval ─────────────────────────────────────────────────────────────────

def test_retrieve_empty_returns_empty_list():
    mem = SessionMemory(dim=DIM)
    results = mem.retrieve(_vec(0), top_k=3)
    assert results == []


def test_retrieve_exact_match():
    mem = SessionMemory(dim=DIM)
    v = _vec(42)
    mem.add(v, "target text")
    results = mem.retrieve(v, top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "target text"
    assert results[0]["similarity"] > 0.99


def test_retrieve_top_k_limit():
    mem = SessionMemory(dim=DIM)
    for i in range(10):
        mem.add(_vec(i), f"entry {i}")
    results = mem.retrieve(_vec(0), top_k=3)
    assert len(results) <= 3


def test_retrieve_sorted_by_similarity():
    mem = SessionMemory(dim=DIM)
    query = _vec(0)
    for i in range(5):
        mem.add(_vec(i), f"entry {i}")
    results = mem.retrieve(query, top_k=5)
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


def test_retrieve_threshold_filters():
    mem = SessionMemory(dim=DIM)
    # Add a dissimilar entry (opposite direction)
    mem.add(-_vec(0), "opposite")
    mem.add(_vec(0), "same direction")
    results = mem.retrieve(_vec(0), top_k=5, threshold=0.5)
    texts = [r["text"] for r in results]
    assert "same direction" in texts
    assert "opposite" not in texts


# ── Eviction ──────────────────────────────────────────────────────────────────

def test_max_entries_evicts_oldest():
    mem = SessionMemory(dim=DIM, max_entries=3)
    for i in range(5):
        mem.add(_vec(i), f"entry {i}")
    assert len(mem) == 3
    # Oldest entries (0, 1) should be gone; only 2, 3, 4 remain
    results = mem.retrieve(_vec(4), top_k=10)
    texts = [r["text"] for r in results]
    assert "entry 4" in texts


def test_evict_oldest_explicit():
    mem = SessionMemory(dim=DIM)
    for i in range(8):
        mem.add(_vec(i), f"entry {i}")
    removed = mem.evict_oldest(max_entries=5)
    assert removed == 3
    assert len(mem) == 5


def test_evict_oldest_no_op_when_under_limit():
    mem = SessionMemory(dim=DIM)
    for i in range(3):
        mem.add(_vec(i), f"entry {i}")
    removed = mem.evict_oldest(max_entries=5)
    assert removed == 0
    assert len(mem) == 3


# ── Persistence ───────────────────────────────────────────────────────────────

def test_save_load_roundtrip():
    mem = SessionMemory(dim=DIM)
    for i in range(5):
        mem.add(_vec(i), f"stored {i}")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "session.npz"
        mem.save(path)
        loaded = SessionMemory.load(path)

    assert len(loaded) == 5
    results = loaded.retrieve(_vec(0), top_k=1)
    assert results[0]["text"] == "stored 0"


def test_load_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        SessionMemory.load("/nonexistent/path/session.npz")


def test_load_or_create_creates_new_when_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "new_session.npz"
        mem = SessionMemory.load_or_create(dim=DIM, path=path)
    assert len(mem) == 0
    assert mem.dim == DIM


def test_load_or_create_loads_existing():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "session.npz"
        mem = SessionMemory(dim=DIM, path=path)
        mem.add(_vec(0), "persisted")
        mem.save(path)

        loaded = SessionMemory.load_or_create(dim=DIM, path=path)
        assert len(loaded) == 1
        assert loaded.retrieve(_vec(0), top_k=1)[0]["text"] == "persisted"


def test_repr_string():
    mem = SessionMemory(dim=DIM)
    r = repr(mem)
    assert "SessionMemory" in r
    assert "dim=64" in r
