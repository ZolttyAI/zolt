"""Tests for RegressionProbe: quality_score and complexity targets, Pearson r, save/load."""
import tempfile
from pathlib import Path

import pytest
import torch

from z1.probe.regress import (
    RegressionProbe,
    build_regression_suite,
    REGRESSION_TARGETS,
)


DIM = 64
N = 80


def _random_embeddings(n: int, dim: int) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(n, dim)


def _quality_targets(n: int) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.rand(n)  # [0, 1]


def _complexity_targets(n: int) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.rand(n) * 100.0  # [0, 100]


# ── Construction ──────────────────────────────────────────────────────────────

def test_registered_targets():
    assert "quality_score" in REGRESSION_TARGETS
    assert "complexity" in REGRESSION_TARGETS


def test_quality_probe_construction():
    probe = RegressionProbe(input_dim=DIM, target_name="quality_score")
    assert probe.target_name == "quality_score"
    assert probe.output_dim == 1


def test_complexity_probe_construction():
    probe = RegressionProbe(input_dim=DIM, target_name="complexity", loss="huber")
    assert probe.loss_name == "huber"


def test_unknown_target_raises():
    with pytest.raises(ValueError, match="Unknown target"):
        RegressionProbe(input_dim=DIM, target_name="nonexistent_target")


def test_build_regression_suite_keys():
    suite = build_regression_suite(input_dim=DIM)
    assert "quality_score" in suite
    assert "complexity" in suite
    assert all(isinstance(p, RegressionProbe) for p in suite.values())


# ── Training ──────────────────────────────────────────────────────────────────

def test_fit_quality_returns_losses():
    probe = RegressionProbe(input_dim=DIM, target_name="quality_score")
    losses = probe.fit(
        _random_embeddings(N, DIM),
        _quality_targets(N),
        n_epochs=5,
    )
    assert len(losses) == 5
    assert all(l >= 0 for l in losses)


def test_fit_complexity_huber():
    probe = RegressionProbe(input_dim=DIM, target_name="complexity", loss="huber")
    losses = probe.fit(
        _random_embeddings(N, DIM),
        _complexity_targets(N),
        n_epochs=5,
    )
    assert losses[-1] < losses[0] + 1.0  # some reduction expected


# ── Prediction and metric ─────────────────────────────────────────────────────

def test_predict_shape():
    probe = RegressionProbe(input_dim=DIM, target_name="quality_score")
    probe.fit(_random_embeddings(N, DIM), _quality_targets(N), n_epochs=1)
    preds = probe.predict(_random_embeddings(8, DIM))
    assert preds.shape == (8,)


def test_pearson_r_range():
    probe = RegressionProbe(input_dim=DIM, target_name="quality_score")
    emb = _random_embeddings(N, DIM)
    tgt = _quality_targets(N)
    probe.fit(emb, tgt, n_epochs=20, lr=1e-2)
    r = probe.pearson_r(emb, tgt)
    assert -1.0 <= r <= 1.0


def test_pearson_r_perfect_target():
    """A probe trained on labels == mean of embeddings should achieve positive correlation."""
    probe = RegressionProbe(input_dim=DIM, target_name="quality_score")
    emb = _random_embeddings(N, DIM)
    # Target is the mean of each embedding (deterministic mapping from emb)
    tgt = emb.mean(dim=-1)
    tgt = (tgt - tgt.min()) / (tgt.max() - tgt.min() + 1e-8)
    probe.fit(emb, tgt, n_epochs=50, lr=1e-2)
    r = probe.pearson_r(emb, tgt)
    assert r > 0.5


# ── Serialization ─────────────────────────────────────────────────────────────

def test_save_load_roundtrip():
    probe = RegressionProbe(input_dim=DIM, target_name="quality_score")
    probe.fit(_random_embeddings(N, DIM), _quality_targets(N), n_epochs=3)
    emb = _random_embeddings(6, DIM)
    preds_before = probe.predict(emb)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "regress.pt"
        probe.save(path)
        loaded = RegressionProbe.load(path, input_dim=DIM)

    preds_after = loaded.predict(emb)
    assert torch.allclose(preds_before, preds_after, atol=1e-5)
    assert loaded.target_name == "quality_score"
