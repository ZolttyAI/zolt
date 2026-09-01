"""Tests for ClassificationProbe: label set, training, prediction, save/load."""
import tempfile
from pathlib import Path

import pytest
import torch

from zolt.probe.classify import (
    ClassificationProbe,
    DEFAULT_INTENT_LABELS,
    DEFAULT_LANG_LABELS,
    build_label_set,
    LinearProbe,
    MLPProbe,
)


DIM = 64
N = 80


def _random_embeddings(n: int, dim: int) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(n, dim)


def _random_labels(labels: list, n: int) -> list:
    import random
    random.seed(7)
    return [random.choice(labels) for _ in range(n)]


# ── Label set ─────────────────────────────────────────────────────────────────

def test_default_intent_labels_nonempty():
    assert len(DEFAULT_INTENT_LABELS) > 0
    assert "code_generation" in DEFAULT_INTENT_LABELS
    assert "refactor" in DEFAULT_INTENT_LABELS


def test_build_label_set_predefined_only():
    labels = build_label_set(["a", "b", "c"])
    assert labels == ["a", "b", "c"]


def test_build_label_set_with_extra():
    labels = build_label_set(["a", "b"], extra=["c", "d"])
    assert labels == ["a", "b", "c", "d"]


def test_build_label_set_deduplicates():
    labels = build_label_set(["a", "b"], extra=["b", "c"])
    assert labels.count("b") == 1
    assert "a" in labels and "c" in labels


def test_build_label_set_hybrid_predefined_first():
    labels = build_label_set(["x", "y"], extra=["z"])
    assert labels[0] == "x" and labels[1] == "y" and labels[2] == "z"


# ── Probe construction ────────────────────────────────────────────────────────

def test_linear_probe_construction():
    probe = ClassificationProbe(input_dim=DIM, arch="linear")
    assert isinstance(probe.model, LinearProbe)
    assert probe.n_classes == len(DEFAULT_INTENT_LABELS)


def test_mlp_probe_construction():
    probe = ClassificationProbe(input_dim=DIM, arch="mlp")
    assert isinstance(probe.model, MLPProbe)


def test_custom_labels_registered():
    probe = ClassificationProbe(
        input_dim=DIM,
        predefined_labels=["a", "b"],
        extra_labels=["c"],
    )
    assert probe.n_classes == 3
    assert "c" in probe.label2id


# ── Training ──────────────────────────────────────────────────────────────────

def test_fit_returns_loss_list():
    probe = ClassificationProbe(input_dim=DIM, predefined_labels=["a", "b"])
    emb = _random_embeddings(N, DIM)
    lbl = _random_labels(["a", "b"], N)
    losses = probe.fit(emb, lbl, n_epochs=3)
    assert len(losses) == 3
    assert all(isinstance(l, float) for l in losses)


def test_fit_decreases_loss():
    probe = ClassificationProbe(input_dim=DIM, predefined_labels=["a", "b"])
    emb = _random_embeddings(N, DIM)
    lbl = ["a"] * (N // 2) + ["b"] * (N // 2)
    losses = probe.fit(emb, lbl, n_epochs=30, lr=1e-2)
    assert losses[-1] < losses[0]


# ── Prediction ────────────────────────────────────────────────────────────────

def test_predict_returns_label_strings():
    probe = ClassificationProbe(input_dim=DIM, predefined_labels=["x", "y"])
    probe.fit(_random_embeddings(N, DIM), ["x"] * N, n_epochs=1)
    preds = probe.predict(_random_embeddings(5, DIM))
    assert all(p in ["x", "y"] for p in preds)
    assert len(preds) == 5


def test_predict_proba_sums_to_one():
    probe = ClassificationProbe(input_dim=DIM, predefined_labels=["a", "b", "c"])
    probe.fit(_random_embeddings(N, DIM), _random_labels(["a", "b", "c"], N), n_epochs=1)
    probs = probe.predict_proba(_random_embeddings(4, DIM))
    assert probs.shape == (4, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_accuracy_returns_float():
    probe = ClassificationProbe(input_dim=DIM, predefined_labels=["a", "b"])
    probe.fit(_random_embeddings(N, DIM), ["a"] * N, n_epochs=1)
    acc = probe.accuracy(_random_embeddings(10, DIM), ["a"] * 10)
    assert 0.0 <= acc <= 1.0


# ── Serialization ─────────────────────────────────────────────────────────────

def test_save_load_roundtrip():
    probe = ClassificationProbe(input_dim=DIM, predefined_labels=["a", "b"])
    probe.fit(_random_embeddings(N, DIM), ["a"] * N, n_epochs=2)
    emb = _random_embeddings(4, DIM)
    preds_before = probe.predict(emb)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "probe.pt"
        probe.save(path)
        loaded = ClassificationProbe.load(path, input_dim=DIM)

    preds_after = loaded.predict(emb)
    assert preds_before == preds_after
    assert loaded.labels == probe.labels
