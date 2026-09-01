"""Tests for grid_search and random_search."""
import json
import tempfile
from pathlib import Path

import pytest

from zolt.optimize.search import (
    Trial,
    SearchResult,
    grid_search,
    random_search,
    _grid_combinations,
    _sample_params,
)
import random


# ── SearchResult ──────────────────────────────────────────────────────────────

def test_search_result_minimize_best():
    sr = SearchResult(direction="minimize")
    sr.add({"lr": 0.1}, 0.5)
    sr.add({"lr": 0.01}, 0.2)
    sr.add({"lr": 0.001}, 0.8)
    assert sr.best.score == 0.2
    assert sr.best.params["lr"] == 0.01


def test_search_result_maximize_best():
    sr = SearchResult(direction="maximize")
    sr.add({"k": 4}, 0.5)
    sr.add({"k": 8}, 0.9)
    sr.add({"k": 16}, 0.7)
    assert sr.best.score == 0.9


def test_search_result_top_k():
    sr = SearchResult(direction="minimize")
    for i, score in enumerate([0.5, 0.1, 0.8, 0.3]):
        sr.add({"i": i}, score)
    top = sr.top_k(2)
    assert len(top) == 2
    assert top[0].rank == 1
    assert top[0].score == 0.1


def test_search_result_save_load_roundtrip():
    sr = SearchResult(direction="minimize")
    sr.add({"lr": 0.1}, 0.5)
    sr.add({"lr": 0.01}, 0.2)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "search.json"
        sr.save(path)
        loaded = SearchResult.load(path)

    assert loaded.direction == "minimize"
    assert loaded.best.score == 0.2
    assert len(loaded.trials) == 2


# ── Grid combinations ─────────────────────────────────────────────────────────

def test_grid_combinations_cartesian():
    grid = {"lr": [0.1, 0.01], "wd": [0.0, 1e-4]}
    combos = list(_grid_combinations(grid))
    assert len(combos) == 4
    lrs = {c["lr"] for c in combos}
    assert lrs == {0.1, 0.01}


def test_grid_combinations_single_param():
    combos = list(_grid_combinations({"k": [4, 8, 16]}))
    assert len(combos) == 3


# ── Grid search ───────────────────────────────────────────────────────────────

def test_grid_search_finds_minimum():
    eval_fn = lambda p: abs(p["x"] - 3.0)
    grid = {"x": [0.0, 1.0, 2.0, 3.0, 4.0]}
    result = grid_search(eval_fn, grid, direction="minimize")
    assert result.best.params["x"] == 3.0
    assert result.best.score == 0.0


def test_grid_search_evaluates_all_combinations():
    calls = []
    def eval_fn(p):
        calls.append(p)
        return 1.0
    grid = {"a": [1, 2], "b": [10, 20]}
    grid_search(eval_fn, grid)
    assert len(calls) == 4


def test_grid_search_handles_eval_exception():
    def bad_fn(p):
        if p["x"] == 2:
            raise RuntimeError("oops")
        return float(p["x"])
    result = grid_search(bad_fn, {"x": [1, 2, 3]}, direction="minimize")
    # x=2 should be recorded with an error; best should be x=1 (score=1.0)
    errored = [t for t in result.trials if t.error is not None]
    assert len(errored) == 1
    assert errored[0].params["x"] == 2


# ── Random search ─────────────────────────────────────────────────────────────

def test_random_search_n_trials():
    calls = []
    def eval_fn(p):
        calls.append(p)
        return p["lr"]
    space = {"lr": (0.0, 1.0, float)}
    random_search(eval_fn, space, n_trials=15, seed=0)
    assert len(calls) == 15


def test_random_search_reproducible():
    eval_fn = lambda p: p["lr"]
    space = {"lr": (0.0, 1.0, float)}
    r1 = random_search(eval_fn, space, n_trials=10, seed=99)
    r2 = random_search(eval_fn, space, n_trials=10, seed=99)
    scores1 = [t.score for t in r1.trials]
    scores2 = [t.score for t in r2.trials]
    assert scores1 == scores2


def test_random_search_int_param():
    values_seen = set()
    def eval_fn(p):
        values_seen.add(p["k"])
        return float(p["k"])
    space = {"k": (2, 8, int)}
    random_search(eval_fn, space, n_trials=20, seed=0)
    assert all(2 <= v <= 8 for v in values_seen)


def test_random_search_list_param():
    seen = set()
    eval_fn = lambda p: seen.add(p["arch"]) or 0.0
    space = {"arch": ["linear", "mlp"]}
    random_search(eval_fn, space, n_trials=30, seed=0)
    assert "linear" in seen and "mlp" in seen


def test_sample_params_callable():
    rng = random.Random(0)
    space = {"val": lambda: 42}
    params = _sample_params(space, rng)
    assert params["val"] == 42
