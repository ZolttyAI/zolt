"""
Grid and random hyperparameter search.

Dependency-free implementation operating over user-supplied eval functions.
Supports: grid search, random search, result serialization to JSON.
"""
from __future__ import annotations

import json
import math
import random
import itertools
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union


@dataclass
class Trial:
    """Result of a single hyperparameter evaluation."""
    params: Dict[str, Any]
    score: float
    rank: int = 0
    error: Optional[str] = None


@dataclass
class SearchResult:
    """Aggregated search results, sortable by score."""
    trials: List[Trial] = field(default_factory=list)
    best: Optional[Trial] = None
    direction: str = "minimize"

    def _is_better(self, a: float, b: float) -> bool:
        return a < b if self.direction == "minimize" else a > b

    def add(self, params: Dict[str, Any], score: float, error: Optional[str] = None) -> None:
        trial = Trial(params=params, score=score, error=error)
        self.trials.append(trial)
        if self.best is None or self._is_better(score, self.best.score):
            self.best = trial
        # Re-rank by score
        sorted_trials = sorted(
            self.trials,
            key=lambda t: t.score,
            reverse=(self.direction == "maximize"),
        )
        for rank, t in enumerate(sorted_trials, 1):
            t.rank = rank

    def top_k(self, k: int) -> List[Trial]:
        """Return the k best trials sorted by rank."""
        return sorted(self.trials, key=lambda t: t.rank)[:k]

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "direction": self.direction,
            "best": asdict(self.best) if self.best else None,
            "trials": [asdict(t) for t in self.trials],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SearchResult":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        obj = cls(direction=data["direction"])
        for t in data["trials"]:
            trial = Trial(**t)
            obj.trials.append(trial)
            if obj.best is None or obj._is_better(trial.score, obj.best.score):
                obj.best = trial
        return obj


# ── Grid Search ──────────────────────────────────────────────────────────────

def _grid_combinations(param_grid: Dict[str, List[Any]]) -> Generator[Dict[str, Any], None, None]:
    """Yield all parameter combinations from a grid spec."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def grid_search(
    eval_fn: Callable[[Dict[str, Any]], float],
    param_grid: Dict[str, List[Any]],
    direction: str = "minimize",
    verbose: bool = False,
) -> SearchResult:
    """
    Exhaustive grid search over all combinations in param_grid.
    eval_fn(params) -> float (loss or score).
    direction: 'minimize' or 'maximize'.
    """
    result = SearchResult(direction=direction)
    combos = list(_grid_combinations(param_grid))
    total = len(combos)

    for i, params in enumerate(combos, 1):
        error = None
        try:
            score = eval_fn(params)
        except Exception as e:
            score = float("inf") if direction == "minimize" else float("-inf")
            error = str(e)
        result.add(params, score, error)
        if verbose:
            print(f"[grid-search] {i}/{total} params={params} score={score:.6f}")

    return result


# ── Random Search ─────────────────────────────────────────────────────────────

def _sample_params(
    param_space: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Sample one configuration from param_space.
    Each value is either:
      - A list:   sample uniformly.
      - A tuple (lo, hi, type):  sample uniform float or int.
      - A callable: call with no args to generate a value.
    """
    sampled: Dict[str, Any] = {}
    for key, spec in param_space.items():
        if isinstance(spec, list):
            sampled[key] = rng.choice(spec)
        elif isinstance(spec, tuple) and len(spec) == 3:
            lo, hi, kind = spec
            if kind == int:
                sampled[key] = rng.randint(int(lo), int(hi))
            else:
                sampled[key] = lo + rng.random() * (hi - lo)
        elif callable(spec):
            sampled[key] = spec()
        else:
            sampled[key] = spec
    return sampled


def random_search(
    eval_fn: Callable[[Dict[str, Any]], float],
    param_space: Dict[str, Any],
    n_trials: int = 20,
    direction: str = "minimize",
    seed: int = 42,
    verbose: bool = False,
) -> SearchResult:
    """
    Random search over param_space for n_trials evaluations.
    Finds near-optimal configs in ~20% of the evaluations of an equivalent grid
    for 4+ dimensional spaces (Bergstra & Bengio 2012).
    """
    result = SearchResult(direction=direction)
    rng = random.Random(seed)

    for i in range(1, n_trials + 1):
        params = _sample_params(param_space, rng)
        error = None
        try:
            score = eval_fn(params)
        except Exception as e:
            score = float("inf") if direction == "minimize" else float("-inf")
            error = str(e)
        result.add(params, score, error)
        if verbose:
            print(f"[random-search] trial {i}/{n_trials} params={params} score={score:.6f}")

    return result
