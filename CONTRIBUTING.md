# Contributing to Zolt

Thank you for your interest in contributing to **Zolt** — a causal language
model for code generation and reasoning by [ZolttyAI](https://zoltty.ai).

This document covers everything you need to know to open a high-quality
pull request.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Development Environment](#development-environment)
3. [Project Structure](#project-structure)
4. [Branch Naming](#branch-naming)
5. [Commit Conventions](#commit-conventions)
6. [Opening a Pull Request](#opening-a-pull-request)
7. [CI Checklist (run locally first)](#ci-checklist)
8. [Code Style](#code-style)
9. [Testing Guidelines](#testing-guidelines)
10. [Reporting Bugs and Proposing Features](#reporting-bugs-and-proposing-features)
11. [Licensing](#licensing)

---

## Code of Conduct

All contributors are expected to follow our
[Code of Conduct](CODE_OF_CONDUCT.md). Be respectful and constructive.

---

## Development Environment

### Prerequisites

- Python ≥ 3.12
- [`uv`](https://docs.astral.sh/uv/) (fast Python package manager)
- `make`
- PyTorch ≥ 2.2 (CPU build is sufficient for tests)

### Setup

```bash
# Clone the repository
git clone https://github.com/ZolttyAI/zolt.git
cd zolt

# Create virtual environment and install all dependencies
make dev
# Equivalent to:
#   uv venv .venv --python 3.12
#   .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
#   .venv/bin/pip install -e ".[dev]"
```

### Available `make` targets

| Target | Description |
|--------|-------------|
| `make dev` | Full dev setup (venv + torch + editable install) |
| `make lint` | Run `ruff check` + `ruff format --check` |
| `make format` | Auto-format all Python files with `ruff format` |
| `make typecheck` | Run `mypy zolt/ --ignore-missing-imports` |
| `make smoke` | Run `smoke_test.py` (architecture + forward pass validation) |
| `make test` | Run full `pytest` suite |
| `make test-cov` | Run tests with coverage report (HTML + XML) |
| `make ci` | Run all of the above in sequence |

---

## Project Structure

```
zolt/
├── zolt/                   # Main package
│   ├── model.py            # ZoltForCausalLM, SwiGLUFFN, Attention, RMSNorm
│   ├── config.py           # ZoltConfig dataclass
│   ├── train.py            # Training loop with curriculum scheduling
│   ├── eval.py             # Perplexity, code syntax, tag validation helpers
│   ├── rope_scaling.py     # RoPE Linear / NTK context extension
│   ├── data/               # Dataset download, filtering, curriculum, distillation
│   ├── inference/          # Generator, diff format, code verification, DB calls
│   ├── probe/              # Classification, regression, and clustering probes
│   ├── optimize/           # Grid and random hyperparameter search
│   ├── memory/             # Session memory management
│   └── tokenizer/          # BPE tokenizer with special tokens
├── tests/                  # pytest unit tests (mirrors zolt/ structure)
├── scripts/                # Utility scripts (debug_train.py, etc.)
├── smoke_test.py           # Quick architecture validation (no data needed)
├── Makefile                # Automation targets
├── pyproject.toml          # Package metadata and dev dependencies
└── ruff.toml               # Linter and formatter configuration
```

---

## Branch Naming

Use the following prefixes:

| Prefix | Use for |
|--------|---------|
| `feat/` | New features or capabilities |
| `fix/` | Bug fixes |
| `chore/` | Maintenance, dependency updates, CI changes |
| `docs/` | Documentation-only changes |
| `refactor/` | Code restructuring with no behavior change |
| `perf/` | Performance improvements |
| `test/` | Adding or updating tests |

**Examples:**
```
feat/add-flash-attention
fix/rope-scaling-ntk-factor
chore/update-ruff-1-2-0
docs/architecture-diagram
```

Branch names should be lowercase, hyphen-separated, and descriptive.

---

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/).

**Format:**
```
<type>(<scope>): <short description>

[optional body]

[optional footer(s)]
```

**Types:**

| Type | When to use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `ci` | CI/CD pipeline changes |
| `docs` | Documentation only |
| `refactor` | Code change with no behavior change |
| `perf` | Performance improvement |
| `test` | Adding or fixing tests |
| `chore` | Dependency bumps, tooling |
| `build` | Build system changes |

**Examples:**
```
feat(inference): add streaming token callback to ZoltGenerator
fix(model): guard hidden_dim None in SwiGLUFFN init
ci: add mypy --strict flag to typecheck job
docs(contributing): add branch naming section
test(probe): add pearson-r edge case for constant output
```

Breaking changes must include `BREAKING CHANGE:` in the footer:
```
feat(tokenizer)!: rename special token <FILL> to <infill>

BREAKING CHANGE: existing tokenizer checkpoints are incompatible;
regenerate with `make tokenizer`.
```

---

## Opening a Pull Request

1. **Fork** the repository and create your branch from `main`.
2. Make your changes following the guidelines below.
3. **Run `make ci` locally** — all checks must pass before opening a PR.
4. Open a PR against `main`.
5. Fill in the [PR template](.github/PULL_REQUEST_TEMPLATE.md) completely.
6. Request review from at least one maintainer.

**PR size guidance:**

- Prefer small, focused PRs. Large PRs take longer to review and are harder
  to bisect if a regression is introduced.
- If your change is large, consider splitting it into a series of incremental PRs.
- Draft PRs are welcome for early feedback before a change is complete.

---

## CI Checklist

Before opening a PR, run the full CI suite locally:

```bash
make ci
```

This runs in order:

```
lint       → ruff check + ruff format --check
typecheck  → mypy zolt/ --ignore-missing-imports
smoke      → python smoke_test.py
test-cov   → pytest tests/ --cov=zolt --cov-report=term-missing
```

All steps must pass with **exit code 0**.

### Running individual checks

```bash
# Lint only
make lint

# Auto-fix formatting
make format

# Type checking only
make typecheck

# Specific test file
.venv/bin/python -m pytest tests/test_model.py -v

# Specific test by name
.venv/bin/python -m pytest tests/ -k "test_encode_default_is_last" -v

# Run with verbose coverage
make test-cov
```

---

## Code Style

All Python code is checked with **ruff** (linter + formatter).
Configuration lives in [`ruff.toml`](ruff.toml).

Key rules enforced:

| Rule set | What it checks |
|----------|---------------|
| `E`, `W` | PEP 8 errors and warnings |
| `F` | Pyflakes (unused imports, undefined names) |
| `I` | Import order (isort-compatible) |
| `B` | Bugbear (common bugs and design issues) |
| `C4` | Comprehension style |
| `UP` | Pyupgrade (modern Python idioms) |
| `SIM` | Code simplification |
| `RUF` | Ruff-specific rules |

**Quick style rules:**
- Line length: **100 characters**
- Use `f-strings` for string interpolation (not `.format()` or `%`)
- Use `pathlib.Path` instead of `os.path` for file operations
- Never shadow argument names by reassigning to `Path(arg)` — use a new
  variable name (e.g., `ckpt_path = Path(checkpoint)`)
- Annotate all public function signatures with types
- Write docstrings for all public classes and non-trivial functions

### Type annotations

We use **mypy** for static type checking. All new code should:

- Annotate function signatures fully
- Avoid `Any` where a more precise type is possible
- Use `X | None` (Python 3.10+ union syntax), not `Optional[X]`
- Use `X | Y` instead of `Union[X, Y]`

---

## Testing Guidelines

Tests live in `tests/` and mirror the `zolt/` package structure.

- **Every new public function must have at least one test.**
- Tests should be fast — avoid loading full model weights or running GPU ops.
  Use small configs (e.g., `dim=64, n_layers=2`) for structural tests.
- Use `pytest.mark.parametrize` for testing multiple inputs.
- Mock external I/O (filesystem, network) where possible.
- Name tests descriptively: `test_<what>_<condition>_<expected>`.

```python
# Good
def test_encode_default_pool_is_last():
    ...

def test_encode_mean_pool_differs_from_last():
    ...

# Avoid
def test_encode():
    ...
```

Test file naming:
- `tests/test_model.py` → tests for `zolt/model.py`
- `tests/test_probe_classify.py` → tests for `zolt/probe/classify.py`

---

## Reporting Bugs and Proposing Features

Use GitHub Issues with the appropriate template:

- 🐛 **Bug report** — something is broken or behaving incorrectly
- ✨ **Feature request** — propose a new capability or improvement

For security vulnerabilities, do **not** open a public issue.
See [SECURITY.md](SECURITY.md) for the responsible disclosure process.

Before opening an issue, please search existing issues and discussions to
avoid duplicates.

---

## Licensing

By submitting a contribution, you agree that your work is licensed under the
**Apache License 2.0**, as described in the project's [`LICENSE`](LICENSE)
file. No separate CLA or signature is required.
