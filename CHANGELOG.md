# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Professional CI pipeline** — 4 GitHub Actions workflows:
  `ci.yml` (lint + typecheck + test matrix + debug train),
  `release.yml` (OIDC PyPI publish + GitHub Release),
  `security.yml` (daily pip-audit + bandit + trivy),
  `nightly.yml` (weekly regression training loop).
- **Ruff linter and formatter** (`ruff.toml`) with rules `E`, `W`, `F`, `I`,
  `B`, `C4`, `UP`, `SIM`, `RUF`; 100-character line limit; Python 3.12 target.
- **Mypy static type checking** across the entire `zolt/` package.
- **`make` targets**: `lint`, `format`, `typecheck`, `test-cov`, `ci`.
- **Repository governance files**: `CODE_OF_CONDUCT.md` (Contributor Covenant
  v2.1), `SECURITY.md` (vulnerability disclosure policy), `CODEOWNERS`.
- **Contributor workflow files**: `CONTRIBUTING.md` (full rewrite),
  `.github/PULL_REQUEST_TEMPLATE.md`, bug-report and feature-request issue
  templates, `ARCHITECTURE.md`.
- **`encode()` pooling corrected** — default pool strategy changed from
  `'mean'` to `'last'` for causal decoder-only models; `'mean'` remains
  available as an explicit opt-in.
- **MatFormer sub-network slicing** — `SwiGLUFFN` and `Attention` support
  `active_dim` parameter for nested sub-network extraction.
- **Probe modules** — `ClassificationProbe`, `RegressionProbe`, `KMeansCluster`
  with save/load, Pearson-r scoring, and cosine-distance K-means++.
- **Session memory** — `SessionMemory` with rolling window and optional
  persistence.
- **Diff format parser** — `<search>/<replace>` block parsing for code editing.
- **Self-correcting code generation** — `verify_base.py` shared loop used by
  Python, JavaScript, and TypeScript verifiers.
- **Structured DB call schema** — `<db_call>` / `</db_call>` with JSON schema
  validation.
- **`zolt-mini` preset** — 109.5M-parameter sub-network extractable from the
  same 250.9M checkpoint via MatFormer slicing.

### Changed

- Repository renamed from `z1` → `zolt`; brand `zone.ai` → `zolt.ai`.
- All Python source files reformatted with `ruff format`.
- `mix_datasets()` signature now includes explicit `seed: int = 42` parameter.
- `extract_code_block` re-exported from `verify_ts` for backward compatibility.

### Fixed

- `hidden_dim` and `n_kv_heads` `None`-guard in `SwiGLUFFN` and `Attention`
  `__init__` (mypy `int | None` safety).
- `Path` variable shadowing in `rope_scaling.py`, `eval.py`, `data/pipeline.py`,
  and `data/download.py`.
- Tensor index `int()` cast in `KMeansCluster._init_centroids` and dead-cluster
  reinitialization.
- `n_tokens` cast to `int` in `eval.compute_perplexity`.
- `tok_id` cast to `int` in `ZoltGenerator.stream`.

---

## [0.1.0] — 2026-08-30

### Added

- Initial public release of the `zolt` package.
- `ZoltForCausalLM` decoder-only transformer with:
  - RMSNorm, SwiGLU FFN, Grouped-Query Attention (GQA).
  - Rotary Position Embeddings (RoPE) with Linear and NTK-aware scaling.
  - MatFormer-compatible nested sub-network dimension slicing.
- `ZoltConfig` dataclass with `zolt` (250.9M) and `zolt-mini` (109.5M) presets.
- `ZoltTokenizer` with 23 functional special tokens.
- Causal training loop with cosine LR, gradient clipping, and checkpoint saving.
- Data pipeline: download, filter, curriculum staging, distillation mixing.
- `smoke_test.py` for quick architecture validation.
- `scripts/debug_train.py` for a fast CPU training sanity check.
- `pytest` unit test suite.

[Unreleased]: https://github.com/ZolttyAI/zolt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ZolttyAI/zolt/releases/tag/v0.1.0
