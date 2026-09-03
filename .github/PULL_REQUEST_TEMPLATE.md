## Summary

<!-- One-paragraph description of what this PR does and why. -->

## Type of Change

<!-- Mark the relevant options with an "x": -->

- [ ] `feat` — New feature (non-breaking)
- [ ] `fix` — Bug fix (non-breaking)
- [ ] `perf` — Performance improvement
- [ ] `refactor` — Code change with no behavior change
- [ ] `test` — Adding or fixing tests
- [ ] `docs` — Documentation only
- [ ] `ci` — CI/CD pipeline change
- [ ] `chore` — Dependency update, tooling
- [ ] `BREAKING CHANGE` — Existing behavior changes (requires major version bump)

## Related Issues

<!-- Link any related issues. Use "Closes #123" to auto-close on merge. -->

Closes #

## Changes Made

<!-- Bullet-point list of the concrete changes in this PR. -->

-
-

## How to Test Locally

```bash
# Reproduce the issue (before fix) or exercise the new feature:

# Verify the fix / feature:
```

## CI Checklist

- [ ] `make lint` passes (`ruff check` + `ruff format --check`)
- [ ] `make typecheck` passes (`mypy`)
- [ ] `make smoke` passes (architecture smoke test)
- [ ] `make test` passes (all 159+ tests green)
- [ ] New code is covered by tests
- [ ] Docstrings added/updated for all new public APIs
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No unintended files included (check `git diff --stat`)

## Screenshots / Logs (if applicable)

<!-- Add any relevant output, benchmark numbers, or screenshots. -->
