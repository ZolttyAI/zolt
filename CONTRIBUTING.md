# Contributing

All changes should go through a pull request into `main`. Direct pushes are disabled by branch protection on the default branch.

## CI requirements

Before a pull request can be merged, the required CI checks must pass. The project uses the smoke test and the pytest suite from the repository's Development Quickstart.

## Licensing

By submitting a contribution, you agree that your work is licensed under the Apache License 2.0, as described in the project's `LICENSE` file and in Section 5 of the license itself. No separate contributor license agreement or signature is required.

## Run checks locally before opening a PR

```bash
# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu --python .venv/bin/python
uv pip install tokenizers datasets pytest einops tqdm --python .venv/bin/python
uv pip install -e . --python .venv/bin/python

# Run architecture smoke test
python smoke_test.py

# Run unit test suite
pytest tests/ -v
```
