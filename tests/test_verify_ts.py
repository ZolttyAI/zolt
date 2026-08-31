"""
Tests for verify_ts.py after refactor onto verify_base.
All existing external behaviour must be unchanged.
"""
import pytest
from z1.inference.verify_ts import (
    verify_typescript_code,
    extract_code_block,
    self_correcting_generate_ts,
    run_tsc_check,
)


class MockGenerator:
    """Mock generator cycling through a fixed list of responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.attempt = 0

    def format_agent_prompt(self, system_prompt="", user_prompt="", include_think_tag=True):
        return f"{system_prompt} | {user_prompt}"

    def generate_stream(self, prompt, **kwargs):
        resp = self.responses[min(self.attempt, len(self.responses) - 1)]
        self.attempt += 1
        yield resp


def test_verify_typescript_valid_syntax():
    code = (
        "interface User {\n"
        "  id: number;\n"
        "  name: string;\n"
        "}\n"
        "const getUser = (id: number): User => ({ id, name: 'Alice' });\n"
    )
    res = verify_typescript_code(code)
    assert res["valid"]


def test_verify_typescript_invalid_syntax():
    code = "const f = (x: number) => { return [1, 2; };"
    res = verify_typescript_code(code)
    assert not res["valid"]
    assert res["error"] is not None


def test_extract_code_block_tags():
    text = "Solution:\n<code>\nconst x: number = 42;\n</code>\nDone."
    assert extract_code_block(text) == "const x: number = 42;"


def test_verify_typescript_result_shape():
    """verify_typescript_code must always return valid, verified, heuristic, error, checker."""
    res = verify_typescript_code("const x = 1;")
    for key in ("valid", "verified", "heuristic", "error", "checker"):
        assert key in res, f"Missing key: {key}"


def test_verified_true_only_from_tool_or_heuristic_passes():
    """
    When tsc is absent, heuristic-passing code yields verified=False, heuristic=True.
    verified=True is reserved for actual tsc runs.
    This test patches shutil.which to simulate absent tsc.
    """
    import shutil
    import unittest.mock as mock

    valid_ts = "const x: number = 1;"
    with mock.patch("z1.inference.verify_ts.shutil.which", return_value=None):
        res = verify_typescript_code(valid_ts)

    # Heuristic passes but verified must be False when tsc is absent
    assert res["valid"]
    assert res["verified"] is False
    assert res["heuristic"] is True


def test_self_correcting_retry_loop_success():
    broken = "<code>\nfunction calc(a: number) { return (a * 2;\n</code>"
    fixed = "<code>\nfunction calc(a: number): number { return a * 2; }\n</code>"
    mock_gen = MockGenerator([broken, fixed])

    res = self_correcting_generate_ts(mock_gen, "Write a double function", max_retries=2)
    assert res["attempts"] == 2
    assert "return a * 2;" in res["code"]


def test_self_correcting_retry_loop_exhausted():
    broken = "<code>\nfunction calc( { return;\n</code>"
    mock_gen = MockGenerator([broken, broken])

    res = self_correcting_generate_ts(mock_gen, "Write function", max_retries=2)
    assert not res["verified"]
    assert res["attempts"] == 2
    assert res["error"] is not None


def test_run_tsc_check_backward_compat():
    """run_tsc_check must return plain dict with valid and error keys."""
    result = run_tsc_check("const x = 1;")
    assert "valid" in result
    assert "error" in result
