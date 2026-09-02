"""
Tests for verify_python.py: valid syntax, invalid syntax, code extraction,
retry loop success/exhausted, and checker distinctions (ast vs mypy).
"""

import unittest.mock as mock

from zolt.inference.verify_base import extract_code_block
from zolt.inference.verify_python import (
    self_correcting_generate_python,
    verify_python_code,
)


class MockGenerator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.attempt = 0

    def format_agent_prompt(self, system_prompt="", user_prompt="", include_think_tag=True):
        return f"{system_prompt} | {user_prompt}"

    def generate_stream(self, prompt, **kwargs):
        resp = self.responses[min(self.attempt, len(self.responses) - 1)]
        self.attempt += 1
        yield resp


def test_verify_python_valid_syntax():
    code = "def add(a: int, b: int) -> int:\n    return a + b\n\nresult = add(1, 2)\n"
    res = verify_python_code(code)
    assert res["valid"]
    # Python is always available, so verified must always be True
    assert res["verified"] is True
    assert res["heuristic"] is False


def test_verify_python_invalid_syntax():
    code = "def broken(\n    return 1\n"
    res = verify_python_code(code)
    assert not res["valid"]
    assert res["error"] is not None
    # ast.parse ran (Python is always available), so verified is still True
    assert res["verified"] is True


def test_verify_python_result_shape():
    """verify_python_code must always return valid, verified, heuristic, error, checker."""
    res = verify_python_code("x = 1")
    for key in ("valid", "verified", "heuristic", "error", "checker"):
        assert key in res, f"Missing key: {key}"


def test_verify_python_checker_is_ast():
    """Without mypy present, checker should be 'ast'."""
    with mock.patch("zolt.inference.verify_python.shutil.which", return_value=None):
        res = verify_python_code("x = 1 + 2")
    assert res["checker"] == "ast"
    assert res["verified"] is True


def test_verify_python_mypy_failure_propagated():
    """When mypy reports errors, valid=False and checker='mypy'."""
    fake_mypy = "/usr/bin/mypy"

    def mock_which(name):
        return fake_mypy if name == "mypy" else None

    with (
        mock.patch("zolt.inference.verify_python.shutil.which", side_effect=mock_which),
        mock.patch("zolt.inference.verify_python.subprocess.run") as mock_run,
    ):
        mock_run.return_value = mock.Mock(
            returncode=1,
            stdout="error: Incompatible types\n",
            stderr="",
        )
        res = verify_python_code("x: int = 'not an int'")

    assert not res["valid"]
    assert res["checker"] == "mypy"
    assert res["verified"] is True  # mypy ran; verified=True because the tool ran
    assert "Incompatible" in res["error"]


def test_verify_python_empty_code():
    res = verify_python_code("")
    assert not res["valid"]
    assert res["checker"] == "none"


def test_self_correcting_retry_python_success():
    broken = "<code>\ndef calc(a\n    return a * 2\n</code>"
    fixed = "<code>\ndef calc(a):\n    return a * 2\n</code>"
    mock_gen = MockGenerator([broken, fixed])

    res = self_correcting_generate_python(mock_gen, "Write a double function", max_retries=2)
    assert res["verified"]
    assert res["attempts"] == 2
    assert "return a * 2" in res["code"]


def test_self_correcting_retry_python_exhausted():
    broken = "<code>\ndef f(\n    return\n</code>"
    mock_gen = MockGenerator([broken, broken])

    res = self_correcting_generate_python(mock_gen, "Write function", max_retries=2)
    assert not res["verified"]
    assert res["attempts"] == 2
    assert res["error"] is not None


def test_extract_code_block_python_fence():
    text = "```python\ndef f(): pass\n```"
    assert extract_code_block(text) == "def f(): pass"
