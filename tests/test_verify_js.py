"""
Tests for verify_js.py: valid syntax, invalid syntax, code extraction,
retry loop success/exhausted, and tool-absent fallback (verified=False).
"""
import unittest.mock as mock
import pytest
from zolt.inference.verify_js import (
    verify_javascript_code,
    self_correcting_generate_js,
)
from zolt.inference.verify_base import extract_code_block


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


def test_verify_javascript_valid_syntax():
    code = (
        "function greet(name) {\n"
        "  return 'Hello, ' + name;\n"
        "}\n"
        "const result = greet('world');\n"
    )
    res = verify_javascript_code(code)
    assert res["valid"]


def test_verify_javascript_invalid_syntax():
    code = "function broken( { return [1, 2; }"
    res = verify_javascript_code(code)
    assert not res["valid"]
    assert res["error"] is not None


def test_verify_javascript_result_shape():
    """verify_javascript_code must always return valid, verified, heuristic, error, checker."""
    res = verify_javascript_code("const x = 1;")
    for key in ("valid", "verified", "heuristic", "error", "checker"):
        assert key in res, f"Missing key: {key}"


def test_verified_false_when_node_absent():
    """
    When node is absent, a heuristically-passing snippet must return
    verified=False and heuristic=True. verified=True is never emitted
    for a heuristic result.
    """
    valid_js = "const add = (a, b) => a + b;"
    with mock.patch("zolt.inference.verify_js.shutil.which", return_value=None):
        res = verify_javascript_code(valid_js)

    assert res["valid"]
    assert res["verified"] is False
    assert res["heuristic"] is True
    assert res["checker"] == "heuristic"


def test_invalid_syntax_detected_without_node():
    """Heuristic catches bracket errors even without node."""
    broken = "function f( { return 1;;"
    with mock.patch("zolt.inference.verify_js.shutil.which", return_value=None):
        res = verify_javascript_code(broken)
    assert not res["valid"]


def test_self_correcting_retry_js_success():
    broken = "<code>\nfunction calc(a) { return (a * 2;\n</code>"
    fixed = "<code>\nfunction calc(a) { return a * 2; }\n</code>"
    mock_gen = MockGenerator([broken, fixed])

    res = self_correcting_generate_js(mock_gen, "Write a double function", max_retries=2)
    assert res["attempts"] == 2
    assert "return a * 2;" in res["code"]


def test_self_correcting_retry_js_exhausted():
    broken = "<code>\nfunction f( { return;\n</code>"
    mock_gen = MockGenerator([broken, broken])

    res = self_correcting_generate_js(mock_gen, "Write function", max_retries=2)
    assert not res["verified"]
    assert res["attempts"] == 2
    assert res["error"] is not None


def test_extract_code_block_fence_js():
    text = "```js\nconst x = 1;\n```"
    assert extract_code_block(text) == "const x = 1;"
