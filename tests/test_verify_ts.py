"""Unit tests for TypeScript self-verification and correction loop."""
import pytest
from z1.inference.verify_ts import (
    verify_typescript_code,
    extract_code_block,
    self_correcting_generate_ts,
)


class MockGenerator:
    """Mock generator that returns broken code on attempt 1, and valid code on attempt 2."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.attempt = 0

    def format_agent_prompt(self, system_prompt="", user_prompt="", include_think_tag=True):
        return f"{system_prompt} | {user_prompt}"

    def generate_stream(self, prompt, **kwargs):
        if self.attempt < len(self.responses):
            resp = self.responses[self.attempt]
            self.attempt += 1
        else:
            resp = self.responses[-1]
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
    code = "const f = (x: number) => { return [1, 2; };"  # unclosed bracket
    res = verify_typescript_code(code)
    assert not res["valid"]
    assert "error" in res


def test_extract_code_block_tags():
    text = "Here is the solution:\n<code>\nconst x: number = 42;\n</code>\nDone."
    assert extract_code_block(text) == "const x: number = 42;"


def test_self_correcting_retry_loop_success():
    # Attempt 1: broken unclosed bracket
    # Attempt 2: valid code
    broken = "<code>\nfunction calc(a: number) { return (a * 2;\n</code>"
    fixed = "<code>\nfunction calc(a: number): number { return a * 2; }\n</code>"
    mock_gen = MockGenerator([broken, fixed])

    res = self_correcting_generate_ts(mock_gen, "Write a double function", max_retries=2)
    assert res["verified"]
    assert res["attempts"] == 2
    assert "return a * 2;" in res["code"]


def test_self_correcting_retry_loop_exhausted():
    # Both attempts broken
    broken = "<code>\nfunction calc( { return;\n</code>"
    mock_gen = MockGenerator([broken, broken])

    res = self_correcting_generate_ts(mock_gen, "Write function", max_retries=2)
    assert not res["verified"]
    assert res["attempts"] == 2
    assert res["error"] is not None
