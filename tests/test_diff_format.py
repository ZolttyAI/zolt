"""Unit tests for native diff format parser and applicator."""
import pytest
from zolt.inference.diff_format import (
    parse_diff_blocks,
    apply_diff_edit,
    apply_diff_block,
    format_diff_block,
    DiffEdit,
)


def test_parse_diff_block_with_path():
    text = (
        "[src/utils.py]\n"
        "<search>\n"
        "def add(a, b):\n"
        "    return a - b\n"
        "<replace>\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "<diff_end>"
    )
    edits = parse_diff_blocks(text)
    assert len(edits) == 1
    assert edits[0].path == "src/utils.py"
    assert "return a - b" in edits[0].search
    assert "return a + b" in edits[0].replace


def test_parse_diff_block_without_path():
    text = (
        "<search>\n"
        "const x = 1;\n"
        "<replace>\n"
        "const x = 2;\n"
        "<diff_end>"
    )
    edits = parse_diff_blocks(text)
    assert len(edits) == 1
    assert edits[0].path is None
    assert "const x = 1;" in edits[0].search
    assert "const x = 2;" in edits[0].replace


def test_apply_diff_edit_exact_match():
    source = (
        "def calculate():\n"
        "    a = 10\n"
        "    b = 20\n"
        "    return a - b\n"
    )
    search = "return a - b"
    replace = "return a + b"
    result = apply_diff_edit(source, search, replace)
    assert "return a + b" in result
    assert "return a - b" not in result


# ── Fail-closed regression: Decision (a) ──────────────────────────────────────

def test_apply_diff_edit_no_match_raises_error():
    """0-match must raise ValueError, never return a best-effort result."""
    source = "def foo(): pass"
    search = "def bar(): pass"
    replace = "def baz(): pass"
    with pytest.raises(ValueError, match="Search block not found"):
        apply_diff_edit(source, search, replace)


def test_apply_diff_edit_multiple_matches_raises_error():
    """2+ matches must raise ValueError, never silently pick the first occurrence."""
    source = (
        "x = 10\n"
        "x = 10\n"
        "y = 20\n"
    )
    search = "x = 10"
    replace = "x = 100"
    with pytest.raises(ValueError, match="matched 2 locations"):
        apply_diff_edit(source, search, replace)


def test_apply_diff_edit_zero_match_does_not_modify_source():
    """Confirm no mutation occurs when the search block is absent."""
    source = "a = 1\nb = 2\n"
    try:
        apply_diff_edit(source, "c = 3", "c = 99")
    except ValueError:
        pass  # expected
    # Source must be untouched (no in-place mutation)
    assert source == "a = 1\nb = 2\n"


def test_apply_diff_block_full():
    source = (
        "function greet(name) {\n"
        "  console.log('Hi ' + name);\n"
        "}\n"
    )
    diff = (
        "[greet.js]\n"
        "<search>\n"
        "  console.log('Hi ' + name);\n"
        "<replace>\n"
        "  console.log(`Hello, ${name}!`);\n"
        "<diff_end>"
    )
    result = apply_diff_block(source, diff)
    assert "console.log(`Hello, ${name}!`);" in result


# ── Cross-language verified guarantee: Decision (b) ───────────────────────────

def test_verified_true_not_emitted_from_heuristic_typescript():
    """TS: verified=True only from tsc, never from heuristic fallback."""
    from zolt.inference.verify_ts import verify_typescript_code
    import unittest.mock as mock

    valid_ts = "const x: string = 'hello';"
    with mock.patch("zolt.inference.verify_ts.shutil.which", return_value=None):
        res = verify_typescript_code(valid_ts)

    if res["valid"]:
        assert res["verified"] is False, "verified=True must not appear on heuristic path (TS)"
        assert res["heuristic"] is True


def test_verified_true_not_emitted_from_heuristic_javascript():
    """JS: verified=True only from node/eslint, never from heuristic fallback."""
    from zolt.inference.verify_js import verify_javascript_code
    import unittest.mock as mock

    valid_js = "const add = (a, b) => a + b;"
    with mock.patch("zolt.inference.verify_js.shutil.which", return_value=None):
        res = verify_javascript_code(valid_js)

    if res["valid"]:
        assert res["verified"] is False, "verified=True must not appear on heuristic path (JS)"
        assert res["heuristic"] is True


def test_python_verified_always_true():
    """Python: verified=True always, because ast.parse always runs."""
    from zolt.inference.verify_python import verify_python_code
    res = verify_python_code("x = 1 + 2")
    assert res["verified"] is True
    assert res["heuristic"] is False
