"""Unit tests for native diff format parser and applicator."""
import pytest
from z1.inference.diff_format import (
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


def test_apply_diff_edit_no_match_raises_error():
    source = "def foo(): pass"
    search = "def bar(): pass"
    replace = "def baz(): pass"
    with pytest.raises(ValueError, match="Search block not found"):
        apply_diff_edit(source, search, replace)


def test_apply_diff_edit_multiple_matches_raises_error():
    source = (
        "x = 10\n"
        "x = 10\n"
        "y = 20\n"
    )
    search = "x = 10"
    replace = "x = 100"
    with pytest.raises(ValueError, match="matched 2 locations"):
        apply_diff_edit(source, search, replace)


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
