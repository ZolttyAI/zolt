"""Unit tests for zolt tokenizer."""

import os
import tempfile

from zolt.tokenizer.train_tokenizer import ZOLT_SPECIAL_TOKENS, get_code_iterator


def test_special_tokens_count():
    """Verify that special tokens are uniquely defined."""
    assert len(ZOLT_SPECIAL_TOKENS) == len(set(ZOLT_SPECIAL_TOKENS)), (
        "Duplicate special tokens detected"
    )
    assert "<think>" in ZOLT_SPECIAL_TOKENS
    assert "</think>" in ZOLT_SPECIAL_TOKENS
    assert "<tool_call>" in ZOLT_SPECIAL_TOKENS
    assert "</tool_call>" in ZOLT_SPECIAL_TOKENS
    assert "<search>" in ZOLT_SPECIAL_TOKENS
    assert "<replace>" in ZOLT_SPECIAL_TOKENS
    assert "<diff_end>" in ZOLT_SPECIAL_TOKENS
    assert "<uncertain>" in ZOLT_SPECIAL_TOKENS
    assert "<db_call>" in ZOLT_SPECIAL_TOKENS
    assert "</db_call>" in ZOLT_SPECIAL_TOKENS
    assert "<bos>" in ZOLT_SPECIAL_TOKENS
    assert "<eos>" in ZOLT_SPECIAL_TOKENS
    assert "<pad>" in ZOLT_SPECIAL_TOKENS
    assert len(ZOLT_SPECIAL_TOKENS) == 23


def test_special_token_ordering():
    """Verify that pad=0, bos=1, eos=2, unk=3 token ordering matches configuration."""
    assert ZOLT_SPECIAL_TOKENS[0] == "<pad>"
    assert ZOLT_SPECIAL_TOKENS[1] == "<bos>"
    assert ZOLT_SPECIAL_TOKENS[2] == "<eos>"
    assert ZOLT_SPECIAL_TOKENS[3] == "<unk>"


def test_code_iterator_finds_files():
    """Verify that code iterator discovers target file extensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic files for testing
        for name, content in [
            ("main.py", "def hello():\n    print('hello') * 5\n" * 20),
            ("app.js", "const x = () => console.log('zolt') * 5\n" * 20),
            ("README.txt", "this file should be ignored"),
        ]:
            with open(os.path.join(tmpdir, name), "w") as f:
                f.write(content)

        found = list(get_code_iterator([tmpdir]))

    # Should match .py and .js but ignore .txt
    assert len(found) == 2


def test_code_iterator_max_files():
    """Verify max_files limit in code iterator."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(10):
            with open(os.path.join(tmpdir, f"file{i}.py"), "w") as f:
                f.write("x = 1\n" * 50)

        found = list(get_code_iterator([tmpdir], max_files=3))

    assert len(found) == 3
