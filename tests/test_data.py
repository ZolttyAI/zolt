"""Unit tests for z1 data pipeline."""
import json
import os
import tempfile

import pytest

from z1.data.filter_code import (
    is_permissive_license,
    is_target_language,
    passes_quality_heuristics,
    content_hash,
    filter_jsonl_file,
)
from z1.data.dataset import PackedSequenceDataset


# --- filter_code Tests ---

def test_permissive_license_accept():
    assert is_permissive_license("MIT")
    assert is_permissive_license("apache-2.0")
    assert is_permissive_license("BSD-3-Clause")


def test_permissive_license_reject():
    assert not is_permissive_license("GPL-3.0")
    assert not is_permissive_license("AGPL-3.0")
    assert not is_permissive_license(None)
    assert not is_permissive_license("")


def test_target_language_accept():
    assert is_target_language("javascript")
    assert is_target_language("TypeScript")
    assert is_target_language("python")
    assert is_target_language("vue")


def test_target_language_reject():
    assert not is_target_language("java")
    assert not is_target_language("ruby")
    assert not is_target_language(None)


def test_quality_heuristics_pass():
    content = "def hello_world():\n    print('Hello, World!')\n" * 10
    assert passes_quality_heuristics(content)


def test_quality_heuristics_fail_short():
    assert not passes_quality_heuristics("x = 1")


def test_quality_heuristics_fail_empty():
    assert not passes_quality_heuristics("")
    assert not passes_quality_heuristics("   \n\n")


def test_content_hash_deterministic():
    content = "const x = 1;"
    assert content_hash(content) == content_hash(content)


def test_content_hash_different():
    assert content_hash("const x = 1;") != content_hash("const x = 2;")


def test_filter_jsonl_file():
    records = [
        {"content": "def foo():\n    return 42\n" * 10, "lang": "python", "license": "mit"},
        {"content": "import java.util.List;", "lang": "java", "license": "mit"},  # reject: wrong lang
        {"content": "const x = 1;", "lang": "javascript", "license": "GPL-3.0"},  # reject: non-permissive license
        {"content": "   ", "lang": "python", "license": "mit"},  # reject: quality heuristic
        {"content": "def foo():\n    return 42\n" * 10, "lang": "python", "license": "mit"},  # reject: duplicate content
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fin:
        for r in records:
            fin.write(json.dumps(r) + "\n")
        input_path = fin.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fout:
        output_path = fout.name

    stats = filter_jsonl_file(input_path, output_path)

    assert stats["accepted"] == 1
    assert stats["lang_filtered"] == 1
    assert stats["license_filtered"] == 1
    assert stats["dedup_filtered"] == 1

    os.unlink(input_path)
    os.unlink(output_path)


# --- PackedSequenceDataset Tests ---

def test_packed_dataset_basic():
    """Verify that PackedSequenceDataset produces properly shaped batches."""
    import tempfile
    import numpy as np

    tokens = list(range(1, 5001))  # 5000 sequential tokens
    arr = np.array(tokens, dtype=np.int32)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        arr.tofile(f)
        bin_path = f.name

    dataset = PackedSequenceDataset(
        token_files=[bin_path],
        max_seq_len=128,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        shuffle=False,
    )

    batches = list(dataset)
    assert len(batches) > 0

    for batch in batches:
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 128
        assert batch["labels"].shape[0] == 128

    os.unlink(bin_path)
