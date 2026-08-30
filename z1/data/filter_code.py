"""
Data filtering by language, permissive license, quality heuristics, and SHA256 deduplication.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Set, Generator


# Target programming languages for z1
TARGET_LANGUAGES = {
    "javascript",
    "typescript",
    "python",
    "vue",
    "tsx",
    "jsx",
    "css",
    "scss",
    "html",
    "markdown",
    "json",
    "yaml",
}

# Permissive license whitelist
PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0",
    "apache 2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "unlicense",
    "cc0-1.0",
    "0bsd",
    "wtfpl",
    "artistic-2.0",
    "eupl-1.2",
    "lgpl-2.1",
    "lgpl-3.0",
    "mpl-2.0",
}

# Quality heuristic thresholds
MIN_CONTENT_LEN = 100     # minimum characters
MAX_CONTENT_LEN = 100_000 # maximum characters
MIN_ALPHANUM_RATIO = 0.20  # minimum alphanumeric character ratio


def is_permissive_license(license_str: Optional[str]) -> bool:
    if not license_str:
        return False
    return license_str.lower().strip() in PERMISSIVE_LICENSES


def is_target_language(lang: Optional[str]) -> bool:
    if not lang:
        return False
    return lang.lower().strip() in TARGET_LANGUAGES


def passes_quality_heuristics(content: str) -> bool:
    if not content or not content.strip():
        return False
    if len(content) < MIN_CONTENT_LEN or len(content) > MAX_CONTENT_LEN:
        return False
    alphanum = sum(c.isalnum() for c in content)
    if alphanum / len(content) < MIN_ALPHANUM_RATIO:
        return False
    return True


def content_hash(content: str) -> str:
    """Compute SHA256 hash for exact deduplication."""
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def filter_jsonl_file(
    input_path: str,
    output_path: str,
    seen_hashes: Optional[Set[str]] = None,
    lang_field: str = "lang",
    license_field: str = "license",
    content_field: str = "content",
    max_records: Optional[int] = None,
) -> dict:
    """Filter code JSONL file by language, license, quality heuristics, and SHA256 hash."""
    if seen_hashes is None:
        seen_hashes = set()

    stats = {
        "total": 0,
        "lang_filtered": 0,
        "license_filtered": 0,
        "quality_filtered": 0,
        "dedup_filtered": 0,
        "accepted": 0,
    }

    with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            stats["total"] += 1
            if max_records and stats["total"] > max_records:
                break

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            lang = obj.get(lang_field, "")
            if not is_target_language(lang):
                stats["lang_filtered"] += 1
                continue

            license_str = obj.get(license_field, "")
            if not is_permissive_license(license_str):
                stats["license_filtered"] += 1
                continue

            content = obj.get(content_field, "")
            if not passes_quality_heuristics(content):
                stats["quality_filtered"] += 1
                continue

            h = content_hash(content)
            if h in seen_hashes:
                stats["dedup_filtered"] += 1
                continue

            seen_hashes.add(h)

            out_record = {
                "content": content,
                "lang": lang,
                "license": license_str,
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            stats["accepted"] += 1

    return stats


def tokenize_and_save(
    filtered_jsonl: str,
    tokenizer_path: str,
    output_bin: str,
    content_field: str = "content",
    bos_id: int = 1,
    eos_id: int = 2,
):
    """Tokenize filtered JSONL file and save as an int32 binary token array."""
    import numpy as np
    from tokenizers import Tokenizer as HFTokenizer

    tok = HFTokenizer.from_file(tokenizer_path)
    all_tokens = []

    with open(filtered_jsonl, "r") as f:
        for line in f:
            obj = json.loads(line)
            content = obj.get(content_field, "")
            enc = tok.encode(content, add_special_tokens=False)
            all_tokens.append(bos_id)
            all_tokens.extend(enc.ids)
            all_tokens.append(eos_id)

    arr = np.array(all_tokens, dtype=np.int32)
    arr.tofile(output_bin)
    print(f"[z1-data] Tokenization complete -> {len(arr):,} tokens -> {output_bin}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="z1 Data Filter and Tokenizer")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output filtered JSONL file")
    parser.add_argument("--tokenizer", default=None, help="Path to z1 tokenizer JSON")
    parser.add_argument("--output_bin", default=None, help="Output int32 .bin token file path")
    parser.add_argument("--max_records", type=int, default=None)
    args = parser.parse_args()

    print(f"[z1-data] Filtering {args.input} -> {args.output}")
    stats = filter_jsonl_file(args.input, args.output, max_records=args.max_records)
    print(f"[z1-data] Stats: {json.dumps(stats, indent=2)}")

    if args.tokenizer and args.output_bin:
        tokenize_and_save(args.output, args.tokenizer, args.output_bin)
