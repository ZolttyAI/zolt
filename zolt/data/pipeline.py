#!/usr/bin/env python3
"""
End-to-end data processing pipeline orchestrator.
Stages: filter -> train tokenizer -> tokenize -> validate.
"""

import argparse
from pathlib import Path


def run_step(label: str, fn, *args, **kwargs):
    print(f"\n{'─' * 60}")
    print(f"[zolt-pipeline] {label}")
    print(f"{'─' * 60}")
    result = fn(*args, **kwargs)
    print(f"[zolt-pipeline] ✓ {label} complete")
    return result


def step_filter_all(
    raw_dir: str, filtered_dir: str, max_records: int | None = None, min_quality_score: float = 0.0
):
    """Filter all raw JSONL files into filtered JSONL files."""
    from zolt.data.filter_code import filter_jsonl_file

    raw_path = Path(raw_dir)
    filtered_path = Path(filtered_dir)
    filtered_path.mkdir(parents=True, exist_ok=True)

    jsonl_files = list(raw_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"[zolt-pipeline] No JSONL files found in {raw_dir}")
        return {}

    total_stats = {
        "total": 0,
        "accepted": 0,
        "lang_filtered": 0,
        "license_filtered": 0,
        "quality_filtered": 0,
        "quality_score_filtered": 0,
        "dedup_filtered": 0,
    }
    seen_hashes: set[str] = set()  # Global deduplication across all files

    for jf in sorted(jsonl_files):
        out_path = filtered_path / jf.name
        print(f"[zolt-pipeline] Filtering: {jf.name}")
        stats = filter_jsonl_file(
            str(jf),
            str(out_path),
            seen_hashes=seen_hashes,
            max_records=max_records,
            min_quality_score=min_quality_score,
        )
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)
        print(
            f"  total={stats['total']:,} | accepted={stats['accepted']:,} | "
            f"quality_score_filtered={stats.get('quality_score_filtered', 0):,} | "
            f"dedup={stats['dedup_filtered']:,} | lang={stats['lang_filtered']:,} | "
            f"license={stats['license_filtered']:,}"
        )

    print(
        f"\n[zolt-pipeline] TOTAL: {total_stats['accepted']:,} / {total_stats['total']:,} documents accepted"
    )
    return total_stats


def step_train_tokenizer(filtered_dir: str, tokenizer_out: str, vocab_size: int = 32000):
    """Train BPE tokenizer on filtered dataset."""
    from zolt.tokenizer.train_tokenizer import train_zolt_tokenizer

    data_dirs = [filtered_dir]
    train_zolt_tokenizer(
        data_dirs=data_dirs,
        output_path=tokenizer_out,
        vocab_size=vocab_size,
    )


def step_tokenize_all(filtered_dir: str, tokenizer_path: str, tokens_dir: str):
    """Tokenize all filtered JSONL files into .bin arrays."""
    from zolt.data.filter_code import tokenize_and_save

    filtered_path = Path(filtered_dir)
    tokens_path = Path(tokens_dir)
    tokens_path.mkdir(parents=True, exist_ok=True)

    for jf in sorted(filtered_path.glob("*.jsonl")):
        out_bin = tokens_path / (jf.stem + ".bin")
        if out_bin.exists():
            print(f"[zolt-pipeline] Already exists: {out_bin} - skipping")
            continue
        print(f"[zolt-pipeline] Tokenize: {jf.name} -> {out_bin.name}")
        tokenize_and_save(str(jf), tokenizer_path, str(out_bin))


def step_validate_tokens(tokens_dir: str, max_seq_len: int = 4096):
    """Validate token counts across processed .bin files."""
    import numpy as np

    tokens_path = Path(tokens_dir)
    bin_files = list(tokens_path.glob("*.bin"))

    if not bin_files:
        print(f"[zolt-pipeline] No .bin files found in {tokens_dir}")
        return

    total_tokens = 0
    for bf in sorted(bin_files):
        arr = np.fromfile(str(bf), dtype=np.int32)
        total_tokens += len(arr)
        print(f"  {bf.name}: {len(arr):,} tokens ({len(arr) / 1e6:.1f}M)")

    print(f"\n[zolt-pipeline] Total tokens: {total_tokens:,} ({total_tokens / 1e9:.2f}B)")

    target = 3_000_000_000
    pct = total_tokens / target * 100
    status = "SUFFICIENT" if total_tokens >= target else f"INSUFFICIENT ({pct:.0f}% of 3B target)"
    print(f"[zolt-pipeline] Minimum target: 3B tokens | Status: {status}")

    return total_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="zolt Data Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # filter
    p_filter = subparsers.add_parser("filter", help="Filter raw JSONL -> filtered JSONL")
    p_filter.add_argument("--raw_dir", default="data/raw")
    p_filter.add_argument("--filtered_dir", default="data/filtered")
    p_filter.add_argument("--min_quality_score", type=float, default=0.0)
    p_filter.add_argument("--max_records", type=int, default=None)

    # tokenizer
    p_tok = subparsers.add_parser("train-tokenizer", help="Train BPE tokenizer")
    p_tok.add_argument("--filtered_dir", default="data/filtered")
    p_tok.add_argument("--output", default="zolt_tokenizer.json")
    p_tok.add_argument("--vocab_size", type=int, default=32000)

    # tokenize
    p_tokenize = subparsers.add_parser("tokenize", help="Tokenize JSONL -> .bin")
    p_tokenize.add_argument("--filtered_dir", default="data/filtered")
    p_tokenize.add_argument("--tokenizer", default="zolt_tokenizer.json")
    p_tokenize.add_argument("--tokens_dir", default="data/tokens")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate .bin files")
    p_validate.add_argument("--tokens_dir", default="data/tokens")

    # all
    p_all = subparsers.add_parser("all", help="Run full pipeline")
    p_all.add_argument("--raw_dir", default="data/raw")
    p_all.add_argument("--filtered_dir", default="data/filtered")
    p_all.add_argument("--tokenizer", default="zolt_tokenizer.json")
    p_all.add_argument("--tokens_dir", default="data/tokens")
    p_all.add_argument("--vocab_size", type=int, default=32000)
    p_all.add_argument("--min_quality_score", type=float, default=0.0)
    p_all.add_argument("--max_records", type=int, default=None)

    args = parser.parse_args()

    if args.command == "filter":
        step_filter_all(args.raw_dir, args.filtered_dir, args.max_records, args.min_quality_score)
    elif args.command == "train-tokenizer":
        step_train_tokenizer(args.filtered_dir, args.output, args.vocab_size)
    elif args.command == "tokenize":
        step_tokenize_all(args.filtered_dir, args.tokenizer, args.tokens_dir)
    elif args.command == "validate":
        step_validate_tokens(args.tokens_dir)
    elif args.command == "all":
        run_step(
            "1. Filter data",
            step_filter_all,
            args.raw_dir,
            args.filtered_dir,
            args.max_records,
            args.min_quality_score,
        )
        run_step(
            "2. Train tokenizer",
            step_train_tokenizer,
            args.filtered_dir,
            args.tokenizer,
            args.vocab_size,
        )
        run_step(
            "3. Tokenize data",
            step_tokenize_all,
            args.filtered_dir,
            args.tokenizer,
            args.tokens_dir,
        )
        run_step("4. Validate tokens", step_validate_tokens, args.tokens_dir)
    else:
        parser.print_help()
