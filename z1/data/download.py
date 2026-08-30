#!/usr/bin/env python3
"""
Download and prepare training datasets (The Stack v2, StarCoderData).
"""
import os
import json
import argparse
from pathlib import Path


# --- Source configuration ---

# Target languages and their respective dataset directory names in The Stack v2
STACK_V2_LANGS = {
    "javascript":  "JavaScript",
    "typescript":  "TypeScript",
    "python":      "Python",
    "vue":         "Vue",
    "css":         "CSS",
    "html":        "HTML",
    "markdown":    "Markdown",
    "json":        "JSON",
    "yaml":        "YAML",
}

STARCODER_LANGS = [
    "javascript", "typescript", "python",
    "vue", "css", "html",
]


def download_stack_v2(
    output_dir: str,
    langs: list = None,
    max_samples_per_lang: int = 500_000,
    hf_token: str = None,
):
    """
    Download a filtered subset of The Stack v2 for target languages.
    Requires Hugging Face Hub authentication.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[z1-data] Missing dependency: install datasets via pip")
        return

    if langs is None:
        langs = list(STACK_V2_LANGS.keys())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        lang_name = STACK_V2_LANGS.get(lang, lang)
        out_path = output_dir / f"stack_v2_{lang}.jsonl"

        if out_path.exists():
            print(f"[z1-data] Already exists: {out_path} - skipping")
            continue

        print(f"[z1-data] Downloading The Stack v2 - {lang_name}...")
        try:
            ds = load_dataset(
                "bigcode/the-stack-v2",
                data_dir=f"data/{lang_name}",
                split="train",
                streaming=True,
                token=hf_token,
            )

            count = 0
            with open(out_path, "w", encoding="utf-8") as f:
                for sample in ds:
                    record = {
                        "content":  sample.get("content", ""),
                        "lang":     lang,
                        "license":  sample.get("license", ""),
                        "repo":     sample.get("repo_name", ""),
                        "path":     sample.get("path", ""),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    if count >= max_samples_per_lang:
                        break

            print(f"[z1-data] {lang}: {count:,} samples -> {out_path}")
        except Exception as e:
            print(f"[z1-data] Error downloading {lang}: {e}")


def download_starcoder(
    output_dir: str,
    langs: list = None,
    max_samples_per_lang: int = 300_000,
    hf_token: str = None,
):
    """
    Download a subset of StarCoderData for target languages.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[z1-data] Missing dependency: install datasets via pip")
        return

    if langs is None:
        langs = STARCODER_LANGS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        out_path = output_dir / f"starcoder_{lang}.jsonl"

        if out_path.exists():
            print(f"[z1-data] Already exists: {out_path} - skipping")
            continue

        print(f"[z1-data] Downloading StarCoderData - {lang}...")
        try:
            ds = load_dataset(
                "bigcode/starcoderdata",
                data_dir=lang,
                split="train",
                streaming=True,
                token=hf_token,
            )

            count = 0
            with open(out_path, "w", encoding="utf-8") as f:
                for sample in ds:
                    record = {
                        "content":  sample.get("content", ""),
                        "lang":     lang,
                        "license":  sample.get("license", "mit"),
                        "repo":     sample.get("repo_name", ""),
                        "path":     sample.get("path", ""),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    if count >= max_samples_per_lang:
                        break

            print(f"[z1-data] {lang}: {count:,} samples -> {out_path}")
        except Exception as e:
            print(f"[z1-data] Error downloading {lang}: {e}")


def estimate_tokens(jsonl_path: str, sample_size: int = 1000) -> dict:
    """Estimate total token count for a JSONL dataset using sampling."""
    total_chars = 0
    total_lines = 0

    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            obj = json.loads(line)
            total_chars += len(obj.get("content", ""))
            total_lines += 1

    if total_lines == 0:
        return {}

    avg_chars = total_chars / total_lines
    # Rough heuristic: ~4 characters per BPE token for source code
    estimated_tokens_per_doc = avg_chars / 4

    # Count total lines in the file
    with open(jsonl_path, "r") as f:
        total_docs = sum(1 for _ in f)

    total_tokens_est = total_docs * estimated_tokens_per_doc

    return {
        "total_docs": total_docs,
        "avg_chars_per_doc": round(avg_chars, 1),
        "estimated_tokens": round(total_tokens_est / 1e9, 2),
        "estimated_tokens_unit": "B",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="z1 Data Downloader")
    parser.add_argument("--source", choices=["starcoder", "stack_v2", "both"], default="starcoder",
                        help="Data source to download")
    parser.add_argument("--output_dir", default="data/raw", help="Output directory")
    parser.add_argument("--langs", nargs="+", default=None, help="Target languages")
    parser.add_argument("--max_samples", type=int, default=300_000, help="Maximum samples per language")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"), help="Hugging Face authentication token")
    parser.add_argument("--estimate", default=None, help="Estimate token count of an existing JSONL file")
    args = parser.parse_args()

    if args.estimate:
        stats = estimate_tokens(args.estimate)
        print(json.dumps(stats, indent=2))
    else:
        if args.source in ("starcoder", "both"):
            download_starcoder(args.output_dir, args.langs, args.max_samples, args.hf_token)
        if args.source in ("stack_v2", "both"):
            download_stack_v2(args.output_dir, args.langs, args.max_samples, args.hf_token)
