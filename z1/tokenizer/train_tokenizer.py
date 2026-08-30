"""
Custom Byte-Level BPE tokenizer training for code and reasoning.
"""
import os
import json
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


# Special tokens for z1 coding agent and reasoning
Z1_SPECIAL_TOKENS = [
    "<pad>",    # 0
    "<bos>",    # 1
    "<eos>",    # 2
    "<unk>",    # 3
    "<think>",  # 4 - internal reasoning block start
    "</think>", # 5 - internal reasoning block end
    "<tool_call>",     # 6 - tool call block start
    "</tool_call>",    # 7 - tool call block end
    "<tool_response>", # 8 - tool response block start
    "</tool_response>",# 9 - tool response block end
    "<code>",   # 10 - code block start
    "</code>",  # 11 - code block end
    "<|im_start|>",    # 12 - ChatML start
    "<|im_end|>",      # 13 - ChatML end
    "<FILL>",   # 14 - FIM middle
    "<PREFIX>", # 15 - FIM prefix
    "<SUFFIX>", # 16 - FIM suffix
]

CODE_EXTENSIONS = [
    ".js", ".jsx", ".ts", ".tsx",
    ".py", ".vue", ".svelte",
    ".json", ".yaml", ".yml",
    ".md", ".mdx",
    ".html", ".css", ".scss",
]


def get_code_iterator(data_dirs: List[str], max_files: Optional[int] = None):
    """Iterate over code files in target directories."""
    count = 0
    for data_dir in data_dirs:
        for root, _, files in os.walk(data_dir):
            for fname in files:
                if any(fname.endswith(ext) for ext in CODE_EXTENSIONS):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        if content.strip():
                            yield content
                            count += 1
                            if max_files and count >= max_files:
                                return
                    except Exception:
                        continue


def train_z1_tokenizer(
    data_dirs: List[str],
    output_path: str = "z1_tokenizer.json",
    vocab_size: int = 32000,
    max_files: Optional[int] = None,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a Byte-Level BPE tokenizer on code corpora."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Byte-level pre-tokenization for code robustness
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=Z1_SPECIAL_TOKENS,
        show_progress=True,
    )

    print(f"[z1-tokenizer] Training BPE tokenizer | vocab_size={vocab_size}")
    iterator = list(get_code_iterator(data_dirs, max_files=max_files))
    print(f"[z1-tokenizer] Found {len(iterator)} files")

    tokenizer.train_from_iterator(iterator, trainer=trainer)

    # Automatically attach BOS/EOS tokens to single sequences
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    tokenizer.save(output_path)
    print(f"[z1-tokenizer] Tokenizer saved to: {output_path}")
    print(f"[z1-tokenizer] Effective vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train z1 BPE tokenizer")
    parser.add_argument("--data_dirs", nargs="+", required=True, help="Input directories containing source code")
    parser.add_argument("--output", default="z1_tokenizer.json", help="Output tokenizer JSON path")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Target vocabulary size (32000 or 50000)")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process")
    args = parser.parse_args()

    train_z1_tokenizer(
        data_dirs=args.data_dirs,
        output_path=args.output,
        vocab_size=args.vocab_size,
        max_files=args.max_files,
    )
