"""
Dataset with sequence packing and causal LM DataLoader.
"""
import os
import random
from pathlib import Path
from typing import List, Optional, Dict, Iterator, Union

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class PackedSequenceDataset(IterableDataset):
    """
    Sequence packing dataset for causal language modeling.
    Concatenates token streams into fixed-length windows without intra-sequence padding.
    """

    def __init__(
        self,
        token_files: List[str],
        max_seq_len: int = 4096,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.token_files = token_files
        self.max_seq_len = max_seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.shuffle = shuffle
        self.seed = seed

    def _load_tokens(self, path: str) -> List[int]:
        """Load int32 binary or JSONL token file into a token list."""
        if path.endswith(".bin"):
            import numpy as np
            return np.fromfile(path, dtype=np.int32).tolist()
        elif path.endswith(".json") or path.endswith(".jsonl"):
            import json
            tokens = []
            with open(path) as f:
                for line in f:
                    obj = json.loads(line)
                    tokens.extend([self.bos_id] + obj.get("input_ids", []) + [self.eos_id])
            return tokens
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        files = list(self.token_files)
        if self.shuffle:
            rng.shuffle(files)

        buffer: List[int] = []

        for path in files:
            try:
                tokens = self._load_tokens(path)
            except Exception as e:
                print(f"[z1-data] Error loading {path}: {e}")
                continue

            buffer.extend(tokens)

            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}

        # Final partial chunk with padding
        if len(buffer) > 1:
            chunk = buffer[:self.max_seq_len + 1]
            pad_len = (self.max_seq_len + 1) - len(chunk)
            chunk = chunk + [self.pad_id] * pad_len

            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            labels = torch.tensor(chunk[1:], dtype=torch.long)
            # Mask padding in labels
            labels[labels == self.pad_id] = -100

            yield {"input_ids": input_ids, "labels": labels}


def build_dataloader(
    token_files: List[str],
    max_seq_len: int = 4096,
    batch_size: int = 8,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 0,
    num_workers: int = 2,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Build DataLoader with sequence packing for causal LM training."""
    dataset = PackedSequenceDataset(
        token_files=token_files,
        max_seq_len=max_seq_len,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
