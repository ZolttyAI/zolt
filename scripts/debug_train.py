#!/usr/bin/env python3
"""
Debug training run with synthetic tokens on CPU.
Validates training loop, learning rate schedule, and checkpoint persistence.
"""
import os
import sys
import numpy as np
import tempfile

def create_debug_tokens(output_path: str, n_tokens: int = 50_000, vocab_size: int = 32000):
    """Generate random token array for smoke-testing the training loop."""
    rng = np.random.default_rng(42)
    tokens = rng.integers(4, vocab_size, size=n_tokens, dtype=np.int32)
    tokens.tofile(output_path)
    print(f"[debug] Generated {n_tokens:,} tokens -> {output_path}")
    return output_path


def run_debug_train():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic token file
        token_file = os.path.join(tmpdir, "debug.bin")
        create_debug_tokens(token_file, n_tokens=20_000)

        # Execute minimal training run
        sys.argv = [
            "train.py",
            "--token_files", token_file,
            "--output_dir", os.path.join(tmpdir, "ckpts"),
            "--max_seq_len", "128",
            "--batch_size", "2",
            "--grad_accum", "1",
            "--lr", "3e-4",
            "--warmup_steps", "5",
            "--total_steps", "20",
            "--save_every", "10",
            "--log_every", "5",
            "--dtype", "fp32",
        ]

        from z1.train import train
        train(
            token_files=[token_file],
            output_dir=os.path.join(tmpdir, "ckpts"),
            max_seq_len=128,
            batch_size=2,
            grad_accum_steps=1,
            lr=3e-4,
            warmup_steps=5,
            total_steps=20,
            save_every=10,
            log_every=5,
            dtype="fp32",
        )

        # Verify generated checkpoints
        import glob
        ckpts = glob.glob(os.path.join(tmpdir, "ckpts", "ckpt-step*"))
        assert len(ckpts) > 0, "No checkpoints saved"
        print(f"\n[debug] Checkpoints created: {[os.path.basename(c) for c in ckpts]}")

        print("\nDebug train completed successfully.")


if __name__ == "__main__":
    run_debug_train()
