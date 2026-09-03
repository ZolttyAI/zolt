"""
RoPE context extension via Linear and NTK-aware scaling.
Updates precomputed cos/sin buffers in checkpoint configs without full retraining.
"""

import json
from pathlib import Path

import torch

from zolt.config import ZoltConfig
from zolt.model import ZoltForCausalLM, precompute_freqs_cis


def apply_rope_scaling_to_checkpoint(
    checkpoint_path: str,
    output_path: str,
    target_seq_len: int = 16384,
    scaling_type: str = "ntk",
    scaling_factor: float | None = None,
):
    """Load a short-context checkpoint and apply RoPE scaling up to target_seq_len."""
    ckpt_path = Path(checkpoint_path)
    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(ckpt_path / "config.json") as f:
        config_dict = json.load(f)

    original_seq_len = config_dict["max_seq_len"]

    if scaling_factor is None:
        scaling_factor = target_seq_len / original_seq_len

    print(f"[zolt-rope] Extending context: {original_seq_len} -> {target_seq_len}")
    print(f"[zolt-rope] Method: {scaling_type}, factor: {scaling_factor:.2f}")

    # Update config
    config_dict["max_seq_len"] = target_seq_len
    config_dict["rope_scaling_type"] = scaling_type
    config_dict["rope_scaling_factor"] = scaling_factor

    config = ZoltConfig(**config_dict)

    # Load model state dict
    device = torch.device("cpu")
    model = ZoltForCausalLM(config)
    state_dict = torch.load(ckpt_path / "model.pt", map_location=device)
    model.load_state_dict(state_dict, strict=False)

    # Recompute RoPE buffers with target scaling
    head_dim = config.dim // config.n_heads
    cos, sin = precompute_freqs_cis(
        dim=head_dim,
        end=target_seq_len,
        theta=config.rope_theta,
        scaling_type=scaling_type,
        scaling_factor=scaling_factor,
    )

    model.model.register_buffer("cos", cos, persistent=False)
    model.model.register_buffer("sin", sin, persistent=False)

    # Save updated checkpoint
    torch.save(model.state_dict(), out_path / "model.pt")
    with open(out_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"[zolt-rope] Saved context-extended model ({target_seq_len}) to: {out_path}")


def ntk_scaling_factor(original_len: int, target_len: int, dim: int) -> float:
    """Compute optimal NTK-aware scaling factor."""
    return (target_len / original_len) ** (dim / (dim - 2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="zolt RoPE Context Extension")
    parser.add_argument("--checkpoint", required=True, help="Source checkpoint directory")
    parser.add_argument("--output", required=True, help="Output checkpoint directory")
    parser.add_argument("--target_len", type=int, default=16384, help="Target sequence length")
    parser.add_argument("--method", default="ntk", choices=["linear", "ntk"], help="Scaling method")
    parser.add_argument("--factor", type=float, default=None, help="Explicit scaling factor")
    args = parser.parse_args()

    apply_rope_scaling_to_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        target_seq_len=args.target_len,
        scaling_type=args.method,
        scaling_factor=args.factor,
    )
