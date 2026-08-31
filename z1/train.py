"""
z1 training loop.
Supports bf16 mixed precision, AdamW, cosine LR schedule, and gradient clipping.
"""
import os
import math
import time
import json
import argparse
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from z1.config import Z1Config
from z1.model import Z1ForCausalLM
from z1.data.dataset import build_dataloader


def get_cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    lr_max: float,
    lr_min: float = 0.0,
) -> float:
    """Cosine annealing learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: Z1ForCausalLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: Z1Config,
    output_dir: str,
):
    path = Path(output_dir) / f"ckpt-step{step:07d}"
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    with open(path / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(path / "train_state.json", "w") as f:
        json.dump({"step": step, "loss": loss}, f)
    print(f"[z1-train] Checkpoint saved: {path}")


def train(
    token_files: List[str],
    output_dir: str = "./checkpoints",
    preset: Optional[str] = None,
    max_seq_len: int = 4096,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    lr: float = 3e-4,
    lr_min: float = 3e-5,
    warmup_steps: int = 500,
    total_steps: Optional[int] = None,
    tokens_per_param_ratio: float = 75.0,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    save_every: int = 1000,
    log_every: int = 50,
    curriculum: bool = False,
    resume_from: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "z1-zone-ai",
    dtype: str = "bf16",  # "bf16", "fp16", or "fp32"
):
    # ─── Setup ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[z1-train] Device: {device}")

    if preset:
        config = Z1Config.preset(preset)
        config.max_seq_len = max_seq_len
    else:
        config = Z1Config(max_seq_len=max_seq_len)

    model = Z1ForCausalLM(config).to(device)

    n_params = count_parameters(model)
    print(f"[z1-train] Parameters: {n_params:,} ({n_params/1e6:.1f}M) [preset: {preset or 'default'}]")

    # ─── Overtraining & Token Budget Calculation ────────────────────────────
    tokens_per_step = batch_size * grad_accum_steps * max_seq_len
    target_token_budget = int(n_params * tokens_per_param_ratio)
    if total_steps is None:
        total_steps = max(1, math.ceil(target_token_budget / max(1, tokens_per_step)))

    regime = "overtraining" if tokens_per_param_ratio > 30.0 else "chinchilla-optimal"
    print(
        f"[z1-train] Regime: {regime} ({tokens_per_param_ratio:.1f}x tokens/param) | "
        f"Target token budget: {target_token_budget:,} ({target_token_budget/1e9:.2f}B tokens) | "
        f"Total steps: {total_steps:,} ({tokens_per_step:,} tok/step) | Curriculum: {curriculum}"
    )

    # ─── Optimizer ──────────────────────────────────────────────────────────
    # Separate 2D+ weights (decay) and 1D biases/norms (no decay)
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ─── Mixed Precision ────────────────────────────────────────────────────
    use_amp = dtype in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    scaler = GradScaler("cuda", enabled=(dtype == "fp16" and use_amp))

    # ─── Resume ─────────────────────────────────────────────────────────────
    start_step = 0
    if resume_from:
        print(f"[z1-train] Resuming from: {resume_from}")
        model.load_state_dict(torch.load(os.path.join(resume_from, "model.pt"), map_location=device))
        optimizer.load_state_dict(torch.load(os.path.join(resume_from, "optimizer.pt"), map_location=device))
        with open(os.path.join(resume_from, "train_state.json")) as f:
            state = json.load(f)
        start_step = state["step"]

    # ─── WandB ──────────────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, config={
            "model_params": n_params,
            "max_seq_len": max_seq_len,
            "lr": lr,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "total_steps": total_steps,
            "tokens_per_param_ratio": tokens_per_param_ratio,
            "curriculum": curriculum,
        })

    # ─── DataLoader ─────────────────────────────────────────────────────────
    dataloader = build_dataloader(
        token_files=token_files,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        bos_id=config.bos_token_id,
        eos_id=config.eos_token_id,
        pad_id=config.pad_token_id,
        curriculum=curriculum,
    )

    # ─── Training Loop ──────────────────────────────────────────────────────
    model.train()
    step = start_step
    running_loss = 0.0
    t0 = time.time()

    data_iter = iter(dataloader)

    optimizer.zero_grad()

    while step < total_steps:
        # Update LR
        current_lr = get_cosine_lr(step, warmup_steps, total_steps, lr, lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Accumulate gradient
        for micro_step in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with autocast(dtype=amp_dtype):
                    _, loss = model(input_ids, labels=labels)
                loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                _, loss = model(input_ids, labels=labels)
                loss = loss / grad_accum_steps
                loss.backward()

            running_loss += loss.item()

        # Gradient clip and optimizer step
        if use_amp and dtype == "fp16":
            scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if use_amp and dtype == "fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()
        step += 1

        # ─── Logging ────────────────────────────────────────────────────────
        if step % log_every == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / log_every
            tokens_per_sec = (log_every * grad_accum_steps * batch_size * max_seq_len) / elapsed
            print(
                f"[z1-train] step={step:6d} | loss={avg_loss:.4f} | "
                f"lr={current_lr:.2e} | tok/s={tokens_per_sec:,.0f} | "
                f"elapsed={elapsed:.1f}s"
            )
            if use_wandb:
                import wandb
                wandb.log({"loss": avg_loss, "lr": current_lr, "step": step})
            running_loss = 0.0
            t0 = time.time()

        # ─── Checkpoint ─────────────────────────────────────────────────────
        if step % save_every == 0:
            save_checkpoint(model, optimizer, step, avg_loss if step % log_every == 0 else 0.0, config, output_dir)

    # Final checkpoint
    save_checkpoint(model, optimizer, step, 0.0, config, output_dir)
    print(f"[z1-train] Training complete in {step} steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="z1 Training Script")
    parser.add_argument("--token_files", nargs="+", required=True, help="Preprocessed .bin token files")
    parser.add_argument("--output_dir", default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--preset", default=None, choices=["125m", "250m"], help="Named architecture preset")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--total_steps", type=int, default=None, help="Explicit total steps (or computed from tokens_per_param_ratio)")
    parser.add_argument("--tokens_per_param_ratio", type=float, default=75.0, help="Tokens/parameter overtraining ratio (50x - 100x)")
    parser.add_argument("--curriculum", action="store_true", help="Enable progressive complexity curriculum sorting")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    train(
        token_files=args.token_files,
        output_dir=args.output_dir,
        preset=args.preset,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        tokens_per_param_ratio=args.tokens_per_param_ratio,
        curriculum=args.curriculum,
        save_every=args.save_every,
        log_every=args.log_every,
        resume_from=args.resume_from,
        dtype=args.dtype,
        use_wandb=args.wandb,
    )
