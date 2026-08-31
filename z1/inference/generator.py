"""
z1 inference and streaming generation engine.
Includes adaptive MatFormer routing, entropy-based uncertainty tagging, and ChatML prompt formatting.
"""
import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Generator, Tuple, List, Union

import torch
import torch.nn.functional as F

from z1.config import Z1Config
from z1.model import Z1ForCausalLM
from z1.tokenizer.z1_tokenizer import Z1Tokenizer


def compute_token_entropy(logits: torch.Tensor) -> float:
    """
    Compute Shannon entropy (in nats) of a logits distribution over the vocabulary.
    Higher entropy indicates higher model uncertainty.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).item()
    return entropy


def is_factual_claim_token(token_text: str) -> bool:
    """Determine whether a token represents a substantive code/factual claim vs whitespace/punctuation."""
    stripped = token_text.strip()
    if not stripped:
        return False
    # Single punctuation characters
    if len(stripped) == 1 and stripped in ",.;:()[]{}<>=+-*/\\\"'`~|&!%^@#$?":
        return False
    return True


def classify_prompt_complexity(
    prompt: str,
    small_slice: int = 512,
    large_slice: int = 1024,
) -> int:
    """
    Lightweight rule-based task classifier for adaptive MatFormer slice routing.
    Selects small_slice for quick completions and large_slice for multi-step reasoning/refactoring.
    """
    prompt_lower = prompt.lower()

    complex_indicators = (
        "refactor", "design", "architecture", "plan", "optimize", "optimization",
        "debug", "security", "vulnerability", "audit", "migrate", "migration",
        "algorithm", "concurrency", "distributed", "microservice", "explain why",
        "trade-off", "tradeoff", "benchmark", "complex", "performance", "<think>",
    )

    if any(ind in prompt_lower for ind in complex_indicators):
        return large_slice

    if len(prompt.strip()) > 250 or prompt.count("\n") >= 4:
        return large_slice

    return small_slice


class Z1Generator:
    """Inference and streaming generation engine for z1 with adaptive routing and uncertainty tagging."""

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
        active_dim: Optional[int] = None,
        auto_slice: bool = False,
        entropy_threshold: Optional[float] = None,
        model: Optional[Z1ForCausalLM] = None,
        config: Optional[Z1Config] = None,
        tokenizer: Optional[Z1Tokenizer] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is not None and config is not None:
            self.model = model.to(self.device)
            self.config = config
        elif checkpoint_dir is not None:
            checkpoint_path = Path(checkpoint_dir)
            with open(checkpoint_path / "config.json", "r") as f:
                config_dict = json.load(f)

            self.config = Z1Config.from_dict(config_dict)
            self.model = Z1ForCausalLM(self.config).to(self.device)

            model_weights = checkpoint_path / "model.pt"
            if model_weights.exists():
                state_dict = torch.load(model_weights, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[z1-inference] Loaded weights from {model_weights}")
            else:
                print("[z1-inference] Warning: 'model.pt' not found, using uninitialized weights.")
        else:
            self.config = Z1Config()
            self.model = Z1ForCausalLM(self.config).to(self.device)

        self.model.eval()

        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_path is not None and Path(tokenizer_path).exists():
            self.tokenizer = Z1Tokenizer(tokenizer_path)
        else:
            self.tokenizer = None

        self.active_dim = active_dim
        self.auto_slice = auto_slice
        self.entropy_threshold = entropy_threshold

    def format_agent_prompt(
        self,
        system_prompt: str = "You are z1, a development assistant and coding agent by zone.ai.",
        user_prompt: str = "",
        include_think_tag: bool = True,
    ) -> str:
        """Format input into ChatML prompt with optional <think> tag."""
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        if include_think_tag:
            prompt += "<think>\n"
        return prompt

    def resolve_active_dim(self, prompt: str) -> Optional[int]:
        """Resolve active MatFormer dimension based on manual setting or auto-routing."""
        if self.auto_slice:
            slices = self.config.matformer_slices or [512, 1024]
            small_s = min(slices)
            large_s = max(slices)
            return classify_prompt_complexity(prompt, small_slice=small_s, large_slice=large_s)
        return self.active_dim

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        entropy_threshold: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """Generate tokens autoregressively as a stream with uncertainty tagging."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be initialized for generation.")

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        eos_id = self.config.eos_token_id
        im_end_id = getattr(self.tokenizer, "im_end_id", None)
        active_dim = self.resolve_active_dim(prompt)
        eff_entropy_thresh = entropy_threshold if entropy_threshold is not None else self.entropy_threshold

        for _ in range(max_new_tokens):
            idx_cond = input_tensor[:, -self.config.max_seq_len :]
            logits, _ = self.model(idx_cond, active_dim=active_dim)
            step_logits = logits[:, -1, :]

            # Uncertainty entropy computation
            token_entropy = compute_token_entropy(step_logits)
            is_uncertain = (eff_entropy_thresh is not None and token_entropy > eff_entropy_thresh)

            scaled_logits = step_logits / max(temperature, 1e-5)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                scaled_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tok_id = next_token.item()
            if tok_id == eos_id or tok_id == im_end_id:
                break

            token_text = self.tokenizer.decode([tok_id], skip_special_tokens=False)

            if is_uncertain and is_factual_claim_token(token_text):
                token_text = f"<uncertain>{token_text}</uncertain>"

            yield token_text

            input_tensor = torch.cat((input_tensor, next_token), dim=1)
