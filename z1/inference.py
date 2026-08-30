"""
z1 inference and streaming generation CLI.
"""
import os
import json
import argparse
from pathlib import Path
from typing import Optional, Generator

import torch
import torch.nn.functional as F

from z1.config import Z1Config
from z1.model import Z1ForCausalLM
from z1.tokenizer.z1_tokenizer import Z1Tokenizer


class Z1Generator:
    """Inference and streaming generation engine for z1."""

    def __init__(
        self,
        checkpoint_dir: str,
        tokenizer_path: str,
        device: Optional[str] = None,
        active_dim: Optional[int] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

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

        self.model.eval()
        self.tokenizer = Z1Tokenizer(tokenizer_path)
        self.active_dim = active_dim

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

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """Generate tokens autoregressively as a stream."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        eos_id = self.config.eos_token_id
        im_end_id = getattr(self.tokenizer, "im_end_id", None)

        for _ in range(max_new_tokens):
            idx_cond = input_tensor[:, -self.config.max_seq_len :]
            logits, _ = self.model(idx_cond, active_dim=self.active_dim)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tok_id = next_token.item()
            if tok_id == eos_id or tok_id == im_end_id:
                break

            token_text = self.tokenizer.decode([tok_id], skip_special_tokens=False)
            yield token_text

            input_tensor = torch.cat((input_tensor, next_token), dim=1)


def main():
    parser = argparse.ArgumentParser(description="z1 Interactive CLI")
    parser.add_argument("--checkpoint", required=True, help="Path to z1 checkpoint directory")
    parser.add_argument("--tokenizer", default="z1_tokenizer.json", help="Path to BPE tokenizer")
    parser.add_argument("--active_dim", type=int, default=None, help="Active dimension for MatFormer slice (e.g. 384)")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()

    generator = Z1Generator(
        checkpoint_dir=args.checkpoint,
        tokenizer_path=args.tokenizer,
        active_dim=args.active_dim,
    )

    print("=" * 60)
    print("z1 (zone.ai) - Coding Agent & Reasoning Assistant")
    print("Type 'exit' or 'quit' to end session.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n[User] > ")
            if user_input.strip().lower() in ("sair", "exit", "quit"):
                break
            if not user_input.strip():
                continue

            prompt = generator.format_agent_prompt(user_prompt=user_input)
            print("\n[z1] > ", end="", flush=True)

            for chunk in generator.generate_stream(prompt, temperature=args.temp, top_p=args.top_p):
                print(chunk, end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
