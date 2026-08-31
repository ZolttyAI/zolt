"""
z1 Inference Package.
Exports generators, diff formatting, TypeScript verification, and structured DB calls.
"""
import argparse
from z1.inference.generator import (
    Z1Generator,
    classify_prompt_complexity,
    compute_token_entropy,
    is_factual_claim_token,
)
from z1.inference.diff_format import (
    parse_diff_blocks,
    apply_diff_edit,
    apply_diff_block,
    format_diff_block,
    DiffEdit,
)
from z1.inference.verify_ts import (
    verify_typescript_code,
    self_correcting_generate_ts,
    run_tsc_check,
)
from z1.inference.db_call import (
    validate_db_call,
    parse_db_calls,
    format_db_call,
    DBCallPayload,
    SUPPORTED_DIALECTS,
    SUPPORTED_OPERATIONS,
)

__all__ = [
    "Z1Generator",
    "classify_prompt_complexity",
    "compute_token_entropy",
    "is_factual_claim_token",
    "parse_diff_blocks",
    "apply_diff_edit",
    "apply_diff_block",
    "format_diff_block",
    "DiffEdit",
    "verify_typescript_code",
    "self_correcting_generate_ts",
    "run_tsc_check",
    "validate_db_call",
    "parse_db_calls",
    "format_db_call",
    "DBCallPayload",
    "SUPPORTED_DIALECTS",
    "SUPPORTED_OPERATIONS",
    "main",
]


def main():
    parser = argparse.ArgumentParser(description="z1 Interactive CLI")
    parser.add_argument("--checkpoint", required=True, help="Path to z1 checkpoint directory")
    parser.add_argument("--tokenizer", default="z1_tokenizer.json", help="Path to BPE tokenizer")
    parser.add_argument("--slice", "--active_dim", dest="active_dim", type=int, default=None, help="Active dimension for MatFormer slice (e.g. 384, 512, 768, 1024)")
    parser.add_argument("--auto-slice", "--auto_slice", dest="auto_slice", action="store_true", help="Enable adaptive MatFormer slice routing based on prompt complexity")
    parser.add_argument("--entropy_threshold", type=float, default=None, help="Output entropy threshold for explicit <uncertain> tag injection")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()

    generator = Z1Generator(
        checkpoint_dir=args.checkpoint,
        tokenizer_path=args.tokenizer,
        active_dim=args.active_dim,
        auto_slice=args.auto_slice,
        entropy_threshold=args.entropy_threshold,
    )

    print("=" * 60)
    print("z1 (zone.ai) - Coding Agent & Reasoning Assistant")
    print(f"Active slice mode: {'Auto-routing' if args.auto_slice else (args.active_dim or 'Full model')}")
    if args.entropy_threshold is not None:
        print(f"Uncertainty tagging: Enabled (threshold={args.entropy_threshold})")
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
