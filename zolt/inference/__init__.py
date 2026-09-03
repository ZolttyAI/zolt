"""
zolt Inference Package.
Exports generators, diff formatting, verification (TS/JS/Python), and structured DB calls.
"""

import argparse

from zolt.inference.db_call import (
    SUPPORTED_DIALECTS,
    SUPPORTED_OPERATIONS,
    DBCallPayload,
    format_db_call,
    parse_db_calls,
    validate_db_call,
)
from zolt.inference.diff_format import (
    DiffEdit,
    apply_diff_block,
    apply_diff_edit,
    format_diff_block,
    parse_diff_blocks,
)
from zolt.inference.generator import (
    Z1Generator,
    ZoltGenerator,
    classify_prompt_complexity,
    compute_token_entropy,
    is_factual_claim_token,
)
from zolt.inference.verify import (
    SUPPORTED_LANGUAGES,
    self_correcting_generate,
    verify_code,
)
from zolt.inference.verify_base import (
    VerifyResult,
    extract_code_block,
)
from zolt.inference.verify_base import (
    self_correcting_generate as self_correcting_generate_base,
)
from zolt.inference.verify_js import (
    self_correcting_generate_js,
    verify_javascript_code,
)
from zolt.inference.verify_python import (
    self_correcting_generate_python,
    verify_python_code,
)
from zolt.inference.verify_ts import (
    run_tsc_check,
    self_correcting_generate_ts,
    verify_typescript_code,
)

__all__ = [
    # Generator
    "ZoltGenerator",
    "Z1Generator",
    "classify_prompt_complexity",
    "compute_token_entropy",
    "is_factual_claim_token",
    # Diff format
    "parse_diff_blocks",
    "apply_diff_edit",
    "apply_diff_block",
    "format_diff_block",
    "DiffEdit",
    # Verification base
    "VerifyResult",
    "extract_code_block",
    "self_correcting_generate_base",
    # Verifiers
    "verify_typescript_code",
    "self_correcting_generate_ts",
    "run_tsc_check",
    "verify_javascript_code",
    "self_correcting_generate_js",
    "verify_python_code",
    "self_correcting_generate_python",
    # Dispatcher
    "verify_code",
    "self_correcting_generate",
    "SUPPORTED_LANGUAGES",
    # DB call
    "validate_db_call",
    "parse_db_calls",
    "format_db_call",
    "DBCallPayload",
    "SUPPORTED_DIALECTS",
    "SUPPORTED_OPERATIONS",
    "main",
]


def main():
    parser = argparse.ArgumentParser(description="zolt Interactive CLI")
    parser.add_argument("--checkpoint", required=True, help="Path to zolt checkpoint directory")
    parser.add_argument("--tokenizer", default="zolt_tokenizer.json", help="Path to BPE tokenizer")
    parser.add_argument(
        "--slice",
        "--active_dim",
        dest="active_dim",
        type=int,
        default=None,
        help="Active dimension for MatFormer slice (e.g. 384, 512, 768, 1024)",
    )
    parser.add_argument(
        "--auto-slice",
        "--auto_slice",
        dest="auto_slice",
        action="store_true",
        help="Enable adaptive MatFormer slice routing based on prompt complexity",
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=None,
        help="Output entropy threshold for <uncertain> tag injection",
    )
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()

    generator = ZoltGenerator(
        checkpoint_dir=args.checkpoint,
        tokenizer_path=args.tokenizer,
        active_dim=args.active_dim,
        auto_slice=args.auto_slice,
        entropy_threshold=args.entropy_threshold,
    )

    print("=" * 60)
    print("zolt (ZolttyAI) - Coding Agent & Reasoning Assistant")
    print(
        f"Active slice mode: {'Auto-routing' if args.auto_slice else (args.active_dim or 'Full model')}"
    )
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
            print("\n[zolt] > ", end="", flush=True)

            for chunk in generator.generate_stream(prompt, temperature=args.temp, top_p=args.top_p):
                print(chunk, end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
