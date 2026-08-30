#!/usr/bin/env python3
"""
z1 smoke test: CPU validation of model architecture, RoPE scaling, and components.
"""
import sys

def run_smoke_test():
    print("=" * 60)
    print("z1 zone.ai - Smoke Test")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed. Run: uv pip install torch --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)

    from z1.config import Z1Config
    from z1.model import Z1ForCausalLM

    # Minimal config for fast CPU execution
    config = Z1Config(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=128,
        max_seq_len=128,
    )
    print(f"✓ Z1Config loaded")

    model = Z1ForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Z1ForCausalLM instantiated | params: {n_params:,} ({n_params/1e6:.3f}M) [tiny config]")

    # Parameter count for default 125M architecture
    real_config = Z1Config()
    real_model = Z1ForCausalLM(real_config)
    real_params = sum(p.numel() for p in real_model.parameters())
    print(f"✓ Target 125M config | params: {real_params:,} ({real_params/1e6:.1f}M)")

    # Forward pass
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(input_ids, labels=labels)
    print(f"✓ Forward pass OK | logits: {logits.shape} | loss: {loss.item():.4f}")

    # MatFormer slice execution
    logits_sliced, loss_sliced = model(input_ids, labels=labels, active_dim=32)
    print(f"✓ MatFormer slice (active_dim=32) OK | loss: {loss_sliced.item():.4f}")

    # Autoregressive generation
    model.eval()
    seed = torch.tensor([[1, 10, 20]])
    out = model.generate(seed, max_new_tokens=5, temperature=0.8)
    print(f"✓ Generate OK | generated tokens: {out.shape[1] - seed.shape[1]}")

    # Rotary Position Embedding scaling
    from z1.model import precompute_freqs_cis
    head_dim = config.dim // config.n_heads
    cos_linear, sin_linear = precompute_freqs_cis(head_dim, 512, scaling_type="linear", scaling_factor=4.0)
    cos_ntk, sin_ntk = precompute_freqs_cis(head_dim, 512, scaling_type="ntk", scaling_factor=4.0)
    print(f"✓ RoPE Linear scaling OK | cos shape: {cos_linear.shape}")
    print(f"✓ RoPE NTK scaling OK    | cos shape: {cos_ntk.shape}")

    # Evaluation utilities
    from z1.eval import check_python_syntax, check_javascript_syntax_heuristic, check_reasoning_tags
    py_ok = check_python_syntax("def foo():\n    return 42")
    py_bad = check_python_syntax("def foo(\n    return 42")
    js_ok = check_javascript_syntax_heuristic("const f = () => { return [1, 2]; };")
    tags_ok = check_reasoning_tags("<think>\nreasoning block\n</think>\nresponse")
    tags_bad = check_reasoning_tags("<think>\nunclosed reasoning")
    assert py_ok["valid"], "Valid Python syntax should pass"
    assert not py_bad["valid"], "Invalid Python syntax should fail"
    assert js_ok["valid"], "Valid JS syntax should pass"
    assert tags_ok["valid"], "Balanced tags should pass"
    assert not tags_bad["valid"], "Unbalanced tags should fail"
    print(f"✓ Eval helpers OK (syntax + reasoning tags)")

    # Data filtering helpers
    from z1.data.filter_code import is_permissive_license, is_target_language, passes_quality_heuristics
    assert is_permissive_license("MIT")
    assert is_target_language("typescript")
    assert not is_target_language("java")
    print(f"✓ Data filters OK")

    # Tokenizer special tokens
    from z1.tokenizer.train_tokenizer import Z1_SPECIAL_TOKENS
    assert "<think>" in Z1_SPECIAL_TOKENS
    assert "<tool_call>" in Z1_SPECIAL_TOKENS
    assert len(Z1_SPECIAL_TOKENS) == len(set(Z1_SPECIAL_TOKENS))
    print(f"✓ Special tokens OK | {len(Z1_SPECIAL_TOKENS)} tokens")

    print()
    print("=" * 60)
    print("✓ All smoke tests passed. z1 is ready.")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
