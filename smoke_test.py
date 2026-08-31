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

    # Parameter counts for presets and default 250M architecture
    real_config_250 = Z1Config()
    real_model_250 = Z1ForCausalLM(real_config_250)
    real_params_250 = sum(p.numel() for p in real_model_250.parameters())
    print(f"✓ Target 250M default config | params: {real_params_250:,} ({real_params_250/1e6:.1f}M)")

    preset_125 = Z1Config.preset("125m")
    model_125 = Z1ForCausalLM(preset_125)
    params_125 = sum(p.numel() for p in model_125.parameters())
    print(f"✓ Preset 125M config        | params: {params_125:,} ({params_125/1e6:.1f}M)")

    preset_250 = Z1Config.preset("250m")
    model_250 = Z1ForCausalLM(preset_250)
    params_250 = sum(p.numel() for p in model_250.parameters())
    print(f"✓ Preset 250M config        | params: {params_250:,} ({params_250/1e6:.1f}M)")

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

    # Data filtering and textbook quality score helpers
    from z1.data.filter_code import (
        is_permissive_license,
        is_target_language,
        passes_quality_heuristics,
        compute_textbook_quality_score,
    )
    assert is_permissive_license("MIT")
    assert is_target_language("typescript")
    assert not is_target_language("java")
    sample_code = '"""Module documentation."""\ndef calculate(x: int) -> int:\n    if x > 0:\n        return x * 2\n    return 0\n'
    q_score = compute_textbook_quality_score(sample_code, lang="python", path_or_repo="tests/test_calc.py")
    assert 0.0 <= q_score <= 1.0
    print(f"✓ Data quality scoring OK | sample score: {q_score:.3f}")

    # Curriculum learning and complexity proxy
    from z1.data.curriculum import estimate_code_complexity, estimate_token_sequence_complexity, sort_by_curriculum
    c_easy = estimate_code_complexity("def a(): return 1")
    c_hard = estimate_code_complexity("class Engine:\n    def run(self):\n        for i in range(10):\n            if i % 2 == 0:\n                try:\n                    pass\n                except Exception:\n                    pass")
    assert c_easy < c_hard
    sorted_items = sort_by_curriculum(["def hard_func():\n    if True:\n        for x in y:\n            return x", "x = 1"])
    assert sorted_items[0] == "x = 1"
    print(f"✓ Curriculum staging OK | easy: {c_easy:.1f}, hard: {c_hard:.1f}")

    # Teacher distillation synthetic generation
    from z1.data.distill import generate_synthetic_instance, mix_datasets
    synth = generate_synthetic_instance(None, {"lang": "python", "topic": "sorting"}, mock_mode=True)
    assert "<think>" in synth["content"]
    assert synth["license"] == "synthetic"
    print(f"✓ Teacher distillation pipeline OK")

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
