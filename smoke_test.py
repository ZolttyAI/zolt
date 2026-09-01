#!/usr/bin/env python3
"""
zolt smoke test: CPU validation of model architecture, RoPE scaling, and components.
"""
import sys

def run_smoke_test():
    print("=" * 60)
    print("zolt (zolt.ai) - Smoke Test")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed. Run: uv pip install torch --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)

    from zolt.config import ZoltConfig
    from zolt.model import ZoltForCausalLM

    # Minimal config for fast CPU execution
    config = ZoltConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=128,
        max_seq_len=128,
    )
    print(f"✓ ZoltConfig loaded")

    model = ZoltForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ ZoltForCausalLM instantiated | params: {n_params:,} ({n_params/1e6:.3f}M) [tiny config]")

    # Parameter counts for presets and default 250M architecture
    real_config_250 = ZoltConfig()
    real_model_250 = ZoltForCausalLM(real_config_250)
    real_params_250 = sum(p.numel() for p in real_model_250.parameters())
    print(f"✓ Target 250M default config (zolt) | params: {real_params_250:,} ({real_params_250/1e6:.1f}M)")

    preset_mini = ZoltConfig.preset("zolt-mini")
    model_mini = ZoltForCausalLM(preset_mini)
    params_mini = sum(p.numel() for p in model_mini.parameters())
    print(f"✓ Preset zolt-mini config            | params: {params_mini:,} ({params_mini/1e6:.1f}M)")

    preset_zolt = ZoltConfig.preset("zolt")
    model_zolt = ZoltForCausalLM(preset_zolt)
    params_zolt = sum(p.numel() for p in model_zolt.parameters())
    print(f"✓ Preset zolt config                 | params: {params_zolt:,} ({params_zolt/1e6:.1f}M)")

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
    from zolt.model import precompute_freqs_cis
    head_dim = config.dim // config.n_heads
    cos_linear, sin_linear = precompute_freqs_cis(head_dim, 512, scaling_type="linear", scaling_factor=4.0)
    cos_ntk, sin_ntk = precompute_freqs_cis(head_dim, 512, scaling_type="ntk", scaling_factor=4.0)
    print(f"✓ RoPE Linear scaling OK | cos shape: {cos_linear.shape}")
    print(f"✓ RoPE NTK scaling OK    | cos shape: {cos_ntk.shape}")

    # Evaluation utilities
    from zolt.eval import check_python_syntax, check_javascript_syntax_heuristic, check_reasoning_tags
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
    from zolt.data.filter_code import (
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
    from zolt.data.curriculum import estimate_code_complexity, estimate_token_sequence_complexity, sort_by_curriculum
    c_easy = estimate_code_complexity("def a(): return 1")
    c_hard = estimate_code_complexity("class Engine:\n    def run(self):\n        for i in range(10):\n            if i % 2 == 0:\n                try:\n                    pass\n                except Exception:\n                    pass")
    assert c_easy < c_hard
    sorted_items = sort_by_curriculum(["def hard_func():\n    if True:\n        for x in y:\n            return x", "x = 1"])
    assert sorted_items[0] == "x = 1"
    print(f"✓ Curriculum staging OK | easy: {c_easy:.1f}, hard: {c_hard:.1f}")

    # Teacher distillation synthetic generation
    from zolt.data.distill import generate_synthetic_instance, mix_datasets
    synth = generate_synthetic_instance(None, {"lang": "python", "topic": "sorting"}, mock_mode=True)
    assert "<think>" in synth["content"]
    assert synth["license"] == "synthetic"
    print(f"✓ Teacher distillation pipeline OK")

    # Tokenizer special tokens
    from zolt.tokenizer.train_tokenizer import ZOLT_SPECIAL_TOKENS
    assert "<think>" in ZOLT_SPECIAL_TOKENS
    assert "<tool_call>" in ZOLT_SPECIAL_TOKENS
    assert "<search>" in ZOLT_SPECIAL_TOKENS
    assert "<replace>" in ZOLT_SPECIAL_TOKENS
    assert "<diff_end>" in ZOLT_SPECIAL_TOKENS
    assert "<uncertain>" in ZOLT_SPECIAL_TOKENS
    assert "<db_call>" in ZOLT_SPECIAL_TOKENS
    assert "</db_call>" in ZOLT_SPECIAL_TOKENS
    assert len(ZOLT_SPECIAL_TOKENS) == 23
    print(f"✓ Special tokens OK | {len(ZOLT_SPECIAL_TOKENS)} tokens")

    # Native diff format
    from zolt.inference.diff_format import apply_diff_edit, parse_diff_blocks
    diff_sample = "<search>\nfoo = 1\n<replace>\nfoo = 2\n<diff_end>"
    edits = parse_diff_blocks(diff_sample)
    assert len(edits) == 1
    assert apply_diff_edit("foo = 1\nbar = 2", edits[0].search, edits[0].replace) == "foo = 2\nbar = 2"
    print("✓ Native diff format parser OK")

    # TypeScript verification
    from zolt.inference.verify_ts import verify_typescript_code
    assert verify_typescript_code("const x: number = 42;")["valid"]
    assert not verify_typescript_code("const x = [1, 2;")["valid"]
    print("✓ TypeScript self-verification OK")

    # Structured DB call
    from zolt.inference.db_call import validate_db_call
    assert validate_db_call({"dialect": "postgresql", "operation": "select", "table": "users", "constraints": {}})["valid"]
    assert not validate_db_call({"dialect": "mongodb", "operation": "select", "table": "users", "constraints": {}})["valid"]
    print("✓ Structured DB call schema validation OK")

    print()
    print("=" * 60)
    print("✓ All smoke tests passed. zolt is ready.")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
