"""Unit tests for adaptive MatFormer routing and explicit uncertainty tagging."""

import torch

from zolt.config import ZoltConfig
from zolt.inference.generator import (
    ZoltGenerator,
    classify_prompt_complexity,
    compute_token_entropy,
    is_factual_claim_token,
)


def test_classify_prompt_complexity_simple():
    simple_prompts = [
        "def add(a, b):",
        "complete: const x =",
        "function square(n) { return",
        "print hello world in python",
    ]
    for p in simple_prompts:
        slice_dim = classify_prompt_complexity(p, small_slice=384, large_slice=768)
        assert slice_dim == 384, f"Prompt '{p}' should route to small slice 384"


def test_classify_prompt_complexity_complex():
    complex_prompts = [
        "refactor this monolithic database layer into repository patterns",
        "design a high-performance distributed caching architecture",
        "explain why this concurrent async queue deadlocks under heavy load",
        "plan a migration from React class components to Next.js App Router",
        "optimize the time complexity of this dynamic programming algorithm",
    ]
    for p in complex_prompts:
        slice_dim = classify_prompt_complexity(p, small_slice=384, large_slice=768)
        assert slice_dim == 768, f"Prompt '{p}' should route to large slice 768"


def test_compute_token_entropy_sharp_vs_flat():
    # Sharp distribution (deterministic token) -> near 0 entropy
    sharp_logits = torch.tensor([[100.0, -100.0, -100.0, -100.0]])
    sharp_entropy = compute_token_entropy(sharp_logits)
    assert sharp_entropy < 0.01

    # Flat uniform distribution -> high entropy (ln(4) ≈ 1.386)
    flat_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    flat_entropy = compute_token_entropy(flat_logits)
    assert 1.30 <= flat_entropy <= 1.45


def test_is_factual_claim_token():
    assert is_factual_claim_token("calculateSum")
    assert is_factual_claim_token("User")
    assert is_factual_claim_token("1024")
    assert is_factual_claim_token("import")

    # Whitespace and punctuation
    assert not is_factual_claim_token(" ")
    assert not is_factual_claim_token("\n")
    assert not is_factual_claim_token("   ")
    assert not is_factual_claim_token(";")
    assert not is_factual_claim_token("(")
    assert not is_factual_claim_token("}")
    assert not is_factual_claim_token(",")


class MockTokenizer:
    """Mock tokenizer for testing uncertainty tag generation."""

    def __init__(self):
        self.im_end_id = 999
        self.vocab = {"def": 1, " ": 2, "compute": 3, "<eos>": 4}

    def encode(self, text, add_special_tokens=False):
        return [1]

    def decode(self, ids, skip_special_tokens=False):
        id_map = {1: "def", 2: " ", 3: "compute", 4: "<eos>"}
        return "".join(id_map.get(i, "tok") for i in ids)


class MockModel(torch.nn.Module):
    """Mock model returning controllable logits for entropy testing."""

    def __init__(self, high_uncertainty: bool):
        super().__init__()
        self.high_uncertainty = high_uncertainty

    def forward(self, idx, active_dim=None):
        if self.high_uncertainty:
            # Uniform logits -> high entropy
            logits = torch.zeros(idx.shape[0], idx.shape[1], 100)
        else:
            # Sharp logits -> low entropy
            logits = torch.full((idx.shape[0], idx.shape[1], 100), -50.0)
            logits[:, :, 3] = 50.0
        return logits, None


def test_generator_uncertainty_tag_fires_above_threshold():
    cfg = ZoltConfig(vocab_size=100, max_seq_len=64, eos_token_id=4)
    model = MockModel(high_uncertainty=True)
    tokenizer = MockTokenizer()

    generator = ZoltGenerator(
        model=model,
        config=cfg,
        tokenizer=tokenizer,
        entropy_threshold=2.0,
    )

    chunks = list(generator.generate_stream("test", max_new_tokens=1))
    assert len(chunks) == 1
    # Because entropy is high on factual token "compute", it should be wrapped with <uncertain>
    assert "<uncertain>" in chunks[0]
    assert "</uncertain>" in chunks[0]


def test_generator_uncertainty_tag_does_not_fire_below_threshold():
    cfg = ZoltConfig(vocab_size=100, max_seq_len=64, eos_token_id=4)
    model = MockModel(high_uncertainty=False)
    tokenizer = MockTokenizer()

    generator = ZoltGenerator(
        model=model,
        config=cfg,
        tokenizer=tokenizer,
        entropy_threshold=2.0,
    )

    chunks = list(generator.generate_stream("test", max_new_tokens=1))
    assert len(chunks) == 1
    # Low entropy -> no <uncertain> tag
    assert "<uncertain>" not in chunks[0]
