"""Unit tests for z1 model architecture."""
import pytest
import torch
from z1.config import Z1Config
from z1.model import (
    Z1ForCausalLM,
    Z1Transformer,
    RMSNorm,
    SwiGLUFFN,
    Attention,
    precompute_freqs_cis,
    apply_rotary_emb,
)


@pytest.fixture
def tiny_config():
    """Minimal config for fast CPU execution."""
    return Z1Config(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=128,
        max_seq_len=128,
        matformer_enabled=True,
        matformer_slices=[32, 64],
    )


def test_rmsnorm_output_shape(tiny_config):
    norm = RMSNorm(tiny_config.dim)
    x = torch.randn(2, 16, tiny_config.dim)
    out = norm(x)
    assert out.shape == x.shape, "RMSNorm should preserve input shape"


def test_rmsnorm_unit_variance(tiny_config):
    norm = RMSNorm(tiny_config.dim)
    x = torch.randn(1, 1, tiny_config.dim) * 100
    out = norm(x)
    # With unit initial weights, RMS should approximate 1.0
    rms = out.pow(2).mean(-1).sqrt()
    assert (rms - 1.0).abs().mean() < 0.2


def test_rope_precompute(tiny_config):
    head_dim = tiny_config.dim // tiny_config.n_heads
    cos, sin = precompute_freqs_cis(head_dim, tiny_config.max_seq_len)
    assert cos.shape == (tiny_config.max_seq_len, head_dim // 2)
    assert sin.shape == (tiny_config.max_seq_len, head_dim // 2)


def test_rope_scaling_ntk(tiny_config):
    head_dim = tiny_config.dim // tiny_config.n_heads
    cos, sin = precompute_freqs_cis(head_dim, tiny_config.max_seq_len, scaling_type="ntk", scaling_factor=4.0)
    assert cos.shape == (tiny_config.max_seq_len, head_dim // 2)


def test_attention_forward(tiny_config):
    attn = Attention(tiny_config)
    head_dim = tiny_config.dim // tiny_config.n_heads
    cos, sin = precompute_freqs_cis(head_dim, tiny_config.max_seq_len)
    x = torch.randn(2, 16, tiny_config.dim)
    out = attn(x, cos, sin)
    assert out.shape == (2, 16, tiny_config.dim)


def test_swiglu_ffn(tiny_config):
    ffn = SwiGLUFFN(tiny_config)
    x = torch.randn(2, 16, tiny_config.dim)
    out = ffn(x)
    assert out.shape == x.shape


def test_swiglu_ffn_matformer_slice(tiny_config):
    ffn = SwiGLUFFN(tiny_config)
    x = torch.randn(2, 16, tiny_config.dim)
    out = ffn(x, active_dim=32)
    assert out.shape == x.shape, "MatFormer slice must return full hidden dimension"
    # Non-active dimensions must be zeroed
    assert out[..., 32:].abs().sum() == 0


def test_z1transformer_forward(tiny_config):
    model = Z1Transformer(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    out = model(input_ids)
    assert out.shape == (2, 16, tiny_config.dim)


def test_z1_causal_lm_loss(tiny_config):
    model = Z1ForCausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    labels = torch.randint(0, tiny_config.vocab_size, (2, 16))
    logits, loss = model(input_ids, labels=labels)
    assert logits.shape == (2, 16, tiny_config.vocab_size)
    assert loss is not None
    assert loss.item() > 0


def test_z1_no_loss_without_labels(tiny_config):
    model = Z1ForCausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
    logits, loss = model(input_ids)
    assert logits.shape == (1, 8, tiny_config.vocab_size)
    assert loss is None


def test_z1_weight_tying(tiny_config):
    model = Z1ForCausalLM(tiny_config)
    assert model.lm_head.weight is model.model.tok_embeddings.weight, \
        "lm_head and tok_embeddings must share weights"


def test_z1_generate(tiny_config):
    model = Z1ForCausalLM(tiny_config)
    model.eval()
    input_ids = torch.tensor([[1, 10, 20]])  # [bos, tok, tok]
    out = model.generate(input_ids, max_new_tokens=5, temperature=1.0)
    assert out.shape[1] > input_ids.shape[1]
    assert out.shape[1] <= input_ids.shape[1] + 5


def test_parameter_count(tiny_config):
    model = Z1ForCausalLM(tiny_config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 0
    print(f"\nParameters (tiny_config): {n_params:,}")
