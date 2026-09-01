import math
from typing import Optional, Tuple, List, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from z1.config import Z1Config


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scaling_type: Optional[str] = None,
    scaling_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE (Rotary Position Embedding) cos and sin values with optional scaling for context extension.
    """
    if scaling_type == "ntk" and scaling_factor > 1.0:
        theta = theta * (scaling_factor ** (dim / (dim - 2)))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)

    if scaling_type == "linear" and scaling_factor > 1.0:
        t = t / scaling_factor

    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    xq, xk shape: [batch_size, seq_len, n_heads, head_dim]
    cos, sin shape: [seq_len, head_dim // 2]
    """
    # Reshape cos and sin for broadcasting: [1, seq_len, 1, head_dim // 2]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network with optional MatFormer (sub-network slicing) support."""

    def __init__(self, config: Z1Config):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # gate_proj
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)  # down_proj
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # up_proj

    def forward(self, x: torch.Tensor, active_dim: Optional[int] = None) -> torch.Tensor:
        if active_dim is not None and active_dim < x.shape[-1]:
            # MatFormer slice execution: slice input dimension to active_dim
            x_sliced = x[..., :active_dim]
            w1_w = self.w1.weight[: self.w1.out_features, :active_dim]
            w3_w = self.w3.weight[: self.w3.out_features, :active_dim]
            w2_w = self.w2.weight[:active_dim, : self.w2.in_features]

            gate = F.silu(F.linear(x_sliced, w1_w))
            up = F.linear(x_sliced, w3_w)
            out_sliced = F.linear(gate * up, w2_w)

            # Pad remaining dimensions if output needs full dim
            out = torch.zeros_like(x)
            out[..., :active_dim] = out_sliced
            return out
        else:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Multi-Head / Grouped-Query Attention with Rotary Embeddings."""

    def __init__(self, config: Z1Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, cos[:seqlen], sin[:seqlen])

        # Grouped-Query expand if n_kv_heads < n_heads
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        # [bsz, n_heads, seqlen, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            # Fast PyTorch SDPA path
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=mask, is_causal=(mask is None and seqlen > 1)
            )
        else:
            # Fallback scaled dot-product attention
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class Z1TransformerBlock(nn.Module):
    """Transformer block with RMSNorm, Attention, and SwiGLU FFN."""

    def __init__(self, config: Z1Config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = SwiGLUFFN(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        active_dim: Optional[int] = None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), cos, sin, mask=mask)
        out = h + self.feed_forward(self.ffn_norm(h), active_dim=active_dim)
        return out


class Z1Transformer(nn.Module):
    """z1 Decoder-Only Core Transformer."""

    def __init__(self, config: Z1Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([Z1TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Precompute RoPE cos and sin
        head_dim = config.dim // config.n_heads
        cos, sin = precompute_freqs_cis(
            dim=head_dim,
            end=config.max_seq_len,
            theta=config.rope_theta,
            scaling_type=config.rope_scaling_type,
            scaling_factor=config.rope_scaling_factor,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        active_dim: Optional[int] = None,
    ) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)

        for layer in self.layers:
            h = layer(h, self.cos, self.sin, active_dim=active_dim)

        return self.norm(h)


class Z1ForCausalLM(nn.Module):
    """z1 Model with Causal LM Head."""

    def __init__(self, config: Z1Config):
        super().__init__()
        self.config = config
        self.model = Z1Transformer(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying between embedding and lm_head
        self.lm_head.weight = self.model.tok_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        active_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.model(input_ids, active_dim=active_dim)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        active_dim: Optional[int] = None,
        pool: str = "mean",
    ) -> torch.Tensor:
        """
        Extract pooled hidden-state representations (no gradient).
        Used by probe and memory modules; backbone weights are unchanged.

        Args:
            input_ids: (B, T) token id tensor.
            active_dim: Optional MatFormer slice dimension.
            pool: 'mean' (mean over T) or 'last' (last non-pad token).

        Returns:
            Tensor of shape (B, dim).
        """
        hidden = self.model(input_ids, active_dim=active_dim)  # (B, T, dim)
        if pool == "last":
            return hidden[:, -1, :]
        return hidden.mean(dim=1)  # mean pooling over sequence length


    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with temperature and top-p sampling."""
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self.forward(idx_cond)
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

            input_ids = torch.cat((input_ids, next_token), dim=1)
            if next_token.item() == eos_token_id:
                break

        return input_ids
