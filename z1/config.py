import dataclasses
from typing import Optional, List, Dict, Any

@dataclasses.dataclass
class Z1Config:
    """
    Configuration for z1 (zone.ai) 125M Coding-Agent and Reasoning model.
    """
    vocab_size: int = 32000
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    hidden_dim: Optional[int] = 2048  # SwiGLU intermediate dimension
    max_seq_len: int = 4096            # Base sequence length (extended to 16K via RoPE scaling)
    rope_theta: float = 10000.0
    rope_scaling_type: Optional[str] = None  # None, "linear", or "ntk"
    rope_scaling_factor: float = 1.0        # e.g., 4.0 to extend 4K -> 16K
    norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # e4b / MatFormer (sparse nested activation support inspired by Gemma 3n)
    matformer_enabled: bool = True
    matformer_slices: List[int] = dataclasses.field(default_factory=lambda: [384, 768])

    # Special Tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    think_start_id: int = 4
    think_end_id: int = 5
    tool_call_start_id: int = 6
    tool_call_end_id: int = 7
    tool_resp_start_id: int = 8
    tool_resp_end_id: int = 9

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.hidden_dim is None:
            # Standard Llama SwiGLU hidden dim formula: 2/3 * 4 * dim aligned to 256
            hidden = int(2 * 4 * self.dim / 3)
            self.hidden_dim = ((hidden + 255) // 256) * 256

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Z1Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
