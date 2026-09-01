import dataclasses
from typing import Optional, List, Dict, Any

@dataclasses.dataclass
class ZoltConfig:
    """
    Configuration for zolt (ZolttyAI) Coding-Agent and Reasoning model.
    Default architecture targets ~250M parameters (zolt) with extractable sub-network (zolt-mini) support.
    """
    vocab_size: int = 32000
    dim: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    hidden_dim: Optional[int] = 3072  # SwiGLU intermediate dimension
    max_seq_len: int = 4096           # Base sequence length (extended to 16K via RoPE scaling)
    rope_theta: float = 10000.0
    rope_scaling_type: Optional[str] = None  # None, "linear", or "ntk"
    rope_scaling_factor: float = 1.0        # e.g., 4.0 to extend 4K -> 16K
    norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # e4b / MatFormer (sparse nested activation support inspired by Gemma 3n)
    matformer_enabled: bool = True
    matformer_slices: List[int] = dataclasses.field(default_factory=lambda: [512, 1024])

    # Overtraining recipe (ratio of training tokens to model parameters, 50x-100x)
    tokens_per_param_ratio: float = 75.0

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
    search_id: int = 17
    replace_id: int = 18
    diff_end_id: int = 19
    uncertain_id: int = 20
    db_call_start_id: int = 21
    db_call_end_id: int = 22

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
    def from_dict(cls, d: Dict[str, Any]) -> "ZoltConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def preset(cls, name: str) -> "ZoltConfig":
        """Load named architecture preset ('zolt-mini' or 'zolt')."""
        normalized = name.lower().strip()
        if normalized in ("zolt-mini", "mini", "125m", "125"):
            return cls(
                dim=768,
                n_layers=12,
                n_heads=12,
                n_kv_heads=12,
                hidden_dim=2048,
                matformer_slices=[384, 768],
                tokens_per_param_ratio=75.0,
            )
        elif normalized in ("zolt", "250m", "250"):
            return cls(
                dim=1024,
                n_layers=16,
                n_heads=16,
                n_kv_heads=16,
                hidden_dim=3072,
                matformer_slices=[512, 1024],
                tokens_per_param_ratio=75.0,
            )
        else:
            raise ValueError(f"Unknown preset '{name}'. Available presets: 'zolt-mini', 'zolt'.")
