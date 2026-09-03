"""
Tokenizer interface with zolt special tokens for reasoning and coding agent interactions.
"""

from __future__ import annotations

from tokenizers import Tokenizer as HFTokenizer

SPECIAL_TOKEN_MAP = {
    "pad_token": "<pad>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
    "think_start": "<think>",
    "think_end": "</think>",
    "tool_call_start": "<tool_call>",
    "tool_call_end": "</tool_call>",
    "tool_resp_start": "<tool_response>",
    "tool_resp_end": "</tool_response>",
    "code_start": "<code>",
    "code_end": "</code>",
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "fill": "<FILL>",
    "prefix": "<PREFIX>",
    "suffix": "<SUFFIX>",
    "search": "<search>",
    "replace": "<replace>",
    "diff_end": "<diff_end>",
    "uncertain": "<uncertain>",
    "db_call_start": "<db_call>",
    "db_call_end": "</db_call>",
}


class ZoltTokenizer:
    """BPE tokenizer wrapper with special token IDs for zolt."""

    def __init__(self, tokenizer_path: str):
        self._tokenizer = HFTokenizer.from_file(tokenizer_path)
        self._build_special_token_ids()

    def _build_special_token_ids(self):
        vocab = self._tokenizer.get_vocab()
        for attr, token in SPECIAL_TOKEN_MAP.items():
            tid = vocab.get(token, None)
            setattr(self, f"{attr}_id", tid)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        enc = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return enc.ids

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(self, texts: list[str], add_special_tokens: bool = True) -> list[list[int]]:
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=32768)
        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]

    def wrap_thinking(self, reasoning_text: str, answer_text: str) -> str:
        """Wrap reasoning text in <think> and </think> tags."""
        return f"<think>\n{reasoning_text}\n</think>\n{answer_text}"

    def wrap_tool_call(self, tool_json: str, response_json: str) -> str:
        """Wrap tool call and response JSON in zolt tool tags."""
        return f"<tool_call>\n{tool_json}\n</tool_call>\n<tool_response>\n{response_json}\n</tool_response>"

    def wrap_chat(self, role: str, content: str) -> str:
        """Wrap message in ChatML format."""
        return f"<|im_start|>{role}\n{content}<|im_end|>"

    def __repr__(self) -> str:
        return f"ZoltTokenizer(vocab_size={self.vocab_size})"

    @classmethod
    def from_pretrained(cls, path: str) -> ZoltTokenizer:
        return cls(path)


# Backward compatibility alias if needed
Z1Tokenizer = ZoltTokenizer
