"""
z1 evaluation: loss, perplexity, code syntax validation, and reasoning tag balance.
"""
import ast
import json
import math
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F

from z1.config import Z1Config
from z1.model import Z1ForCausalLM


def compute_perplexity(
    model: Z1ForCausalLM,
    token_sequences: List[List[int]],
    device: torch.device,
    max_seq_len: int = 4096,
) -> Tuple[float, float]:
    """Compute average cross-entropy loss and perplexity across token sequences."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for seq in token_sequences:
            if len(seq) < 2:
                continue

            seq = seq[:max_seq_len + 1]
            input_ids = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
            labels = torch.tensor([seq[1:]], dtype=torch.long, device=device)

            _, loss = model(input_ids, labels=labels)
            if loss is not None:
                n_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20))  # cap to avoid overflow
    return avg_loss, ppl


def check_python_syntax(code: str) -> Dict[str, Any]:
    """Validate Python syntax with ast.parse."""
    try:
        ast.parse(code)
        return {"valid": True, "error": None}
    except SyntaxError as e:
        return {"valid": False, "error": str(e)}


def check_javascript_syntax_heuristic(code: str) -> Dict[str, Any]:
    """Heuristic bracket and string balance check for JS/TS code snippets."""
    stack = []
    pairs = {")": "(", "}": "{", "]": "["}
    in_string = None
    i = 0

    while i < len(code):
        c = code[i]

        if in_string:
            if c == "\\" and i + 1 < len(code):
                i += 2
                continue
            if c == in_string:
                in_string = None
        elif c in ('"', "'", "`"):
            in_string = c
        elif c in ("(", "{", "["):
            stack.append(c)
        elif c in (")", "}", "]"):
            if not stack or stack[-1] != pairs[c]:
                return {"valid": False, "error": f"Unbalanced bracket '{c}' at position {i}"}
            stack.pop()
        i += 1

    if stack:
        return {"valid": False, "error": f"Unclosed brackets: {stack}"}
    if in_string:
        return {"valid": False, "error": f"Unterminated string: {in_string}"}
    return {"valid": True, "error": None}


def check_reasoning_tags(text: str) -> Dict[str, Any]:
    """Verify that reasoning and tool call tags are balanced."""
    tag_pairs = [
        ("<think>", "</think>"),
        ("<tool_call>", "</tool_call>"),
        ("<tool_response>", "</tool_response>"),
        ("<code>", "</code>"),
    ]
    issues = []

    for open_tag, close_tag in tag_pairs:
        opens = len(re.findall(re.escape(open_tag), text))
        closes = len(re.findall(re.escape(close_tag), text))
        if opens != closes:
            issues.append(f"Unbalanced tag: {opens}x {open_tag} vs {closes}x {close_tag}")

    return {"valid": len(issues) == 0, "issues": issues}


def evaluate_checkpoint(
    checkpoint_dir: str,
    eval_sequences: List[List[int]],
    code_samples_python: Optional[List[str]] = None,
    code_samples_js: Optional[List[str]] = None,
    reasoning_samples: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate checkpoint on perplexity, code syntax, and tag validity."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        config_dict = json.load(f)

    config = Z1Config(**config_dict)
    model = Z1ForCausalLM(config).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))

    results: Dict[str, Any] = {}

    # ─── Perplexity ─────────────────────────────────────────────────────────
    avg_loss, ppl = compute_perplexity(model, eval_sequences, device, config.max_seq_len)
    results["perplexity"] = ppl
    results["avg_loss"] = avg_loss
    print(f"[z1-eval] Perplexity: {ppl:.2f} | Loss: {avg_loss:.4f}")

    # ─── Python Syntax ──────────────────────────────────────────────────────
    if code_samples_python:
        py_results = [check_python_syntax(c) for c in code_samples_python]
        py_valid = sum(r["valid"] for r in py_results)
        results["python_syntax_pass_rate"] = py_valid / len(py_results)
        print(f"[z1-eval] Python Syntax: {py_valid}/{len(py_results)} valid ({results['python_syntax_pass_rate']:.1%})")

    # ─── JS Syntax ──────────────────────────────────────────────────────────
    if code_samples_js:
        js_results = [check_javascript_syntax_heuristic(c) for c in code_samples_js]
        js_valid = sum(r["valid"] for r in js_results)
        results["js_syntax_pass_rate"] = js_valid / len(js_results)
        print(f"[z1-eval] JS Syntax: {js_valid}/{len(js_results)} valid ({results['js_syntax_pass_rate']:.1%})")

    # ─── Reasoning Tags ─────────────────────────────────────────────────────
    if reasoning_samples:
        tag_results = [check_reasoning_tags(s) for s in reasoning_samples]
        tags_valid = sum(r["valid"] for r in tag_results)
        results["reasoning_tag_pass_rate"] = tags_valid / len(tag_results)
        print(f"[z1-eval] Reasoning Tags: {tags_valid}/{len(tag_results)} valid ({results['reasoning_tag_pass_rate']:.1%})")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="z1 Evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_jsonl", required=True, help="Evaluation JSONL file with input_ids")
    args = parser.parse_args()

    eval_seqs = []
    with open(args.eval_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            eval_seqs.append(obj["input_ids"])

    results = evaluate_checkpoint(args.checkpoint, eval_seqs)
    print(json.dumps(results, indent=2))
