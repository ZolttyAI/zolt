"""
Curriculum learning utilities for progressive code difficulty staging.
Provides lightweight complexity scoring and ordering without large-model overhead.
"""

from collections.abc import Callable
import math
from typing import Any


def estimate_code_complexity(text: str) -> float:
    """
    Compute a lightweight complexity score for code text (0.0 to 100.0).
    Factors: line count, maximum indentation nesting depth, and branching keyword density.
    """
    lines = text.split("\n")
    if not lines:
        return 0.0

    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return 0.0

    # 1. Length factor (scaled logarithmic)
    n_lines = len(non_empty)
    length_score = min(40.0, math.log2(max(1, n_lines)) * 5.0)

    # 2. Indentation nesting depth factor
    max_indent = 0
    total_indent = 0
    for line in non_empty:
        leading_spaces = len(line) - len(line.lstrip(" "))
        indent_level = leading_spaces // 2  # normalize 2/4 spaces
        max_indent = max(max_indent, indent_level)
        total_indent += indent_level

    avg_indent = total_indent / len(non_empty)
    indent_score = min(30.0, (max_indent * 2.5) + (avg_indent * 3.0))

    # 3. Control flow and cyclomatic branching density
    branch_keywords = (
        "if ",
        "elif ",
        "else if ",
        "for ",
        "while ",
        "catch ",
        "except ",
        "switch ",
        "case ",
        "&&",
        "||",
        "lambda ",
        "async ",
        "await ",
        "class ",
    )
    branch_count = sum(any(kw in line for kw in branch_keywords) for line in non_empty)
    branch_density = branch_count / len(non_empty)
    branch_score = min(30.0, branch_density * 80.0)

    total_complexity = length_score + indent_score + branch_score
    return round(total_complexity, 2)


def estimate_token_sequence_complexity(token_ids: list[int]) -> float:
    """
    Compute a complexity proxy from token sequence IDs.
    Uses length and unique token diversity (type-token ratio).
    """
    if not token_ids:
        return 0.0

    length = len(token_ids)
    length_score = min(50.0, math.log2(max(1, length)) * 4.0)

    unique_tokens = len(set(token_ids))
    diversity_ratio = unique_tokens / length
    diversity_score = min(50.0, diversity_ratio * 50.0)

    return round(length_score + diversity_score, 2)


def sort_by_curriculum(
    items: list[Any],
    complexity_fn: Callable[[Any], float] | None = None,
    reverse: bool = False,
) -> list[Any]:
    """Sort a collection of items in order of increasing complexity."""
    if complexity_fn is None:

        def default_complexity(item):
            if isinstance(item, str):
                return estimate_code_complexity(item)
            elif isinstance(item, list):
                return estimate_token_sequence_complexity(item)
            elif isinstance(item, dict) and "content" in item:
                return estimate_code_complexity(item["content"])
            return 0.0

        complexity_fn = default_complexity

    return sorted(items, key=complexity_fn, reverse=reverse)
