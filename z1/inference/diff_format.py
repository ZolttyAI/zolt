"""
Native diff format parser and patch application engine using <search>, <replace>, <diff_end> tokens.
Provides deterministic, fail-closed code modification without whole-file rewrites.
"""
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DiffEdit:
    """Structured representation of a single code search/replace edit block."""
    search: str
    replace: str
    path: Optional[str] = None


def parse_diff_blocks(diff_text: str) -> List[DiffEdit]:
    """
    Parse model output containing <search> ... <replace> ... <diff_end> tags.
    Extracts optional file path preceding the block and exact search/replace contents.
    """
    edits: List[DiffEdit] = []
    # Pattern to capture optional path header followed by <search>...<replace>...<diff_end>
    pattern = re.compile(
        r'(?:(?:(?:File|Path|Target|===)?\s*[:\[]?([^\n<>\[\]]+\.[a-zA-Z0-9_.-]+)[\]:]?\s*\n))?'
        r'<search>\n?(.*?)\n?<replace>\n?(.*?)\n?<diff_end>',
        re.DOTALL
    )

    for match in pattern.finditer(diff_text):
        path = match.group(1).strip() if match.group(1) else None
        search_content = match.group(2)
        replace_content = match.group(3)

        # Normalize line endings
        search_content = search_content.replace("\r\n", "\n")
        replace_content = replace_content.replace("\r\n", "\n")

        edits.append(DiffEdit(search=search_content, replace=replace_content, path=path))

    return edits


def apply_diff_edit(source: str, search: str, replace: str) -> str:
    """
    Apply a single search/replace edit to a source string.
    Raises ValueError if search block is not found or occurs multiple times.
    """
    source_norm = source.replace("\r\n", "\n")
    search_norm = search.replace("\r\n", "\n")
    replace_norm = replace.replace("\r\n", "\n")

    matches = source_norm.count(search_norm)
    if matches == 0:
        raise ValueError(
            f"Search block not found in source text.\nSearch block:\n{search_norm}"
        )
    if matches > 1:
        raise ValueError(
            f"Search block matched {matches} locations in source text. Must match uniquely to avoid ambiguous edits.\nSearch block:\n{search_norm}"
        )

    return source_norm.replace(search_norm, replace_norm, 1)


def apply_diff_block(source: str, diff_text: str) -> str:
    """
    Parse and apply all diff edits in diff_text sequentially to source code.
    Raises ValueError if any edit cannot be applied uniquely.
    """
    edits = parse_diff_blocks(diff_text)
    if not edits:
        raise ValueError("No valid <search>...<replace>...<diff_end> blocks found in diff text.")

    current = source
    for edit in edits:
        current = apply_diff_edit(current, edit.search, edit.replace)

    return current


def format_diff_block(search: str, replace: str, path: Optional[str] = None) -> str:
    """Format an edit block into standard z1 diff special tokens."""
    header = f"[{path}]\n" if path else ""
    return f"{header}<search>\n{search}\n<replace>\n{replace}\n<diff_end>"
