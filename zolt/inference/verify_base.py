"""
Shared base for self-verification and self-correction retry loops.

Each language verifier provides a checker function with the signature:
    checker(code: str) -> VerifyResult

The retry orchestration, correction-turn construction, and return shape
are defined here once and reused by every language-specific verifier.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class VerifyResult:
    """
    Uniform result shape returned by every checker and the retry loop.

    Fields:
        valid     -- True only if a real tool completed successfully.
        verified  -- True only if a real tool run (not heuristic) passed.
        heuristic -- True when the check was performed via a fallback heuristic.
        error     -- Error string when valid is False; None otherwise.
        checker   -- Name of the checker that produced this result.
    """
    valid: bool
    verified: bool
    heuristic: bool = False
    error: Optional[str] = None
    checker: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for callers that expect a dict return."""
        return {
            "valid": self.valid,
            "verified": self.verified,
            "heuristic": self.heuristic,
            "error": self.error,
            "checker": self.checker,
        }


def extract_code_block(text: str) -> str:
    """
    Extract code from <code>...</code> tags, markdown fences, or raw text.
    Priority: zolt <code> tags > markdown fence > raw text.
    """
    code_match = re.search(r"<code>\n?(.*?)\n?</code>", text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    fence_match = re.search(
        r"```(?:ts|typescript|javascript|js|python|py)?\n(.*?)```",
        text,
        re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()

    return text.strip()


def _build_correction_prompt(
    original_prompt: str,
    last_error: str,
    language: str,
) -> str:
    """Build a ChatML correction turn feeding the checker error back to the model."""
    return (
        f"The previous {language} code produced the following error:\n"
        f"```\n{last_error}\n```\n"
        f"Original task: {original_prompt}\n"
        f"Please fix all errors and provide the complete, working {language} solution inside <code> tags."
    )


def self_correcting_generate(
    generator: Any,
    prompt: str,
    checker: Callable[[str], VerifyResult],
    language: str,
    max_retries: int = 2,
    temperature: float = 0.5,
    top_p: float = 0.9,
    system_prompt: str = "",
) -> Dict[str, Any]:
    """
    Generic self-correcting generation loop.

    Calls generator.generate_stream, extracts code, runs checker.
    On failure, builds a correction turn and retries up to max_retries times.
    Returns the standard result dict: {code, verified, heuristic, attempts, error, checker}.
    """
    current_prompt = prompt
    last_error: Optional[str] = None
    last_code = ""
    last_checker = "unknown"

    for attempt in range(1, max_retries + 1):
        formatted_prompt = generator.format_agent_prompt(
            system_prompt=system_prompt,
            user_prompt=current_prompt,
            include_think_tag=True,
        )

        response_chunks = list(generator.generate_stream(
            formatted_prompt,
            temperature=temperature,
            top_p=top_p,
        ))
        response_text = "".join(response_chunks)
        extracted_code = extract_code_block(response_text)
        last_code = extracted_code

        result = checker(extracted_code)
        last_checker = result.checker

        if result.valid:
            return {
                "code": extracted_code,
                "verified": result.verified,
                "heuristic": result.heuristic,
                "attempts": attempt,
                "error": None,
                "checker": result.checker,
                "raw_response": response_text,
            }

        last_error = result.error or "Unknown error."
        current_prompt = _build_correction_prompt(prompt, last_error, language)

    return {
        "code": last_code,
        "verified": False,
        "heuristic": False,
        "attempts": max_retries,
        "error": last_error,
        "checker": last_checker,
    }
