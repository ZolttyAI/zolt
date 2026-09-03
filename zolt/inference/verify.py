"""
Language dispatcher for self-verification.
Routes generated code to the correct language-specific verifier based on a language tag.
"""

from typing import Any

from zolt.inference.verify_js import self_correcting_generate_js, verify_javascript_code
from zolt.inference.verify_python import self_correcting_generate_python, verify_python_code
from zolt.inference.verify_ts import self_correcting_generate_ts, verify_typescript_code

# Maps language tags to verifier functions
_VERIFY_DISPATCH: dict[str, Any] = {
    "typescript": verify_typescript_code,
    "ts": verify_typescript_code,
    "javascript": verify_javascript_code,
    "js": verify_javascript_code,
    "python": verify_python_code,
    "py": verify_python_code,
}

_GENERATE_DISPATCH: dict[str, Any] = {
    "typescript": self_correcting_generate_ts,
    "ts": self_correcting_generate_ts,
    "javascript": self_correcting_generate_js,
    "js": self_correcting_generate_js,
    "python": self_correcting_generate_python,
    "py": self_correcting_generate_python,
}

SUPPORTED_LANGUAGES = frozenset(_VERIFY_DISPATCH.keys())


def verify_code(code: str, language: str) -> dict[str, Any]:
    """
    Verify code in the specified language.
    Raises ValueError for unsupported language tags.
    """
    lang = language.lower().strip()
    verifier = _VERIFY_DISPATCH.get(lang)
    if verifier is None:
        raise ValueError(
            f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}"
        )
    return verifier(code)


def self_correcting_generate(
    generator: Any,
    prompt: str,
    language: str,
    max_retries: int = 2,
    temperature: float = 0.5,
    top_p: float = 0.9,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """
    Dispatch self-correcting generation to the correct language verifier.
    Raises ValueError for unsupported language tags.
    """
    lang = language.lower().strip()
    gen_fn = _GENERATE_DISPATCH.get(lang)
    if gen_fn is None:
        raise ValueError(
            f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}"
        )
    kwargs: dict[str, Any] = {
        "generator": generator,
        "prompt": prompt,
        "max_retries": max_retries,
        "temperature": temperature,
        "top_p": top_p,
    }
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return gen_fn(**kwargs)
