"""
TypeScript self-verification and self-correction loop.

Primary checker: tsc --noEmit.
Fallback: zolt.eval bracket/string-balance heuristic (marked as heuristic, verified=False).
"""
import os
import shutil
import tempfile
import subprocess
from typing import Any, Dict, Optional

from zolt.eval import check_javascript_syntax_heuristic
from zolt.inference.verify_base import (
    VerifyResult,
    extract_code_block,
    self_correcting_generate,
)


_SYSTEM_PROMPT = "You are an expert TypeScript engineer. Write valid, type-safe TypeScript code."


def run_tsc_check(ts_code: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Run tsc --noEmit against a temporary file.
    Falls back to the zolt.eval heuristic when tsc is not installed.
    Returns a plain dict (valid, error) for backward compatibility.
    """
    result = _check_typescript(ts_code, timeout=timeout)
    return {"valid": result.valid, "error": result.error}


def _check_typescript(ts_code: str, timeout: int = 10) -> VerifyResult:
    """
    Internal checker returning a VerifyResult.
    verified=True only when tsc actually ran and passed.
    """
    if not ts_code or not ts_code.strip():
        return VerifyResult(valid=False, verified=False, error="Empty TypeScript code snippet.", checker="none")

    # Fast heuristic pre-check (bracket balance, string balance)
    heuristic = check_javascript_syntax_heuristic(ts_code)
    if not heuristic["valid"]:
        return VerifyResult(
            valid=False,
            verified=False,
            heuristic=True,
            error=heuristic["error"],
            checker="heuristic",
        )

    tsc_path = shutil.which("tsc")
    if not tsc_path:
        # Heuristic passed but no real compiler: mark explicitly as heuristic
        return VerifyResult(
            valid=True,
            verified=False,
            heuristic=True,
            error=None,
            checker="heuristic",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "verification.ts")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(ts_code)

        try:
            proc = subprocess.run(
                [tsc_path, "--noEmit", "--target", "ES2022", "--skipLibCheck", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            if proc.returncode == 0:
                return VerifyResult(valid=True, verified=True, heuristic=False, error=None, checker="tsc")
            err_msg = (proc.stdout + "\n" + proc.stderr).strip()
            return VerifyResult(valid=False, verified=False, error=err_msg, checker="tsc")
        except subprocess.TimeoutExpired:
            return VerifyResult(
                valid=False,
                verified=False,
                error=f"tsc timed out after {timeout}s",
                checker="tsc",
            )
        except Exception as e:
            return VerifyResult(valid=False, verified=False, error=str(e), checker="tsc")


def verify_typescript_code(ts_code: str) -> Dict[str, Any]:
    """
    Verify TypeScript code. Returns {valid, verified, heuristic, error, checker}.
    verified=True only when tsc ran and passed.
    """
    result = _check_typescript(ts_code)
    return result.to_dict()


def self_correcting_generate_ts(
    generator: Any,
    prompt: str,
    max_retries: int = 2,
    temperature: float = 0.5,
    top_p: float = 0.9,
    system_prompt: str = _SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """
    Generate TypeScript with self-verification and correction loop.
    Returns {code, verified, heuristic, attempts, error, checker}.
    verified=True only when tsc ran and passed.
    """
    return self_correcting_generate(
        generator=generator,
        prompt=prompt,
        checker=_check_typescript,
        language="TypeScript",
        max_retries=max_retries,
        temperature=temperature,
        top_p=top_p,
        system_prompt=system_prompt,
    )
