"""
JavaScript self-verification and self-correction loop.

Primary checker: node --check <file>.
Optional richer errors: eslint --no-eslintrc --rule 'no-undef: error' when eslint is present.
Fallback: zolt.eval bracket/string-balance heuristic (verified=False, heuristic=True).
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


_SYSTEM_PROMPT = "You are an expert JavaScript engineer. Write valid, modern JavaScript code."


def _check_javascript(js_code: str, timeout: int = 10) -> VerifyResult:
    """
    Internal checker returning a VerifyResult.
    verified=True only when node --check or eslint actually ran and passed.
    """
    if not js_code or not js_code.strip():
        return VerifyResult(valid=False, verified=False, error="Empty JavaScript code snippet.", checker="none")

    # Fast heuristic pre-check
    heuristic = check_javascript_syntax_heuristic(js_code)
    if not heuristic["valid"]:
        return VerifyResult(
            valid=False,
            verified=False,
            heuristic=True,
            error=heuristic["error"],
            checker="heuristic",
        )

    node_path = shutil.which("node")
    if not node_path:
        # No Node available: heuristic passed but not tool-verified
        return VerifyResult(
            valid=True,
            verified=False,
            heuristic=True,
            error=None,
            checker="heuristic",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "verification.js")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(js_code)

        # Prefer eslint when available for richer error messages
        eslint_path = shutil.which("eslint")
        if eslint_path:
            result = _run_eslint(eslint_path, file_path, timeout)
            if result is not None:
                return result

        # Primary: node --check
        return _run_node_check(node_path, file_path, timeout)


def _run_node_check(node_path: str, file_path: str, timeout: int) -> VerifyResult:
    """Run node --check on file_path and return a VerifyResult."""
    try:
        proc = subprocess.run(
            [node_path, "--check", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return VerifyResult(valid=True, verified=True, heuristic=False, error=None, checker="node")
        err_msg = (proc.stdout + "\n" + proc.stderr).strip()
        return VerifyResult(valid=False, verified=False, error=err_msg, checker="node")
    except subprocess.TimeoutExpired:
        return VerifyResult(valid=False, verified=False, error=f"node --check timed out after {timeout}s", checker="node")
    except Exception as e:
        return VerifyResult(valid=False, verified=False, error=str(e), checker="node")


def _run_eslint(eslint_path: str, file_path: str, timeout: int) -> Optional[VerifyResult]:
    """
    Run eslint with minimal config for syntax checking.
    Returns None if eslint errors out unexpectedly (caller falls back to node --check).
    """
    try:
        proc = subprocess.run(
            [eslint_path, "--no-eslintrc", "--env", "es2022", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return VerifyResult(valid=True, verified=True, heuristic=False, error=None, checker="eslint")
        err_msg = (proc.stdout + "\n" + proc.stderr).strip()
        return VerifyResult(valid=False, verified=False, error=err_msg, checker="eslint")
    except subprocess.TimeoutExpired:
        return VerifyResult(valid=False, verified=False, error=f"eslint timed out after {timeout}s", checker="eslint")
    except Exception:
        # Unexpected eslint failure: fall back to node --check
        return None


def verify_javascript_code(js_code: str) -> Dict[str, Any]:
    """
    Verify JavaScript code. Returns {valid, verified, heuristic, error, checker}.
    verified=True only when node or eslint ran and passed.
    """
    return _check_javascript(js_code).to_dict()


def self_correcting_generate_js(
    generator: Any,
    prompt: str,
    max_retries: int = 2,
    temperature: float = 0.5,
    top_p: float = 0.9,
    system_prompt: str = _SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """
    Generate JavaScript with self-verification and correction loop.
    Returns {code, verified, heuristic, attempts, error, checker}.
    verified=True only when node or eslint ran and passed.
    """
    return self_correcting_generate(
        generator=generator,
        prompt=prompt,
        checker=_check_javascript,
        language="JavaScript",
        max_retries=max_retries,
        temperature=temperature,
        top_p=top_p,
        system_prompt=system_prompt,
    )
