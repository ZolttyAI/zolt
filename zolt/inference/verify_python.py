"""
Python self-verification and self-correction loop.

Primary checker: ast.parse (stdlib, always available) / python -m py_compile.
Optional stricter pass: mypy when installed.
Python is always available in this environment, so there is no "no checker" case.
The result distinguishes syntax-only from mypy-augmented checks via the `checker` field.
"""

import ast
import os
import shutil
import subprocess
import tempfile
from typing import Any

from zolt.inference.verify_base import (
    VerifyResult,
    self_correcting_generate,
)

_SYSTEM_PROMPT = "You are an expert Python engineer. Write valid, idiomatic Python 3 code."


def _check_python(py_code: str, timeout: int = 10) -> VerifyResult:
    """
    Internal checker returning a VerifyResult.
    Always runs ast.parse (syntax only). If mypy is present, runs it on top.
    verified=True in both cases because Python is always available.
    The checker field distinguishes 'ast' from 'mypy'.
    """
    if not py_code or not py_code.strip():
        return VerifyResult(
            valid=False, verified=False, error="Empty Python code snippet.", checker="none"
        )

    # Syntax check via ast.parse (no subprocess, no failure possible due to missing tool)
    try:
        ast.parse(py_code)
    except SyntaxError as e:
        return VerifyResult(
            valid=False,
            verified=True,  # ast.parse is real, the check itself ran
            heuristic=False,
            error=f"SyntaxError: {e}",
            checker="ast",
        )

    # Syntax passed. Optionally run mypy for stricter checks.
    mypy_path = shutil.which("mypy")
    if mypy_path:
        mypy_result = _run_mypy(mypy_path, py_code, timeout)
        if mypy_result is not None:
            return mypy_result

    # Syntax-only verified pass
    return VerifyResult(
        valid=True,
        verified=True,  # ast.parse ran successfully
        heuristic=False,
        error=None,
        checker="ast",
    )


def _run_mypy(mypy_path: str, py_code: str, timeout: int) -> VerifyResult | None:
    """
    Run mypy on a temporary file. Returns None on unexpected subprocess errors.
    Returns VerifyResult with checker='mypy' on success or failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "verification.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(py_code)

        try:
            proc = subprocess.run(
                [mypy_path, "--ignore-missing-imports", "--no-error-summary", file_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode == 0:
                return VerifyResult(
                    valid=True, verified=True, heuristic=False, error=None, checker="mypy"
                )
            err_msg = (proc.stdout + "\n" + proc.stderr).strip()
            return VerifyResult(
                valid=False, verified=True, heuristic=False, error=err_msg, checker="mypy"
            )
        except subprocess.TimeoutExpired:
            return VerifyResult(
                valid=False,
                verified=True,
                heuristic=False,
                error=f"mypy timed out after {timeout}s",
                checker="mypy",
            )
        except Exception:
            # Unexpected failure: skip mypy, caller returns ast result
            return None


def verify_python_code(py_code: str) -> dict[str, Any]:
    """
    Verify Python code. Returns {valid, verified, heuristic, error, checker}.
    verified=True always (Python always available); checker distinguishes 'ast' vs 'mypy'.
    """
    return _check_python(py_code).to_dict()


def self_correcting_generate_python(
    generator: Any,
    prompt: str,
    max_retries: int = 2,
    temperature: float = 0.5,
    top_p: float = 0.9,
    system_prompt: str = _SYSTEM_PROMPT,
) -> dict[str, Any]:
    """
    Generate Python with self-verification and correction loop.
    Returns {code, verified, heuristic, attempts, error, checker}.
    verified=True when ast.parse (always) or mypy ran and passed.
    """
    return self_correcting_generate(
        generator=generator,
        prompt=prompt,
        checker=_check_python,
        language="Python",
        max_retries=max_retries,
        temperature=temperature,
        top_p=top_p,
        system_prompt=system_prompt,
    )
