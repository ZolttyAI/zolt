"""
TypeScript self-verification and self-correction loop at inference.
Executes verification on generated TypeScript code, feeding compiler errors back to the model for correction.
"""
import os
import re
import shutil
import tempfile
import subprocess
from typing import Dict, Any, Optional, Tuple

from z1.eval import check_javascript_syntax_heuristic


def extract_code_block(text: str, default_lang: str = "typescript") -> str:
    """Extract code block from <code> tags, markdown code fences, or fallback to raw text."""
    # 1. Check z1 <code>...</code> tags
    code_match = re.search(r"<code>\n?(.*?)\n?</code>", text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # 2. Check markdown ```typescript or ```ts or ```
    fence_match = re.search(r"```(?:ts|typescript|javascript|js)?\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    return text.strip()


def run_tsc_check(ts_code: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Run tsc --noEmit against a temporary TypeScript file if tsc is available.
    Falls back to heuristic syntax checking from z1.eval if tsc is not found.
    """
    tsc_path = shutil.which("tsc")
    if not tsc_path:
        # Fallback to z1.eval syntax validator
        return check_javascript_syntax_heuristic(ts_code)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "verification.ts")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(ts_code)

        try:
            result = subprocess.run(
                [tsc_path, "--noEmit", "--target", "ES2022", "--skipLibCheck", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return {"valid": True, "error": None}
            else:
                err_msg = (result.stdout + "\n" + result.stderr).strip()
                return {"valid": False, "error": err_msg}
        except subprocess.TimeoutExpired:
            return {"valid": False, "error": f"TypeScript compilation timed out after {timeout}s"}
        except Exception as e:
            return {"valid": False, "error": str(e)}


def verify_typescript_code(ts_code: str) -> Dict[str, Any]:
    """Verify TypeScript code with syntax heuristic and compiler validation."""
    if not ts_code or not ts_code.strip():
        return {"valid": False, "error": "Empty TypeScript code snippet."}

    # Fast syntax check from z1.eval
    syntax_result = check_javascript_syntax_heuristic(ts_code)
    if not syntax_result["valid"]:
        return syntax_result

    # Compiler check
    return run_tsc_check(ts_code)


def self_correcting_generate_ts(
    generator: Any,
    prompt: str,
    max_retries: int = 2,
    temperature: float = 0.5,
    top_p: float = 0.9,
    system_prompt: str = "You are an expert TypeScript engineer. Write valid, type-safe TypeScript code.",
) -> Dict[str, Any]:
    """
    Generate TypeScript code with self-verification and automatic compiler error correction turns.
    Returns dictionary with final code, verification status, attempt count, and last error.
    """
    current_prompt = prompt
    last_error: Optional[str] = None
    last_code = ""

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

        verification = verify_typescript_code(extracted_code)
        if verification["valid"]:
            return {
                "code": extracted_code,
                "verified": True,
                "attempts": attempt,
                "error": None,
                "raw_response": response_text,
            }

        last_error = verification.get("error", "Unknown compilation error.")
        # Construct correction prompt turn for next attempt
        current_prompt = (
            f"The previous TypeScript code produced the following compilation error:\n"
            f"```\n{last_error}\n```\n"
            f"Original task: {prompt}\n"
            f"Please fix all errors and provide the complete, working TypeScript solution inside <code> tags."
        )

    return {
        "code": last_code,
        "verified": False,
        "attempts": max_retries,
        "error": last_error,
    }
