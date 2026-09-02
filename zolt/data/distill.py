#!/usr/bin/env python3
"""
Teacher distillation pipeline for synthetic coding and reasoning pairs.
Generates structured training instances with reasoning traces (<think>...</think>)
and tool interactions for the zolt stack (Python, TypeScript, JavaScript, Next.js, NestJS).
"""

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Any
import urllib.error
import urllib.request

DEFAULT_FOCUS_TOPICS = [
    {"lang": "python", "topic": "algorithmic optimization and time-complexity reasoning"},
    {"lang": "python", "topic": "AST transformation and static analysis tool creation"},
    {
        "lang": "typescript",
        "topic": "type-safe state management in React 19 and Next.js App Router",
    },
    {
        "lang": "typescript",
        "topic": "NestJS microservice controller with DTO validation and guard logic",
    },
    {"lang": "javascript", "topic": "high-performance DOM manipulation and virtual scrolling"},
    {"lang": "vue", "topic": "Vue 3 Composition API store with reactive computed caching"},
]

PROMPT_TEMPLATE = """You are an expert software engineer generating high-quality training data for a coding and reasoning AI assistant.
Task topic: {topic}
Language: {lang}

Generate a realistic programming problem or architectural task and provide the complete solution.
Follow this format strictly:
1. Explain the reasoning process, edge cases, complexity trade-offs, and architecture inside <think> and </think> tags.
2. Provide clean, idiomatic, fully-working code with type annotations and docstrings inside <code> and </code> tags.
3. If tool interactions are relevant, demonstrate them using <tool_call>{{"tool": "name", "args": {{}}}}</tool_call> and <tool_response>{{"result": "..."}}</tool_response>.

Begin immediately with the user request followed by the assistant response.
"""


class TeacherClient:
    """Configurable HTTP client for teacher model distillation with retry and backoff."""

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        model: str = "gpt-4o",
        timeout: int = 60,
        max_retries: int = 5,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = (
            api_key or os.environ.get("TEACHER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        )
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(
        self, prompt: str, system_prompt: str = "You are a code synthesis assistant."
    ) -> str:
        """Call teacher endpoint with exponential backoff on rate-limits (HTTP 429)."""
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    choices = resp_data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                    return ""
            except urllib.error.HTTPError as e:
                status = e.code
                err_body = e.read().decode("utf-8", errors="replace")
                if status == 429 or status >= 500:
                    delay = (2**attempt) + random.uniform(0.5, 2.0)
                    print(
                        f"[zolt-distill] HTTP {status} (attempt {attempt}/{self.max_retries}). Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Teacher API error HTTP {status}: {err_body}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                delay = (2**attempt) + random.uniform(0.5, 1.5)
                print(
                    f"[zolt-distill] Network error ({e}) (attempt {attempt}/{self.max_retries}). Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        raise TimeoutError(f"Teacher API failed after {self.max_retries} attempts.")


def generate_synthetic_instance(
    client: TeacherClient | None,
    topic_spec: dict[str, str],
    mock_mode: bool = False,
) -> dict[str, Any]:
    """Generate a single formatted synthetic code & reasoning instance."""
    lang = topic_spec.get("lang", "python")
    topic = topic_spec.get("topic", "code logic")

    if mock_mode or client is None or not client.api_key:
        # Synthetic mock generator for testing and offline development
        reasoning = (
            f"Analyzing requirement for {topic} in {lang}.\n"
            f"1. Identify edge cases and constraints.\n"
            f"2. Structure solution using idiomatic patterns.\n"
            f"3. Verify complexity and type safety."
        )
        if lang == "python":
            code = (
                f"# Implementation for {topic}\n"
                f"def solve(data: list) -> dict:\n"
                f'    """Process data efficiently."""\n'
                f"    result = {{k: v for k, v in enumerate(data)}}\n"
                f"    return result\n"
            )
        else:
            code = (
                f"// Implementation for {topic}\n"
                f"export function solve<T>(items: T[]): Map<number, T> {{\n"
                f"  const map = new Map<number, T>();\n"
                f"  items.forEach((item, idx) => map.set(idx, item));\n"
                f"  return map;\n"
                f"}}\n"
            )

        content = (
            f"<|im_start|>user\nImplement a solution for {topic}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n{code}<|im_end|>"
        )
    else:
        prompt = PROMPT_TEMPLATE.format(topic=topic, lang=lang)
        teacher_output = client.complete(prompt)
        content = (
            f"<|im_start|>user\nProvide an optimal implementation and reasoning for {topic}.<|im_end|>\n"
            f"<|im_start|>assistant\n{teacher_output}<|im_end|>"
        )

    return {
        "content": content,
        "lang": lang,
        "license": "synthetic",
        "quality_score": 1.0,
        "topic": topic,
    }


def run_distillation(
    output_jsonl: str,
    n_samples: int = 100,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4o",
    api_key: str | None = None,
    mock_mode: bool = False,
) -> int:
    """Run distillation loop and save generated instances to JSONL."""
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = (
        TeacherClient(api_base=api_base, api_key=api_key, model=model) if not mock_mode else None
    )

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n_samples):
            topic = DEFAULT_FOCUS_TOPICS[i % len(DEFAULT_FOCUS_TOPICS)]
            record = generate_synthetic_instance(client, topic, mock_mode=mock_mode)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count % 10 == 0 or count == n_samples:
                print(
                    f"[zolt-distill] Generated {count}/{n_samples} synthetic instances -> {out_path}"
                )

    return count


def mix_datasets(
    corpus_jsonl_paths: list[str],
    distilled_jsonl_paths: list[str],
    output_path: str,
    distill_ratio: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Mix raw corpus data with teacher-distilled data at specified ratio.
    distill_ratio: fraction of final dataset composed of distilled data (e.g. 0.20 = 20% distilled, 80% corpus).
    """
    rng = random.Random(seed)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corpus_records = []
    for cp in corpus_jsonl_paths:
        if os.path.exists(cp):
            with open(cp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        corpus_records.append(line)

    distilled_records = []
    for dp in distilled_jsonl_paths:
        if os.path.exists(dp):
            with open(dp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        distilled_records.append(line)

    if not corpus_records and not distilled_records:
        raise ValueError("No records found in provided corpus or distilled paths.")

    if not distilled_records:
        selected_corpus = corpus_records
        selected_distilled = []
    elif not corpus_records:
        selected_corpus = []
        selected_distilled = distilled_records
    else:
        # Determine sample sizes to match distill_ratio
        # n_distilled / (n_corpus + n_distilled) = distill_ratio
        # n_distilled = n_corpus * distill_ratio / (1 - distill_ratio)
        n_corpus = len(corpus_records)
        desired_distilled = int(n_corpus * distill_ratio / max(1e-5, (1.0 - distill_ratio)))
        if desired_distilled <= len(distilled_records):
            selected_distilled = rng.sample(distilled_records, desired_distilled)
            selected_corpus = list(corpus_records)
        else:
            selected_distilled = list(distilled_records)
            desired_corpus = int(
                len(distilled_records) * (1.0 - distill_ratio) / max(1e-5, distill_ratio)
            )
            selected_corpus = rng.sample(corpus_records, min(len(corpus_records), desired_corpus))

    mixed = selected_corpus + selected_distilled
    rng.shuffle(mixed)

    with open(out_path, "w", encoding="utf-8") as f:
        for line in mixed:
            f.write(line + "\n")

    stats = {
        "total": len(mixed),
        "corpus_count": len(selected_corpus),
        "distilled_count": len(selected_distilled),
        "actual_distill_ratio": len(selected_distilled) / max(1, len(mixed)),
    }
    print(f"[zolt-distill] Dataset mixed -> {output_path} | stats: {stats}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="zolt Teacher Distillation")
    parser.add_argument(
        "--output", default="data/distilled/synthetic_reasoning.jsonl", help="Output JSONL path"
    )
    parser.add_argument(
        "--n_samples", type=int, default=50, help="Number of synthetic instances to generate"
    )
    parser.add_argument(
        "--api_base", default="https://api.openai.com/v1", help="Teacher API base URL"
    )
    parser.add_argument("--model", default="gpt-4o", help="Teacher model name")
    parser.add_argument("--api_key", default=None, help="API Key (or set TEACHER_API_KEY env var)")
    parser.add_argument(
        "--mock", action="store_true", help="Generate synthetic instances offline without API call"
    )
    parser.add_argument(
        "--mix_corpus", nargs="*", default=None, help="Corpus JSONL paths to mix with"
    )
    parser.add_argument("--mix_output", default=None, help="Output path for mixed dataset")
    parser.add_argument(
        "--distill_ratio",
        type=float,
        default=0.20,
        help="Fraction of distilled data in mixed dataset",
    )
    args = parser.parse_args()

    run_distillation(
        output_jsonl=args.output,
        n_samples=args.n_samples,
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
        mock_mode=args.mock,
    )

    if args.mix_corpus and args.mix_output:
        mix_datasets(
            corpus_jsonl_paths=args.mix_corpus,
            distilled_jsonl_paths=[args.output],
            output_path=args.mix_output,
            distill_ratio=args.distill_ratio,
        )
