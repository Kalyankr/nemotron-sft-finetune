"""Generate high-quality SFT training data using a strong LLM as teacher.

Sends each puzzle to a teacher LLM (Gemini, DeepSeek, OpenAI-compatible),
collects CoT reasoning + answer, validates against known answers, and
outputs chat-formatted JSONL for SFT training.

Usage:
    # Using GitHub Models (free with GitHub account - recommended to start)
    export GITHUB_TOKEN="your-github-personal-access-token"
    python scripts/generate_teacher_data.py \
        --provider github --model openai/gpt-4o \
        --data-dir data/ --output data/teacher_train.jsonl

    # Using Google Gemini (highest accuracy - 0.81 on this competition)
    export GEMINI_API_KEY="your-key"
    python scripts/generate_teacher_data.py \
        --provider gemini --model gemini-3.1-pro \
        --data-dir data/ --output data/teacher_train.jsonl

    # Using DeepSeek
    export DEEPSEEK_API_KEY="your-key"
    python scripts/generate_teacher_data.py \
        --provider deepseek --model deepseek-reasoner \
        --data-dir data/ --output data/teacher_train.jsonl

    # Using any OpenAI-compatible endpoint
    export OPENAI_API_KEY="your-key"
    python scripts/generate_teacher_data.py \
        --provider openai --model gpt-4o \
        --data-dir data/ --output data/teacher_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

from nemotron.config import DATA_DIR, SYSTEM_PROMPT
from nemotron.data_utils import save_jsonl
from nemotron.solvers import classify_puzzle

# Prompt template matching the competition evaluation format
TEACHER_PROMPT_SUFFIX = (
    "\n\nSolve this step by step. Be concise — show key reasoning steps only, "
    "not exhaustive enumeration. You MUST end your response with your final answer "
    "inside \\boxed{}. For example: \\boxed{42} or \\boxed{11010011}"
)

# Shorter suffix for models with built-in thinking (Gemini 2.5+)
TEACHER_PROMPT_SUFFIX_THINKING = (
    "\nSolve this puzzle. Put your final answer inside \\boxed{}. For example: \\boxed{42} or \\boxed{11010011}"
)


def _extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} answer from LLM output."""
    # Handle nested braces: \boxed{f(x) = {2}}
    matches = []
    for m in re.finditer(r"\\boxed\{", text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            matches.append(text[start : i - 1].strip())
    if matches:
        return matches[-1]

    # Fallback: look for **Answer: X** or similar patterns
    for pattern in [
        r"\*\*(?:Final )?[Aa]nswer\*\*[:\s]+(.+?)(?:\n|$)",
        r"(?:Final |The )?[Aa]nswer[:\s]+\*?\*?(.+?)\*?\*?(?:\n|$)",
        r"\$\\boxed\{([^}]+)\}\$",
    ]:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return None


def _check_answer(predicted: str, expected: str, tol: float = 1e-2) -> bool:
    """Check if predicted answer matches expected (exact or numeric tolerance)."""
    if predicted.strip().lower() == expected.strip().lower():
        return True
    try:
        pred_f = float(predicted)
        exp_f = float(expected)
        if exp_f == 0:
            return abs(pred_f) < tol
        return abs(pred_f - exp_f) / abs(exp_f) < tol
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Provider clients (lazy-loaded, no SDK dependencies at import time)
# ---------------------------------------------------------------------------


def _call_gemini(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """Call Google Gemini API via google.genai SDK."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY environment variable")

    config_kwargs: dict = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    # Gemini 2.5+ and 3.x models have built-in thinking — give it a separate budget
    # so thinking tokens don't consume the visible output budget
    _has_thinking = any(v in model for v in ("2.5", "3-", "3.1"))
    if _has_thinking:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=16384,
        )
        # Ensure visible output has enough room for the answer
        config_kwargs["max_output_tokens"] = max(max_tokens, 8192)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    # Debug: log finish reason
    if response.candidates:
        c = response.candidates[0]
        reason = getattr(c, "finish_reason", None)
        if reason and str(reason) != "STOP" and str(reason) != "FinishReason.STOP":
            print(f"  [gemini] finish_reason={reason}", file=sys.stderr)

    # Collect all text parts (thinking + response) for CoT training data
    parts = []
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "thought") and part.thought:
                # Thinking part — wrap it so the student model learns reasoning
                parts.append(f"<think>\n{part.text}\n</think>\n")
            elif part.text:
                parts.append(part.text)
    if parts:
        return "".join(parts)
    return response.text or ""


def _call_openai_compat(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> str:
    """Call any OpenAI-compatible API (OpenAI, DeepSeek, local)."""
    from openai import OpenAI

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(f"Set {api_key_env} environment variable")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


PROVIDER_MAP = {
    "gemini": {
        "call": _call_gemini,
        "kwargs": {},
    },
    "openai": {
        "call": _call_openai_compat,
        "kwargs": {"api_key_env": "OPENAI_API_KEY"},
    },
    "deepseek": {
        "call": _call_openai_compat,
        "kwargs": {
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
    },
    "github": {
        "call": _call_openai_compat,
        "kwargs": {
            "base_url": "https://models.github.ai/inference",
            "api_key_env": "GITHUB_TOKEN",
        },
    },
}


def generate_for_puzzle(
    prompt: str,
    provider: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str | None:
    """Send a single puzzle to the teacher LLM and return the raw response."""
    # Use shorter prompt for models with built-in thinking
    if provider == "gemini" and any(v in model for v in ("2.5", "3-", "3.1")):
        suffix = TEACHER_PROMPT_SUFFIX_THINKING
    else:
        suffix = TEACHER_PROMPT_SUFFIX
    full_prompt = prompt + suffix
    info = PROVIDER_MAP[provider]
    call_fn = info["call"]
    kwargs = dict(info["kwargs"])

    for backoff in (0, 30, 60, 120, 240):
        if backoff:
            print(f"  Rate limited, retrying in {backoff}s...", file=sys.stderr)
            time.sleep(backoff)
        try:
            return call_fn(full_prompt, model, temperature, max_tokens, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if (
                "429" in err_str
                or "resource_exhausted" in err_str
                or "too many requests" in err_str
                or "rate" in err_str
            ):
                continue  # retry with backoff
            print(f"  API error: {e}", file=sys.stderr)
            return None
    print("  Rate limit persists after retries, skipping puzzle", file=sys.stderr)
    return None


def _load_completed_prompts(output_path: Path) -> set[str]:
    """Load prompts already processed from an existing output file (for resume)."""
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # The user message is the puzzle prompt
                    for msg in entry.get("messages", []):
                        if msg.get("role") == "user":
                            completed.add(msg["content"])
                            break
                except json.JSONDecodeError:
                    continue
    return completed


def _append_jsonl(entry: dict, path: Path) -> None:
    """Append a single JSON entry to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher LLM training data.")
    parser.add_argument("--provider", type=str, required=True, choices=list(PROVIDER_MAP), help="LLM provider")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: gemini-2.5-pro for gemini, gpt-4o for openai/github, deepseek-reasoner for deepseek)",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing train.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "teacher_train.jsonl", help="Output JSONL path")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per response")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per puzzle on validation failure")
    parser.add_argument("--rate-limit", type=float, default=0.5, help="Seconds between API calls")
    parser.add_argument("--skip-validation", action="store_true", help="Keep all responses, even wrong ones")
    parser.add_argument("--limit", type=int, default=None, help="Max puzzles to process (for testing)")
    args = parser.parse_args()

    # Default model per provider
    if args.model is None:
        args.model = {
            "gemini": "gemini-3.1-pro-preview",
            "openai": "gpt-4o",
            "deepseek": "deepseek-reasoner",
            "github": "openai/gpt-4o",
        }[args.provider]
        print(f"Using default model for {args.provider}: {args.model}")

    # Load training data
    train_path = args.data_dir / "train.csv"
    if not train_path.exists():
        print(f"Error: {train_path} not found", file=sys.stderr)
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    train_df["puzzle_type"] = train_df["prompt"].apply(classify_puzzle)

    # Resume support: load already-completed prompts
    completed = _load_completed_prompts(args.output)
    if completed:
        print(f"Resuming: {len(completed)} puzzles already completed in {args.output}")

    remaining = train_df[~train_df["prompt"].isin(completed)]

    # Sort by difficulty: easiest puzzle types first
    difficulty_order = {
        "unit_conversion": 0,
        "gravitational": 1,
        "text_encryption": 2,
        "unknown": 3,
        "equation_transform": 4,
    }
    remaining = remaining.copy()
    remaining["_difficulty"] = remaining["puzzle_type"].map(difficulty_order).fillna(5)
    remaining = remaining.sort_values("_difficulty").drop(columns=["_difficulty"])

    if args.limit:
        remaining = remaining.head(args.limit)

    print(
        f"Processing {len(remaining)} puzzles ({len(train_df)} total, {len(completed)} done) with {args.provider}/{args.model}"
    )
    print("Puzzle type distribution:")
    print(remaining["puzzle_type"].value_counts().to_string())
    print()

    stats = {"total": len(completed), "valid": len(completed), "invalid": 0, "error": 0, "skipped": len(completed)}

    for idx, row in remaining.iterrows():
        stats["total"] += 1
        prompt = row["prompt"]
        expected = str(row.get("answer", ""))
        puzzle_type = row.get("puzzle_type", "unknown")

        best_response = None

        for attempt in range(1, args.max_retries + 1):
            if args.rate_limit > 0:
                time.sleep(args.rate_limit)

            response = generate_for_puzzle(
                prompt,
                args.provider,
                args.model,
                args.temperature,
                args.max_tokens,
            )
            if response is None:
                stats["error"] += 1
                continue

            predicted = _extract_boxed_answer(response)
            if predicted is None:
                # Show debug info on first failure
                if attempt == 1:
                    tail = response[-200:].replace("\n", " ")
                    print(f"  [{idx}] No \\boxed{{}} found (attempt {attempt}) | len={len(response)} | tail: ...{tail}")
                else:
                    print(f"  [{idx}] No \\boxed{{}} found (attempt {attempt})")
                continue

            if args.skip_validation or not expected or _check_answer(predicted, expected):
                best_response = response
                break
            else:
                print(f"  [{idx}] Wrong answer: got '{predicted}', expected '{expected}' (attempt {attempt})")

        if best_response:
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": best_response},
                ],
                "metadata": {
                    "puzzle_type": puzzle_type,
                    "expected_answer": expected,
                    "teacher_model": f"{args.provider}/{args.model}",
                },
            }
            # Save immediately after each puzzle
            _append_jsonl(example, args.output)
            stats["valid"] += 1
        else:
            stats["invalid"] += 1

        done = stats["total"] - stats["skipped"]
        if done % 10 == 0:
            print(
                f"Progress: {done}/{len(remaining)} | "
                f"valid={stats['valid']} invalid={stats['invalid']} error={stats['error']}"
            )

    # Save final stats
    print(f"\nDone! {stats['valid']} total examples in {args.output}")
    print(f"Stats: {json.dumps(stats, indent=2)}")

    stats_path = args.output.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
