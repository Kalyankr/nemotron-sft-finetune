"""Data loading, classification, and SFT training data generation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from nemotron.config import DIFFICULTY_WEIGHTS, SYSTEM_PROMPT
from nemotron.solvers import classify_puzzle, solve_puzzle


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and classify train/test data."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    train["puzzle_type"] = train["prompt"].apply(classify_puzzle)
    return train, test


def create_sft_example(prompt: str, expected_answer: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Create a single chat-formatted SFT training example."""
    answer, cot = solve_puzzle(prompt, expected_answer)
    if answer is None:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{cot}\n\n\\boxed{{{answer}}}"},
        ]
    }


def build_sft_dataset(
    train_df: pd.DataFrame,
    oversample: bool = True,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build the full SFT dataset from training data, with optional oversampling."""
    examples: list[dict[str, Any]] = []

    for _, row in train_df.iterrows():
        ex = create_sft_example(row["prompt"], row.get("answer"))
        if ex is None:
            continue
        weight = DIFFICULTY_WEIGHTS.get(row.get("puzzle_type", "unknown"), 1) if oversample else 1
        for _ in range(weight):
            examples.append(ex)

    random.seed(seed)
    random.shuffle(examples)
    return examples


def save_jsonl(data: list[dict], path: str | Path) -> None:
    """Write a list of dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f]
