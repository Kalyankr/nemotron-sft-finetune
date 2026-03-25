"""Reward functions for GRPO reinforcement learning.

Provides answer-correctness rewards for the Nemotron reasoning challenge,
matching the competition's exact-match + numeric tolerance scoring.
"""

from __future__ import annotations

import re


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} answer from model output."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None


def check_answer(predicted: str, expected: str, tol: float = 1e-2) -> bool:
    """Check if predicted matches expected (exact string or numeric tolerance)."""
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


def correctness_reward(completions: list[str], expected_answers: list[str]) -> list[float]:
    """Compute binary correctness rewards for a batch of completions.

    Returns 1.0 if the extracted \\boxed{} answer matches expected, else 0.0.
    This is the primary reward signal for GRPO.
    """
    rewards = []
    for completion, expected in zip(completions, expected_answers):
        predicted = extract_boxed_answer(completion)
        if predicted is not None and check_answer(predicted, expected):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions: list[str]) -> list[float]:
    """Reward for having a properly formatted \\boxed{} answer.

    Returns 0.5 if \\boxed{} is present (encouraging the format), else 0.0.
    """
    rewards = []
    for completion in completions:
        if extract_boxed_answer(completion) is not None:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def length_penalty(completions: list[str], max_tokens: int = 7680) -> list[float]:
    """Small penalty for excessively long responses that might get truncated.

    Returns 0.0 for normal length, negative for very long responses.
    """
    rewards = []
    for completion in completions:
        token_estimate = len(completion.split())
        if token_estimate > max_tokens * 0.9:
            rewards.append(-0.5)
        else:
            rewards.append(0.0)
    return rewards


def combined_reward(
    completions: list[str],
    expected_answers: list[str],
    correctness_weight: float = 1.0,
    format_weight: float = 0.1,
    length_weight: float = 0.1,
) -> list[float]:
    """Combined reward: correctness + format bonus + length penalty."""
    correct = correctness_reward(completions, expected_answers)
    fmt = format_reward(completions)
    length = length_penalty(completions)
    return [correctness_weight * c + format_weight * f + length_weight * ln for c, f, ln in zip(correct, fmt, length)]
