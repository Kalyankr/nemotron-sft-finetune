"""Prepare SFT training data from competition CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from nemotron.config import DATA_DIR
from nemotron.data_utils import build_sft_dataset, load_data, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT training JSONL from competition data.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory with train.csv and test.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "sft_train.jsonl", help="Output JSONL path")
    parser.add_argument("--no-oversample", action="store_true", help="Disable hard-type oversampling")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir} ...")
    train_df, _ = load_data(args.data_dir)

    print(f"Train: {len(train_df)} rows")
    print(f"Puzzle type distribution:\n{train_df['puzzle_type'].value_counts().to_string()}")

    print(f"\nBuilding SFT dataset (oversample={not args.no_oversample}) ...")
    dataset = build_sft_dataset(train_df, oversample=not args.no_oversample, seed=args.seed)

    print(f"Total training examples: {len(dataset)}")
    save_jsonl(dataset, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
