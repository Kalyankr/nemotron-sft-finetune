"""GRPO (Group Relative Policy Optimization) training for Nemotron.

Runs GRPO reinforcement learning on top of an SFT-trained LoRA adapter,
using answer correctness as the reward signal.

Usage:
    python scripts/train_grpo.py \
        --model-path /path/to/nemotron-3-nano-30b \
        --adapter-path outputs/sft_output/lora_adapter \
        --data data/train.csv \
        --config configs/grpo.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from nemotron.config import (
    DATA_DIR,
    KAGGLE_MODEL_PATH,
    OUTPUT_DIR,
    SYSTEM_PROMPT,
)
from nemotron.rewards import extract_boxed_answer, check_answer
from nemotron.solvers import classify_puzzle


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_GRPO_ARGS = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-6,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "bf16": True,
    "gradient_checkpointing": True,
    "max_completion_length": 4096,
    "max_prompt_length": 512,
    "num_generations": 4,
    "temperature": 0.7,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def build_grpo_dataset(train_df: pd.DataFrame) -> Dataset:
    """Build a prompt-only dataset for GRPO with answer metadata."""
    records = []
    for _, row in train_df.iterrows():
        prompt_text = row["prompt"]
        answer = str(row.get("answer", ""))
        if not answer:
            continue
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text + "\nPlease put your final answer inside \\boxed{}."},
                ],
                "expected_answer": answer,
            }
        )
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Reward function for GRPOTrainer
# ---------------------------------------------------------------------------


def make_reward_fn(dataset: Dataset):
    """Create a reward function that works with GRPOTrainer.

    GRPOTrainer passes completions and prompts; we look up expected answers
    from the dataset to compute rewards.
    """
    # Build a lookup from prompt text to expected answer
    answer_lookup = {}
    for row in dataset:
        # Extract the user message content as key
        user_content = row["prompt"][-1]["content"]
        answer_lookup[user_content] = row["expected_answer"]

    def reward_fn(completions, **kwargs):
        """Reward function for GRPO: 1.0 for correct, 0.1 for formatted, 0.0 otherwise."""
        rewards = []
        prompts = kwargs.get("prompts", [None] * len(completions))

        for completion, prompt in zip(completions, prompts):
            # Extract completion text
            if isinstance(completion, list):
                text = completion[-1]["content"] if completion else ""
            else:
                text = str(completion)

            # Find expected answer
            expected = None
            if prompt is not None:
                if isinstance(prompt, list):
                    user_content = prompt[-1]["content"]
                else:
                    user_content = str(prompt)
                expected = answer_lookup.get(user_content)

            predicted = extract_boxed_answer(text)

            if predicted is not None and expected is not None and check_answer(predicted, expected):
                rewards.append(1.0)
            elif predicted is not None:
                # Correct format but wrong answer
                rewards.append(0.1)
            else:
                rewards.append(0.0)

        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Tracking helpers (reused from train.py pattern)
# ---------------------------------------------------------------------------


def _setup_tracker(tracker: str, cfg: dict, run_name: str, all_params: dict) -> None:
    if tracker == "wandb":
        import wandb

        wandb_cfg = cfg.get("wandb", {})
        wandb.init(
            project=wandb_cfg.get("project", "nemotron-reasoning"),
            entity=wandb_cfg.get("entity"),
            name=run_name,
            tags=[f"{k}:{v}" for k, v in cfg.get("tags", {}).items()],
            config=all_params,
        )
    elif tracker == "mlflow":
        import mlflow

        mlflow_cfg = cfg.get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "./mlruns"))
        mlflow.set_experiment(cfg.get("experiment_name", "nemotron-grpo"))
        mlflow.start_run(run_name=run_name, tags=cfg.get("tags", {}))
        flat = {}
        for k, v in all_params.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}.{k2}"] = str(v2)
            else:
                flat[k] = str(v)
        mlflow.log_params(flat)


def _finish_tracker(tracker: str, adapter_dir: Path) -> None:
    if tracker == "wandb":
        import wandb

        artifact = wandb.Artifact("grpo-lora-adapter", type="model")
        artifact.add_dir(str(adapter_dir))
        wandb.log_artifact(artifact)
        wandb.finish()
    elif tracker == "mlflow":
        import mlflow

        mlflow.log_artifacts(str(adapter_dir), artifact_path="grpo_lora_adapter")
        mlflow.end_run()


def _generate_run_name(cfg: dict) -> str:
    name = cfg.get("run_name")
    if name:
        return name
    ts = datetime.now().strftime("%m%d_%H%M")
    train = cfg.get("training", {})
    lr = train.get("learning_rate", "?")
    ep = train.get("num_train_epochs", "?")
    ng = train.get("num_generations", "?")
    return f"grpo_lr{lr}_ep{ep}_g{ng}_{ts}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO RL fine-tuning for Nemotron.")
    parser.add_argument("--model-path", type=str, default=str(KAGGLE_MODEL_PATH), help="Path to base model")
    parser.add_argument("--adapter-path", type=Path, default=None, help="Path to SFT LoRA adapter (optional)")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "train.csv", help="Training CSV with prompts+answers")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "grpo_output", help="Output directory")
    parser.add_argument("--config", type=Path, default=None, help="YAML config")
    parser.add_argument("--tracker", type=str, default=None, choices=["wandb", "mlflow", "none"])
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    # Load config
    full_cfg: dict = {}
    train_cfg = dict(DEFAULT_GRPO_ARGS)
    if args.config and args.config.exists():
        with open(args.config) as f:
            full_cfg = yaml.safe_load(f) or {}
        train_cfg.update(full_cfg.get("training", {}))

    tracker = args.tracker or full_cfg.get("tracker", "none")
    if args.run_name:
        full_cfg["run_name"] = args.run_name
    run_name = _generate_run_name(full_cfg)

    # Load dataset
    print(f"Loading training data from {args.data} ...")
    train_df = pd.read_csv(args.data)
    train_df["puzzle_type"] = train_df["prompt"].apply(classify_puzzle)
    dataset = build_grpo_dataset(train_df)
    print(f"GRPO training prompts: {len(dataset)}")

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    print("Loading model in BF16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    print(f"Model parameters: {model.num_parameters():,}")

    # Merge SFT adapter if provided
    if args.adapter_path and args.adapter_path.exists():
        print(f"Loading SFT adapter from {args.adapter_path} ...")
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
        model = model.merge_and_unload()
        print("SFT adapter merged into base model")

    # Build reward function
    reward_fn = make_reward_fn(dataset)

    # Remove non-GRPOConfig fields
    max_prompt_length = train_cfg.pop("max_prompt_length", 512)
    max_completion_length = train_cfg.pop("max_completion_length", 4096)
    num_generations = train_cfg.pop("num_generations", 4)
    temperature = train_cfg.pop("temperature", 0.7)

    hf_report_to = {"wandb": "wandb", "mlflow": "mlflow"}.get(tracker, "none")

    all_params = {
        "training": {
            **train_cfg,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
            "num_generations": num_generations,
            "temperature": temperature,
        },
        "dataset_size": len(dataset),
        "model_path": args.model_path,
        "sft_adapter": str(args.adapter_path) if args.adapter_path else "none",
    }

    # Init tracker
    if tracker in ("wandb", "mlflow"):
        _setup_tracker(tracker, full_cfg, run_name, all_params)
        print(f"Tracking with {tracker} (run: {run_name})")

    # GRPO Config
    output_dir = str(args.output_dir)
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        temperature=temperature,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=hf_report_to,
        run_name=run_name,
        **train_cfg,
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    print("Starting GRPO training ...")
    trainer.train()
    print("GRPO training complete!")

    # Save adapter
    adapter_dir = args.output_dir / "lora_adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"GRPO adapter saved to {adapter_dir}")

    # Log artifact and close tracker
    if tracker != "none":
        _finish_tracker(tracker, adapter_dir)
        print(f"Artifacts logged to {tracker}")


if __name__ == "__main__":
    main()
