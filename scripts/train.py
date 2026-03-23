"""Fine-tune Nemotron-3-Nano-30B with LoRA using SFT."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from nemotron.config import (
    DATA_DIR,
    DEFAULT_LORA_CONFIG,
    DEFAULT_TRAINING_ARGS,
    KAGGLE_MODEL_PATH,
    OUTPUT_DIR,
)
from nemotron.data_utils import load_jsonl


# ---------------------------------------------------------------------------
# Experiment tracking helpers
# ---------------------------------------------------------------------------


def _setup_wandb(cfg: dict, run_name: str, all_params: dict) -> None:
    """Initialize wandb run and log config."""
    import wandb

    wandb_cfg = cfg.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "nemotron-reasoning"),
        entity=wandb_cfg.get("entity"),
        name=run_name,
        tags=_tags_list(cfg),
        config=all_params,
    )


def _setup_mlflow(cfg: dict, run_name: str, all_params: dict) -> None:
    """Initialize mlflow run and log config."""
    import mlflow

    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.get("experiment_name", "nemotron-sft"))
    mlflow.start_run(run_name=run_name, tags=cfg.get("tags", {}))
    mlflow.log_params(_flatten_dict(all_params))


def _finish_tracking(tracker: str, adapter_dir: Path) -> None:
    """Log final artifacts and close the tracking run."""
    if tracker == "wandb":
        import wandb

        artifact = wandb.Artifact("lora-adapter", type="model")
        artifact.add_dir(str(adapter_dir))
        wandb.log_artifact(artifact)
        wandb.finish()
    elif tracker == "mlflow":
        import mlflow

        mlflow.log_artifacts(str(adapter_dir), artifact_path="lora_adapter")
        mlflow.end_run()


def _tags_list(cfg: dict) -> list[str]:
    """Convert tags dict to a list of 'key:value' strings for wandb."""
    tags = cfg.get("tags", {})
    return [f"{k}:{v}" for k, v in tags.items()] if isinstance(tags, dict) else list(tags)


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict for mlflow param logging."""
    items: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        elif isinstance(v, list):
            items[key] = str(v)
        else:
            items[key] = v
    return items


def _generate_run_name(cfg: dict) -> str:
    """Generate a descriptive run name from config."""
    name = cfg.get("run_name")
    if name:
        return name
    ts = datetime.now().strftime("%m%d_%H%M")
    train = cfg.get("training", {})
    lora = cfg.get("lora", {})
    lr = train.get("learning_rate", "?")
    r = lora.get("r", "?")
    ep = train.get("num_train_epochs", "?")
    return f"sft_r{r}_lr{lr}_ep{ep}_{ts}"


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT LoRA fine-tuning for Nemotron.")
    parser.add_argument("--model-path", type=str, default=str(KAGGLE_MODEL_PATH), help="Path to base model")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "sft_train.jsonl", help="Training JSONL")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "sft_output", help="Checkpoint directory")
    parser.add_argument("--config", type=Path, default=None, help="YAML config to override defaults")
    parser.add_argument(
        "--tracker", type=str, default=None, choices=["wandb", "mlflow", "none"], help="Override tracker from config"
    )
    parser.add_argument("--run-name", type=str, default=None, help="Override run name")
    args = parser.parse_args()

    # Load config overrides
    full_cfg: dict = {}
    train_cfg = dict(DEFAULT_TRAINING_ARGS)
    lora_cfg = dict(DEFAULT_LORA_CONFIG)
    if args.config and args.config.exists():
        with open(args.config) as f:
            full_cfg = yaml.safe_load(f) or {}
        train_cfg.update(full_cfg.get("training", {}))
        lora_cfg.update(full_cfg.get("lora", {}))

    # Resolve tracker
    tracker = args.tracker or full_cfg.get("tracker", "none")
    if args.run_name:
        full_cfg["run_name"] = args.run_name
    run_name = _generate_run_name(full_cfg)

    # Load data
    print(f"Loading training data from {args.data} ...")
    raw_data = load_jsonl(args.data)
    dataset = Dataset.from_list(raw_data)
    print(f"Training examples: {len(dataset)}")

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    print(f"Loading model in BF16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    print(f"Model parameters: {model.num_parameters():,}")

    # LoRA
    target_modules = lora_cfg.pop("target_modules")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        **lora_cfg,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Determine HF report_to value
    hf_report_to = {"wandb": "wandb", "mlflow": "mlflow"}.get(tracker, "none")

    # Collect all params for logging
    all_params = {
        "training": train_cfg,
        "lora": {**lora_cfg, "target_modules": target_modules},
        "dataset_size": len(dataset),
        "model_path": args.model_path,
    }

    # Initialize tracker
    if tracker == "wandb":
        _setup_wandb(full_cfg, run_name, all_params)
        print(f"Tracking with wandb (run: {run_name})")
    elif tracker == "mlflow":
        _setup_mlflow(full_cfg, run_name, all_params)
        print(f"Tracking with mlflow (run: {run_name})")
    else:
        print("Experiment tracking disabled")

    # Training
    output_dir = str(args.output_dir)
    training_args = SFTConfig(
        output_dir=output_dir,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=True,
        report_to=hf_report_to,
        run_name=run_name,
        **train_cfg,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training ...")
    trainer.train()
    print("Training complete!")

    # Save adapter
    adapter_dir = args.output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to {adapter_dir}")

    # Log artifact and close tracker
    if tracker != "none":
        _finish_tracking(tracker, adapter_dir)
        print(f"Artifacts logged to {tracker}")


if __name__ == "__main__":
    main()
