"""Configuration and constants for training and submission."""

from pathlib import Path

# ---- Paths ----
# Local paths (for development)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Kaggle paths (for notebook execution)
KAGGLE_DATA_DIR = Path("/kaggle/input/nvidia-nemotron-model-reasoning-challenge")
KAGGLE_MODEL_PATH = Path("/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default/1")
KAGGLE_WORKING_DIR = Path("/kaggle/working")

# ---- Model ----
MODEL_NAME = "nvidia/nemotron-3-nano-30b-a3b-bf16"
MAX_LORA_RANK = 32

# ---- Training Defaults ----
SYSTEM_PROMPT = (
    "You are a helpful assistant that solves reasoning puzzles. "
    "Think through each problem step by step, showing your work clearly. "
    "Always put your final answer inside \\boxed{} at the end of your response."
)

DEFAULT_TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "bf16": True,
    "max_seq_length": 4096,
    "gradient_checkpointing": True,
    "optim": "adamw_torch_fused",
    "seed": 42,
}

# ---- LoRA Defaults ----
DEFAULT_LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}

# ---- Puzzle Oversampling Weights ----
# Higher weight = more repetitions for hard puzzle types
DIFFICULTY_WEIGHTS = {
    "text_encryption": 4,
    "bit_manipulation": 4,
    "equation_transform": 3,
    "gravitational": 2,
    "unit_conversion": 2,
    "number_base": 1,
    "unknown": 2,
}
