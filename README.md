# Nemotron Reasoning Challenge

Fine-tuning pipeline for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle.

Train a LoRA adapter on **Nemotron-3-Nano-30B-A3B-BF16** to solve 6 types of reasoning puzzles: number base conversion, unit conversion, gravitational calculations, equation transforms, text encryption, and bit manipulation.

## Repo Structure

```
nemotron/
├── configs/
│   ├── sft.yaml               # SFT local config (mlflow tracking)
│   ├── sft_kaggle.yaml        # SFT Kaggle config (wandb tracking)
│   ├── grpo.yaml              # GRPO local config (mlflow tracking)
│   └── grpo_kaggle.yaml       # GRPO Kaggle config (wandb tracking)
├── data/                      # Competition CSVs + generated JSONL (gitignored)
│   ├── train.csv
│   ├── test.csv
│   └── sft_train.jsonl
├── notebooks/                 # Kaggle notebooks (run as-is on Kaggle)
│   ├── nemotron-eda-puzzle-types.ipynb
│   └── nemotron-sft-lora-submission.ipynb
├── scripts/
│   ├── prepare_data.py            # Generate SFT training JSONL (programmatic solvers)
│   ├── generate_teacher_data.py   # Generate SFT data via teacher LLM (Gemini/DeepSeek)
│   ├── train.py                   # Stage 1: SFT LoRA fine-tuning
│   ├── train_grpo.py              # Stage 2: GRPO reinforcement learning
│   └── package_submission.py      # Zip adapter for Kaggle submission
├── src/nemotron/
│   ├── __init__.py
│   ├── config.py              # Paths, constants, defaults
│   ├── solvers.py             # Programmatic puzzle solvers
│   ├── rewards.py             # Reward functions for GRPO
│   └── data_utils.py          # Data loading + SFT formatting
├── docs/
│   └── next-steps.md
└── pyproject.toml
```

## Quick Start

### 1. Install

```bash
uv sync --extra tracking

# For teacher LLM data generation (optional)
uv sync --extra teacher
```

### 2. Prepare training data

**One-command pipeline** (recommended -- runs all tiers, crash-safe, resumable):
```bash
# Tier 1 only (free, no API key)
bash scripts/run_pipeline.sh data/

# Tier 1 + GitHub Models (free)
export GITHUB_TOKEN="ghp_..."
bash scripts/run_pipeline.sh data/

# Tier 1 + GitHub + OpenAI (covers everything)
export GITHUB_TOKEN="ghp_..."
export OPENAI_API_KEY="sk-..."
bash scripts/run_pipeline.sh data/
```

Each result is saved immediately. Re-run the same command to resume from where you left off.

**Or run individual steps:**

```bash
# Programmatic solvers (free, fast)
python scripts/prepare_data.py --data-dir data/

# Teacher LLM distillation (higher quality)
python scripts/generate_teacher_data.py \
    --provider github --model openai/gpt-4o \
    --data-dir data/ --output data/teacher_train.jsonl
```

Teacher data is validated against known answers before saving.

### 3. Train

**Stage 1: SFT** (supervised fine-tuning)
```bash
python scripts/train.py \
    --model-path /path/to/nemotron-3-nano-30b \
    --data data/teacher_train.jsonl \
    --config configs/sft.yaml
```

**Stage 2: GRPO** (reinforcement learning, builds on SFT)
```bash
python scripts/train_grpo.py \
    --model-path /path/to/nemotron-3-nano-30b \
    --adapter-path outputs/sft_output/lora_adapter \
    --data data/train.csv \
    --config configs/grpo.yaml
```

GRPO generates multiple candidate answers per puzzle and reinforces correct ones.

```bash
# Override tracker or run name from CLI
python scripts/train.py --config configs/sft.yaml --tracker wandb --run-name "exp7_lr1e4_r16"
```

### 4. Package submission

```bash
python scripts/package_submission.py
```

### Kaggle Notebook

The notebooks in `notebooks/` are self-contained and can be uploaded directly to Kaggle. Attach the competition dataset and the Nemotron-3-Nano-30B model, select the RTX PRO 6000 GPU, and run all cells.

## Competition Details

| Item | Value |
|------|-------|
| Model | Nemotron-3-Nano-30B-A3B-BF16 (MoE, 30B total, ~3B active) |
| Adapter | LoRA, max rank 32 |
| Inference | vLLM, temperature=0.0, max_tokens=7680 |
| Scoring | Exact match or relative tolerance 1e-2 |
| GPU | RTX PRO 6000 Blackwell (needs CUDA 12.8+ PyTorch) |

## Experiment Tracking

| Environment | Config | Tracker | View UI |
|---|---|---|---|
| Local | `configs/sft.yaml` | **mlflow** | `mlflow ui --backend-store-uri ./mlruns` |
| Kaggle | `configs/sft_kaggle.yaml` | **wandb** | wandb.ai dashboard |

**Tracked automatically per run:**
- All hyperparameters (training + LoRA config, dataset size, model path)
- Training loss and LR curves (via HuggingFace Trainer integration)
- Auto-generated run names (e.g. `sft_r32_lr0.0002_ep3_0322_2045`)
- Tags for filtering (`model`, `method`, `env`)
- LoRA adapter artifact at the end of training

## Approach

1. **Teacher LLM distillation** -- use Gemini/DeepSeek to generate high-quality CoT training data (validated against known answers)
2. **SFT (Stage 1)** -- fine-tune with LoRA to learn reasoning-then-boxed-answer format
3. **GRPO (Stage 2)** -- reinforcement learning that samples multiple answers and reinforces correct ones
4. **Oversampling** hard puzzle types (encryption 4x, bit manipulation 4x, equations 3x)
5. **BF16 training** matches the BF16 inference precision (QLoRA adapters degrade at BF16 inference)

### Training Pipeline

```
train.csv --> [Teacher LLM] --> teacher_train.jsonl --> [SFT] --> sft_adapter/
                                                                      |
              train.csv ----------------------------> [GRPO] <--------+
                                                         |
                                                   grpo_adapter/ --> submission.zip
```

