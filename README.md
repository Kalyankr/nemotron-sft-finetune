# Nemotron Reasoning Challenge

Fine-tuning pipeline for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle.

Train a LoRA adapter on **Nemotron-3-Nano-30B-A3B-BF16** to solve 6 types of reasoning puzzles: number base conversion, unit conversion, gravitational calculations, equation transforms, text encryption, and bit manipulation.

## Repo Structure

```
nemotron/
├── configs/
│   ├── sft.yaml               # Local config (mlflow tracking)
│   └── sft_kaggle.yaml        # Kaggle config (wandb tracking)
├── data/                      # Competition CSVs + generated JSONL (gitignored)
│   ├── train.csv
│   ├── test.csv
│   └── sft_train.jsonl
├── notebooks/                 # Kaggle notebooks (run as-is on Kaggle)
│   ├── nemotron-eda-puzzle-types.ipynb
│   └── nemotron-sft-lora-submission.ipynb
├── scripts/
│   ├── prepare_data.py        # Generate SFT training JSONL
│   ├── train.py               # LoRA fine-tuning + experiment tracking
│   └── package_submission.py  # Zip adapter for Kaggle submission
├── src/nemotron/
│   ├── __init__.py
│   ├── config.py              # Paths, constants, defaults
│   ├── solvers.py             # Programmatic puzzle solvers
│   └── data_utils.py          # Data loading + SFT formatting
├── docs/
│   └── next-steps.md
└── pyproject.toml
```

## Quick Start

### 1. Install

```bash
uv sync --extra tracking
```

### 2. Prepare training data

```bash
python scripts/prepare_data.py --data-dir data/
```

This classifies all puzzles, solves them programmatically, generates Chain-of-Thought reasoning traces, and oversamples hard puzzle types.

### 3. Train (on GPU with enough VRAM)

```bash
# Local with mlflow tracking (default)
python scripts/train.py \
    --model-path /path/to/nemotron-3-nano-30b \
    --data data/sft_train.jsonl \
    --config configs/sft.yaml

# Kaggle with wandb tracking
python scripts/train.py \
    --model-path /path/to/nemotron-3-nano-30b \
    --config configs/sft_kaggle.yaml

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

1. **Programmatic solvers** generate verified correct answers + CoT traces for training data
2. **SFT** fine-tunes the model to follow the reasoning-then-boxed-answer format
3. **Oversampling** hard puzzle types (encryption 4x, bit manipulation 4x, equations 3x)
4. **BF16 training** matches the BF16 inference precision (QLoRA adapters degrade at BF16 inference)

