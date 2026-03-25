#!/usr/bin/env bash
# Full data generation pipeline.
#
# Generates training data in 3 tiers:
#   1. Programmatic solvers   (free, instant)
#   2. GitHub Models API      (free with GitHub account)
#   3. OpenAI API / ChatGPT   (paid, resumes from where tier 2 left off)
#
# Each step is crash-safe and resumable.  Re-run to continue.
#
# Usage:
#   # Tier 1 only (no API key needed)
#   bash scripts/run_pipeline.sh --data-dir data/
#
#   # Tier 1 + 2 (set GITHUB_TOKEN)
#   export GITHUB_TOKEN="ghp_..."
#   bash scripts/run_pipeline.sh --data-dir data/
#
#   # Tier 1 + 2 + 3 (set both tokens)
#   export GITHUB_TOKEN="ghp_..."
#   export OPENAI_API_KEY="sk-..."
#   bash scripts/run_pipeline.sh --data-dir data/
#
set -euo pipefail

DATA_DIR="${1:-data}"
SOLVER_OUTPUT="${DATA_DIR}/solver_train.jsonl"
TEACHER_OUTPUT="${DATA_DIR}/teacher_train.jsonl"
FINAL_OUTPUT="${DATA_DIR}/sft_train.jsonl"

echo "============================================"
echo "  Nemotron Training Data Pipeline"
echo "============================================"
echo "Data dir: ${DATA_DIR}"
echo ""

# ------------------------------------------------------------------
# Tier 1: Programmatic solvers (free, no API)
# ------------------------------------------------------------------
echo "[Tier 1] Running programmatic solvers..."
python scripts/prepare_data.py \
    --data-dir "${DATA_DIR}" \
    --output "${SOLVER_OUTPUT}" \
    --no-oversample

SOLVER_COUNT=$(wc -l < "${SOLVER_OUTPUT}" 2>/dev/null || echo "0")
echo "[Tier 1] Done: ${SOLVER_COUNT} examples from programmatic solvers"
echo ""

# ------------------------------------------------------------------
# Tier 2: GitHub Models (free with GitHub account)
# ------------------------------------------------------------------
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    echo "[Tier 2] Using GitHub Models (openai/gpt-4o)..."
    python scripts/generate_teacher_data.py \
        --provider github \
        --model openai/gpt-4o \
        --data-dir "${DATA_DIR}" \
        --output "${TEACHER_OUTPUT}" \
        --rate-limit 1.0 \
        || echo "[Tier 2] Stopped (quota limit or error). Progress saved."
    echo ""
else
    echo "[Tier 2] Skipped (GITHUB_TOKEN not set)"
    echo "  Set GITHUB_TOKEN to use free GitHub Models API"
    echo ""
fi

# ------------------------------------------------------------------
# Tier 3: OpenAI API / ChatGPT Plus (paid, resumes automatically)
# ------------------------------------------------------------------
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    echo "[Tier 3] Using OpenAI API (gpt-4o)..."
    python scripts/generate_teacher_data.py \
        --provider openai \
        --model gpt-4o \
        --data-dir "${DATA_DIR}" \
        --output "${TEACHER_OUTPUT}" \
        --rate-limit 1.0 \
        || echo "[Tier 3] Stopped (quota limit or error). Progress saved."
    echo ""
else
    echo "[Tier 3] Skipped (OPENAI_API_KEY not set)"
    echo "  Set OPENAI_API_KEY to use ChatGPT Plus API"
    echo ""
fi

# ------------------------------------------------------------------
# Merge: Combine solver + teacher data, deduplicate by prompt
# ------------------------------------------------------------------
echo "[Merge] Combining all training data..."
python -c "
import json
from pathlib import Path

data_dir = Path('${DATA_DIR}')
solver = data_dir / 'solver_train.jsonl'
teacher = data_dir / 'teacher_train.jsonl'
output = data_dir / 'sft_train.jsonl'

seen_prompts = set()
examples = []

# Prefer teacher data (higher quality) -- load it first
for path in [teacher, solver]:
    if not path.exists():
        continue
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Extract user prompt as dedup key
            prompt = None
            for msg in entry.get('messages', []):
                if msg.get('role') == 'user':
                    prompt = msg['content']
                    break
            if prompt and prompt not in seen_prompts:
                seen_prompts.add(prompt)
                examples.append(entry)

with open(output, 'w') as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

print(f'Merged: {len(examples)} unique examples -> {output}')
"

FINAL_COUNT=$(wc -l < "${FINAL_OUTPUT}" 2>/dev/null || echo "0")
echo ""
echo "============================================"
echo "  Pipeline Complete!"
echo "============================================"
echo "  Solver examples:  $(wc -l < "${SOLVER_OUTPUT}" 2>/dev/null || echo 0)"
echo "  Teacher examples: $(wc -l < "${TEACHER_OUTPUT}" 2>/dev/null || echo 0)"
echo "  Final merged:     ${FINAL_COUNT}  (deduplicated)"
echo "  Output:           ${FINAL_OUTPUT}"
echo ""
echo "Next step:"
echo "  python scripts/train.py --data ${FINAL_OUTPUT} --config configs/sft.yaml"
