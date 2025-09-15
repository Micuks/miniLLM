#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv run python -m miniLLM.train \
    --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
    --dataset-name b-mc2/sql-create-context \
    --output-dir outputs/sft-qwen2.5-7b-instruct-sql \
    --num-train-epochs 1 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-4 \
    --lora-r 32 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --max-steps 100 || true
else
  python -m miniLLM.train \
    --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
    --dataset-name b-mc2/sql-create-context \
    --output-dir outputs/sft-qwen2.5-7b-instruct-sql \
    --num-train-epochs 1 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-4 \
    --lora-r 32 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --max-steps 100 || true
fi


