#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv run python -m miniLLM.eval \
    --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
    --dataset-name b-mc2/sql-create-context \
    --num-samples 20
else
  python -m miniLLM.eval \
    --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
    --dataset-name b-mc2/sql-create-context \
    --num-samples 20
fi


