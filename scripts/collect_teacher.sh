#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-api}
DATASET=${DATASET_NAME:-b-mc2/sql-create-context}
OUTPUT=${OUTPUT:-outputs/teacher_outputs.jsonl}
N_SAMPLES=${N_SAMPLES:-}
TOP_K=${TOP_K:-20}

# API mode
API_BASE=${API_BASE:-https://api.together.xyz/v1}
API_KEY=${API_KEY:-}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen2.5-72B-Instruct}

# Local mode
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-}

run_cmd() {
  if command -v uv >/dev/null 2>&1; then
    uv run "$@"
  else
    "$@"
  fi
}

CMD=(python -m miniLLM.distill.collect_teacher
  --mode "$MODE"
  --dataset-name "$DATASET"
  --output-path "$OUTPUT"
  --top-k "$TOP_K")

if [[ -n "$N_SAMPLES" ]]; then
  CMD+=(--n-samples "$N_SAMPLES")
fi

if [[ "$MODE" == "api" ]]; then
  CMD+=(--api-base "$API_BASE" --api-key "$API_KEY" --teacher-model "$TEACHER_MODEL")
else
  CMD+=(--model-name-or-path "$MODEL_NAME_OR_PATH")
fi

run_cmd "${CMD[@]}"
