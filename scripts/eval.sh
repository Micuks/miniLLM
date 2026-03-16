#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}
DATASET_NAME=${DATASET_NAME:-b-mc2/sql-create-context}
NUM_SAMPLES=${NUM_SAMPLES:-20}
ADAPTER_PATH=${ADAPTER_PATH:-}
WITH_EXECUTION=${WITH_EXECUTION:-0}
REPORT_PATH=${REPORT_PATH:-outputs/eval_report.json}

CMD=(python -m miniLLM.eval \
  --model-name-or-path "$MODEL_NAME_OR_PATH" \
  --dataset-name "$DATASET_NAME" \
  --num-samples "$NUM_SAMPLES" \
  --report-path "$REPORT_PATH")

if [[ -n "$ADAPTER_PATH" ]]; then
  CMD+=(--adapter-path "$ADAPTER_PATH")
fi

if [[ "$WITH_EXECUTION" == "1" ]]; then
  CMD+=(--with-execution)
fi

if command -v uv >/dev/null 2>&1; then
  uv run "${CMD[@]}"
else
  "${CMD[@]}"
fi
