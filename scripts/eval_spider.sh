#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-}
BACKEND=${BACKEND:-hf}
VLLM_URL=${VLLM_URL:-http://localhost:8001}
SPIDER_DB_DIR=${SPIDER_DB_DIR:-}
MAX_SAMPLES=${MAX_SAMPLES:-}
REPORT_PATH=${REPORT_PATH:-outputs/spider_eval_report.json}

run_cmd() {
  if command -v uv >/dev/null 2>&1; then
    uv run "$@"
  else
    "$@"
  fi
}

CMD=(python -m miniLLM.eval_spider
  --model-name-or-path "$MODEL_NAME_OR_PATH"
  --backend "$BACKEND"
  --vllm-url "$VLLM_URL"
  --report-path "$REPORT_PATH")

if [[ -n "$ADAPTER_PATH" ]]; then
  CMD+=(--adapter-path "$ADAPTER_PATH")
fi

if [[ -n "$SPIDER_DB_DIR" ]]; then
  CMD+=(--spider-db-dir "$SPIDER_DB_DIR")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

run_cmd "${CMD[@]}"
