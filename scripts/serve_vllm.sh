#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}

if command -v uv >/dev/null 2>&1; then
  uv run uvicorn miniLLM.service.vllm_app:app --host 0.0.0.0 --port 8001
else
  uvicorn miniLLM.service.vllm_app:app --host 0.0.0.0 --port 8001
fi
