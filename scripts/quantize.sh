#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}
OUTPUT_BASE=${OUTPUT_BASE:-outputs/quantized}
N_CALIBRATION=${N_CALIBRATION:-128}
DATASET=${DATASET_NAME:-b-mc2/sql-create-context}

run_cmd() {
  if command -v uv >/dev/null 2>&1; then
    uv run "$@"
  else
    "$@"
  fi
}

echo "=== GPTQ Quantization ==="
run_cmd python -m miniLLM.quant.quantize_gptq \
  --model-name-or-path "$MODEL" \
  --output-dir "${OUTPUT_BASE}/gptq-4bit" \
  --bits 4 --group-size 128 \
  --n-calibration "$N_CALIBRATION" \
  --dataset-name "$DATASET"

echo "=== AWQ Quantization ==="
run_cmd python -m miniLLM.quant.quantize_awq \
  --model-name-or-path "$MODEL" \
  --output-dir "${OUTPUT_BASE}/awq-4bit" \
  --bits 4 --group-size 128 \
  --n-calibration "$N_CALIBRATION" \
  --dataset-name "$DATASET"

echo "=== Done. Models saved to ${OUTPUT_BASE}/ ==="
