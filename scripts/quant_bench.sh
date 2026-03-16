#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}
QUANT_BASE=${QUANT_BASE:-outputs/quantized}
N_SAMPLES=${N_SAMPLES:-100}
REPORT_DIR=${REPORT_DIR:-outputs}

run_cmd() {
  if command -v uv >/dev/null 2>&1; then
    uv run "$@"
  else
    "$@"
  fi
}

echo "=== Benchmarking FP16 ==="
run_cmd python -m miniLLM.quant.benchmark \
  --model-path "$MODEL" --variant fp16 \
  --n-samples "$N_SAMPLES" --output "${REPORT_DIR}/bench_fp16.json"

echo "=== Benchmarking QLoRA NF4 ==="
run_cmd python -m miniLLM.quant.benchmark \
  --model-path "$MODEL" --variant qlora-nf4 \
  --n-samples "$N_SAMPLES" --output "${REPORT_DIR}/bench_qlora_nf4.json"

echo "=== Benchmarking GPTQ ==="
run_cmd python -m miniLLM.quant.benchmark \
  --model-path "${QUANT_BASE}/gptq-4bit" --variant gptq \
  --n-samples "$N_SAMPLES" --output "${REPORT_DIR}/bench_gptq.json"

echo "=== Benchmarking AWQ ==="
run_cmd python -m miniLLM.quant.benchmark \
  --model-path "${QUANT_BASE}/awq-4bit" --variant awq \
  --n-samples "$N_SAMPLES" --output "${REPORT_DIR}/bench_awq.json"

echo "=== Generating Report ==="
run_cmd python -m miniLLM.quant.report \
  "${REPORT_DIR}/bench_fp16.json" \
  "${REPORT_DIR}/bench_qlora_nf4.json" \
  "${REPORT_DIR}/bench_gptq.json" \
  "${REPORT_DIR}/bench_awq.json" \
  --output "${REPORT_DIR}/quant_report.json"

echo "=== Report saved to ${REPORT_DIR}/quant_report.json ==="
