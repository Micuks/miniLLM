#!/usr/bin/env bash
set -euo pipefail

HF_URL=${HF_URL:-http://localhost:8000}
VLLM_URL=${VLLM_URL:-http://localhost:8001}
N_SAMPLES=${N_SAMPLES:-50}
CONCURRENCY=${CONCURRENCY:-1,4,8,16,32}
OUTPUT=${OUTPUT:-outputs/throughput_bench.json}

run_cmd() {
  if command -v uv >/dev/null 2>&1; then
    uv run "$@"
  else
    "$@"
  fi
}

echo "=== HF vs vLLM Throughput Benchmark ==="
echo "HF:   $HF_URL"
echo "vLLM: $VLLM_URL"
echo "Samples: $N_SAMPLES, Concurrency: $CONCURRENCY"

run_cmd python -m miniLLM.bench.throughput_bench \
  --hf-url "$HF_URL" \
  --vllm-url "$VLLM_URL" \
  --n-samples "$N_SAMPLES" \
  --concurrency "$CONCURRENCY" \
  --output "$OUTPUT"
