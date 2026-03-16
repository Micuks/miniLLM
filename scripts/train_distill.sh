#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}
TEACHER_DATA=${TEACHER_DATA:-outputs/teacher_outputs.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/distill-qwen2.5-7b}

ALPHA=${ALPHA:-1.0}
BETA=${BETA:-0.5}
GAMMA=${GAMMA:-0.5}
KD_TEMP=${KD_TEMP:-4.0}
MAX_STEPS=${MAX_STEPS:-100}

run_cmd() {
  if command -v uv >/dev/null 2>&1; then
    uv run "$@"
  else
    "$@"
  fi
}

run_cmd python -m miniLLM.train_distill \
  --model-name-or-path "$MODEL" \
  --teacher-data "$TEACHER_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --lora-r 32 --lora-alpha 16 --lora-dropout 0.05 \
  --alpha "$ALPHA" --beta "$BETA" --gamma "$GAMMA" \
  --kd-temperature "$KD_TEMP" \
  --max-steps "$MAX_STEPS"
