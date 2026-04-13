#!/usr/bin/env bash
# GRPO RL: train the agent with execution-based rewards and DeepSpeed ZeRO-2
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-outputs/react-sft}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo-agent}"
MAX_STEPS="${MAX_STEPS:-300}"
NUM_GEN="${NUM_GEN:-4}"
LR="${LR:-3e-6}"
TEMP="${TEMP:-0.6}"
MIN_TEMP="${MIN_TEMP:-0.2}"
TOP_P="${TOP_P:-0.95}"
GREEDY_GEN="${GREEDY_GEN:-1}"
KL_COEF="${KL_COEF:-0.02}"
REWARD_PROFILE="${REWARD_PROFILE:-dense}"

echo "=== Stable GRPO Training with DeepSpeed ZeRO-2 ==="
echo "  Model: $MODEL"
echo "  SFT adapter: $SFT_ADAPTER"
echo "  Output: $OUTPUT_DIR"
echo "  Generations per prompt: $NUM_GEN"
echo "  Temperature: $TEMP -> $MIN_TEMP"
echo "  Greedy completions per group: $GREEDY_GEN"
echo "  KL coefficient: $KL_COEF"
echo "  Reward profile: $REWARD_PROFILE"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -x "$REPO_ROOT/.venv/bin/deepspeed" ]; then
    DS_CMD=("$REPO_ROOT/.venv/bin/deepspeed")
elif command -v deepspeed >/dev/null 2>&1; then
    DS_CMD=(deepspeed)
elif command -v uv >/dev/null 2>&1; then
    DS_CMD=(uv run deepspeed)
else
    echo "Error: neither 'deepspeed' nor 'uv' is available in PATH." >&2
    exit 1
fi

echo "  Launcher: ${DS_CMD[*]}"

"${DS_CMD[@]}" --num_gpus=1 --module miniLLM.train_grpo \
    --model-name-or-path "$MODEL" \
    --adapter-path "$SFT_ADAPTER" \
    --source spider \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --num-generations "$NUM_GEN" \
    --max-completion-length 1024 \
    --temperature "$TEMP" \
    --min-temperature "$MIN_TEMP" \
    --top-p "$TOP_P" \
    --greedy-generations "$GREEDY_GEN" \
    --learning-rate "$LR" \
    --kl-coef "$KL_COEF" \
    --reward-profile "$REWARD_PROFILE" \
    --curriculum \
    --deepspeed configs/ds_zero2.json
