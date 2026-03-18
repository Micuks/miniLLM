#!/usr/bin/env bash
# Run held-out evaluation: train on samples [500:], eval on samples [78377:78577]
# This ensures zero overlap between train and eval data.
#
# Prerequisites:
#   - At least 5GB free GPU VRAM (check: nvidia-smi --query-gpu=memory.free --format=csv,noheader)
#   - Trained adapter at outputs/sft-qwen2.5-3b-instruct-sql/
#
# Usage:
#   bash scripts/run_held_out_eval.sh
set -euo pipefail

MODEL=Qwen/Qwen2.5-3B-Instruct
ADAPTER=outputs/sft-qwen2.5-3b-instruct-sql
DATASET=b-mc2/sql-create-context
EVAL_OFFSET=78377
NUM_SAMPLES=200

run_cmd() {
  if command -v uv >/dev/null 2>&1; then
    uv run "$@"
  else
    "$@"
  fi
}

echo "=== Step 1: Baseline eval (no adapter) ==="
run_cmd python -m miniLLM.eval \
  --model-name-or-path "$MODEL" \
  --dataset-name "$DATASET" \
  --num-samples "$NUM_SAMPLES" \
  --eval-offset "$EVAL_OFFSET" \
  --load-in-4bit \
  --with-execution \
  --report-path outputs/eval_baseline_3b_held.json

echo ""
echo "=== Step 2: Fine-tuned eval (with adapter) ==="
run_cmd python -m miniLLM.eval \
  --model-name-or-path "$MODEL" \
  --adapter-path "$ADAPTER" \
  --dataset-name "$DATASET" \
  --num-samples "$NUM_SAMPLES" \
  --eval-offset "$EVAL_OFFSET" \
  --load-in-4bit \
  --with-execution \
  --report-path outputs/eval_finetuned_3b_held.json

echo ""
echo "=== Step 3: Difficulty analysis ==="
run_cmd python -c "
import json
from miniLLM.data.spider import _classify_difficulty

for name, path in [('Baseline', 'outputs/eval_baseline_3b_held.json'), ('Fine-tuned', 'outputs/eval_finetuned_3b_held.json')]:
    with open(path) as f:
        data = json.load(f)
    by_diff = {}
    for r in data['records']:
        diff = _classify_difficulty(r['gold_sql'])
        if diff not in by_diff:
            by_diff[diff] = {'total': 0, 'em': 0, 'ex': 0, 'ex_attempted': 0}
        by_diff[diff]['total'] += 1
        by_diff[diff]['em'] += int(r['exact_match'])
        if r['execution_match'] is not None:
            by_diff[diff]['ex_attempted'] += 1
            by_diff[diff]['ex'] += int(r['execution_match'])
    print(f'\n=== {name} (n={len(data[\"records\"])}) ===')
    print(f'Overall: EM={data[\"summary\"][\"exact_match\"]:.1%}, EX={data[\"summary\"][\"execution_match\"]:.1%}')
    for diff in ['easy', 'medium', 'hard']:
        if diff in by_diff:
            d = by_diff[diff]
            em = d['em']/d['total']
            ex = d['ex']/d['ex_attempted'] if d['ex_attempted'] else 0
            print(f'  {diff:8s}: n={d[\"total\"]:3d}, EM={em:.1%}, EX={ex:.1%}')
"

echo ""
echo "=== Done! Reports saved to outputs/eval_*_held.json ==="
