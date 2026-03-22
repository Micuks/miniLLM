#!/usr/bin/env bash
# GPU Watchdog: wait for stable free VRAM, then run benchmark.
# Checks every 60s. Only triggers after GPU stays free for 3 consecutive checks (3 min).
#
# Usage: nohup bash scripts/gpu_watchdog.sh &
set -euo pipefail

MIN_FREE_MB=6000
STABLE_CHECKS=3
CHECK_INTERVAL=60
LOG=outputs/gpu_watchdog.log

mkdir -p outputs

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

consecutive=0

log "Watchdog started. Waiting for ${MIN_FREE_MB}MiB free for ${STABLE_CHECKS} consecutive checks."

while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)

  if [ "$FREE" -ge "$MIN_FREE_MB" ]; then
    consecutive=$((consecutive + 1))
    log "Check passed: ${FREE}MiB free ($consecutive/$STABLE_CHECKS)"

    if [ "$consecutive" -ge "$STABLE_CHECKS" ]; then
      log "=== GPU stable at ${FREE}MiB. Starting experiments ==="
      break
    fi
  else
    if [ "$consecutive" -gt 0 ]; then
      log "Reset: ${FREE}MiB free (was $consecutive/$STABLE_CHECKS)"
    fi
    consecutive=0
  fi

  sleep "$CHECK_INTERVAL"
done

cd "$(dirname "$0")/.."

# --- Experiment 1: vLLM benchmark ---
log "=== Running vLLM benchmark ==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c "
import json, time, torch, gc
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from miniLLM.prompts import build_inference_prompt

MODEL = 'Qwen/Qwen2.5-3B-Instruct'
ADAPTER = 'outputs/sft-qwen2.5-3b-instruct-sql'
N = 50

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset('b-mc2/sql-create-context', split='train').select(range(N))
samples = [{'schema': s['context'], 'question': s['question']} for s in ds]
prompts = [build_inference_prompt(s['schema'], s['question'], tokenizer) for s in samples]
print(f'Loaded {len(samples)} samples')

# --- HF ---
print('\n=== HF Backend ===')
quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(MODEL,
    quantization_config=quant_cfg, device_map='auto', trust_remote_code=True)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()
hf_gpu = torch.cuda.memory_allocated() / 1e9
print(f'GPU mem: {hf_gpu:.2f} GB')

# warmup
inp = tokenizer(prompts[0], return_tensors='pt').to(model.device)
with torch.no_grad(): model.generate(**inp, max_new_tokens=64, do_sample=False)

hf_lats = []
for i, p in enumerate(prompts):
    inp = tokenizer(p, return_tensors='pt').to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=256, do_sample=False, eos_token_id=tokenizer.eos_token_id)
    hf_lats.append((time.perf_counter() - t0) * 1000)
    if (i+1) % 10 == 0: print(f'  [{i+1}/{N}] {hf_lats[-1]:.0f}ms')

hf_lats.sort()
n = len(hf_lats)
import statistics
hf = {
    'backend': 'HuggingFace (sync)', 'n': n,
    'mean_ms': round(statistics.mean(hf_lats), 1),
    'p50_ms': round(hf_lats[n//2], 1),
    'p95_ms': round(hf_lats[int(n*0.95)], 1),
    'rps': round(n / (sum(hf_lats)/1000), 3),
    'gpu_gb': round(hf_gpu, 2),
}
print(f'HF: mean={hf[\"mean_ms\"]}ms p50={hf[\"p50_ms\"]}ms p95={hf[\"p95_ms\"]}ms rps={hf[\"rps\"]}')

del model, base; torch.cuda.empty_cache(); gc.collect()
print('HF model freed')

# --- vLLM ---
print('\n=== vLLM Backend ===')
try:
    from vllm import LLM, SamplingParams
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    util = min(0.85, (free_gb - 1.0) / 23.64)  # leave 1GB headroom
    print(f'Free: {free_gb:.1f}GB, using gpu_memory_utilization={util:.2f}')

    llm = LLM(model=MODEL, max_model_len=2048,
        gpu_memory_utilization=util, trust_remote_code=True,
        quantization='bitsandbytes', load_format='bitsandbytes',
        enforce_eager=True)
    vllm_gpu = torch.cuda.memory_allocated() / 1e9

    params = SamplingParams(max_tokens=256, temperature=0)
    llm.generate(prompts[:2], params)  # warmup

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    total = time.perf_counter() - t0

    vllm = {
        'backend': 'vLLM (offline batch)', 'n': N,
        'total_s': round(total, 2),
        'mean_ms': round(total/N*1000, 1),
        'rps': round(N/total, 3),
        'gpu_gb': round(vllm_gpu, 2),
    }
    speedup = vllm['rps'] / hf['rps']
    print(f'vLLM: total={total:.1f}s mean={vllm[\"mean_ms\"]}ms rps={vllm[\"rps\"]} speedup={speedup:.1f}x')

    del llm; torch.cuda.empty_cache(); gc.collect()
except Exception as e:
    print(f'vLLM failed: {e}')
    vllm = {'backend': 'vLLM', 'error': str(e)}
    speedup = None

# --- Save ---
report = {'hf': hf, 'vllm': vllm}
if speedup: report['speedup'] = round(speedup, 2)
Path('outputs/bench_hf_vs_vllm.json').write_text(json.dumps(report, indent=2))
print(f'\nReport saved to outputs/bench_hf_vs_vllm.json')
" 2>&1 | tee -a "$LOG"

log "=== Benchmark complete ==="

# --- Experiment 2: Held-out eval (if not already done) ---
if [ ! -f outputs/eval_baseline_3b_held_server.json ]; then
  log "=== Running held-out baseline eval ==="
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m miniLLM.eval \
    --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
    --dataset-name b-mc2/sql-create-context \
    --num-samples 200 --eval-offset 78377 \
    --load-in-4bit --with-execution \
    --report-path outputs/eval_baseline_3b_held_server.json 2>&1 | tee -a "$LOG"
  log "Baseline eval done"

  log "=== Running held-out fine-tuned eval ==="
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m miniLLM.eval \
    --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
    --adapter-path outputs/sft-qwen2.5-3b-instruct-sql \
    --dataset-name b-mc2/sql-create-context \
    --num-samples 200 --eval-offset 78377 \
    --load-in-4bit --with-execution \
    --report-path outputs/eval_finetuned_3b_held_server.json 2>&1 | tee -a "$LOG"
  log "Fine-tuned eval done"
fi

log "=== All experiments complete ==="
