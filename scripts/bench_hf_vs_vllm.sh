#!/usr/bin/env bash
# Benchmark HF generate vs vLLM: single-GPU, sequential comparison.
# Only one model loaded at a time — needs ~5GB free VRAM.
#
# Measures: latency per query (mean, p50, p95) and throughput.
# Usage: bash scripts/bench_hf_vs_vllm.sh
set -euo pipefail

MODEL=Qwen/Qwen2.5-3B-Instruct
ADAPTER=outputs/sft-qwen2.5-3b-instruct-sql
N_SAMPLES=50
REPORT=outputs/bench_hf_vs_vllm.json

run_cmd() {
  if command -v uv >/dev/null 2>&1; then
    uv run "$@"
  else
    "$@"
  fi
}

echo "=== HF vs vLLM Single-GPU Benchmark ==="
echo "Model: $MODEL"
echo "Adapter: $ADAPTER"
echo "Samples: $N_SAMPLES"
echo ""

run_cmd python -c "
import json, time, statistics
from pathlib import Path
from datasets import load_dataset

# ---- Load samples ----
ds = load_dataset('b-mc2/sql-create-context', split='train').select(range(${N_SAMPLES}))
samples = [{'schema': s['context'], 'question': s['question']} for s in ds]
print(f'Loaded {len(samples)} samples')

# ---- HF Backend ----
print('\n=== HF Backend (sync generate) ===')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from miniLLM.prompts import build_inference_prompt

tokenizer = AutoTokenizer.from_pretrained('${MODEL}', use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained('${MODEL}',
    quantization_config=quant_cfg, device_map='auto', trust_remote_code=True)
model = PeftModel.from_pretrained(base, '${ADAPTER}')
model.eval()

gpu_mem_hf = torch.cuda.memory_allocated() / 1e9
print(f'GPU memory: {gpu_mem_hf:.2f} GB')

# Warmup
prompt = build_inference_prompt(samples[0]['schema'], samples[0]['question'], tokenizer)
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=64, do_sample=False)

hf_latencies = []
for i, s in enumerate(samples):
    prompt = build_inference_prompt(s['schema'], s['question'], tokenizer)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
            eos_token_id=tokenizer.eos_token_id)
    lat = (time.perf_counter() - t0) * 1000
    hf_latencies.append(lat)
    tokens = out.shape[1] - inputs['input_ids'].shape[1]
    if (i+1) % 10 == 0:
        print(f'  [{i+1}/{len(samples)}] {lat:.0f}ms ({tokens} tokens)')

hf_latencies.sort()
n = len(hf_latencies)
hf_result = {
    'backend': 'HuggingFace (sync)',
    'n_requests': n,
    'mean_ms': statistics.mean(hf_latencies),
    'p50_ms': hf_latencies[n // 2],
    'p95_ms': hf_latencies[int(n * 0.95)],
    'throughput_rps': n / (sum(hf_latencies) / 1000),
    'gpu_mem_gb': gpu_mem_hf,
}
print(f'HF: mean={hf_result[\"mean_ms\"]:.0f}ms, p50={hf_result[\"p50_ms\"]:.0f}ms, '
      f'p95={hf_result[\"p95_ms\"]:.0f}ms, rps={hf_result[\"throughput_rps\"]:.2f}')

# Free HF model
del model, base
torch.cuda.empty_cache()
import gc; gc.collect()
print('HF model unloaded')

# ---- vLLM Backend ----
print('\n=== vLLM Backend (offline) ===')
try:
    from vllm import LLM, SamplingParams

    llm = LLM(model='${MODEL}', quantization='bitsandbytes', load_format='bitsandbytes',
        max_model_len=2048, gpu_memory_utilization=0.7, enable_lora=True,
        max_lora_rank=32, trust_remote_code=True)
    gpu_mem_vllm = torch.cuda.memory_allocated() / 1e9

    prompts = []
    for s in samples:
        prompts.append(build_inference_prompt(s['schema'], s['question'], tokenizer))

    params = SamplingParams(max_tokens=256, temperature=0)

    # Warmup
    llm.generate(prompts[:2], params)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    total_time = time.perf_counter() - t0

    vllm_result = {
        'backend': 'vLLM (offline batch)',
        'n_requests': len(prompts),
        'total_time_s': total_time,
        'throughput_rps': len(prompts) / total_time,
        'mean_ms': total_time / len(prompts) * 1000,
        'gpu_mem_gb': gpu_mem_vllm,
    }
    print(f'vLLM: total={total_time:.1f}s, rps={vllm_result[\"throughput_rps\"]:.2f}, '
          f'mean={vllm_result[\"mean_ms\"]:.0f}ms/req')

    del llm
    torch.cuda.empty_cache(); gc.collect()
    print('vLLM model unloaded')

except ImportError:
    print('vLLM not installed, skipping')
    vllm_result = {'backend': 'vLLM', 'error': 'not installed'}
except Exception as e:
    print(f'vLLM failed: {e}')
    vllm_result = {'backend': 'vLLM', 'error': str(e)}

# ---- Save results ----
report = {'hf': hf_result, 'vllm': vllm_result}
Path('${REPORT}').parent.mkdir(parents=True, exist_ok=True)
Path('${REPORT}').write_text(json.dumps(report, indent=2))

print(f'\n=== Summary ===')
print(f'HF:   mean={hf_result[\"mean_ms\"]:.0f}ms, p50={hf_result[\"p50_ms\"]:.0f}ms, p95={hf_result[\"p95_ms\"]:.0f}ms, rps={hf_result[\"throughput_rps\"]:.2f}')
if 'throughput_rps' in vllm_result:
    speedup = vllm_result['throughput_rps'] / hf_result['throughput_rps']
    print(f'vLLM: mean={vllm_result[\"mean_ms\"]:.0f}ms, rps={vllm_result[\"throughput_rps\"]:.2f}')
    print(f'Speedup: {speedup:.1f}x')
print(f'\nReport saved to ${REPORT}')
"
