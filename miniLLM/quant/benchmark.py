"""Benchmark quantized models: size, memory, latency, throughput, quality."""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from miniLLM.prompts import build_inference_prompt
from miniLLM.sql_eval import score_text2sql

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    variant: str
    model_size_mb: float
    gpu_memory_mb: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_tok_per_s: float
    exact_match: float
    execution_match: float | None


def _get_model_size_mb(path: str) -> float:
    """Estimate model size from safetensors/bin files on disk."""
    total = 0
    p = Path(path)
    if p.is_dir():
        for f in p.rglob("*"):
            if f.suffix in (".safetensors", ".bin", ".pt"):
                total += f.stat().st_size
    return total / (1024 * 1024)


def _get_gpu_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def _load_model(model_path: str, variant: str):
    """Load model based on variant type."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if variant == "qlora-nf4":
        from transformers import BitsAndBytesConfig

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=quant_cfg, device_map="auto", trust_remote_code=True
        )
    elif variant == "gptq":
        from auto_gptq import AutoGPTQForCausalLM

        model = AutoGPTQForCausalLM.from_quantized(
            model_path, device_map="auto", trust_remote_code=True
        )
    elif variant == "awq":
        from awq import AutoAWQForCausalLM

        model = AutoAWQForCausalLM.from_quantized(
            model_path, fuse_layers=False, trust_remote_code=True
        )
    else:  # fp16
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )

    return tokenizer, model


def benchmark_variant(
    model_path: str,
    variant: str,
    dataset_name: str = "b-mc2/sql-create-context",
    n_samples: int = 100,
    with_execution: bool = True,
) -> BenchmarkResult:
    """Run a full benchmark for a single quantization variant."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer, model = _load_model(model_path, variant)
    model.eval()

    gpu_mem = _get_gpu_memory_mb()
    model_size = _get_model_size_mb(model_path)

    ds = load_dataset(dataset_name, split="train").select(range(n_samples))

    latencies: list[float] = []
    total_tokens = 0
    exact_hits = 0
    exec_hits = 0
    exec_total = 0

    for sample in ds:
        schema = sample.get("context", "")
        question = sample.get("question", "")
        gold = sample.get("answer", "")

        prompt = build_inference_prompt(schema, question, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        new_tokens = output_ids.shape[1] - input_len
        total_tokens += new_tokens
        latencies.append((t1 - t0) * 1000)

        pred = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        result = score_text2sql(schema, pred, gold, with_execution=with_execution)
        exact_hits += int(result.exact_match)
        if result.execution_match is not None:
            exec_total += 1
            exec_hits += int(result.execution_match)

    latencies.sort()
    n = len(latencies)
    total_time_s = sum(latencies) / 1000

    return BenchmarkResult(
        variant=variant,
        model_size_mb=model_size,
        gpu_memory_mb=gpu_mem,
        latency_mean_ms=sum(latencies) / n,
        latency_p50_ms=latencies[n // 2],
        latency_p95_ms=latencies[int(n * 0.95)],
        latency_p99_ms=latencies[int(n * 0.99)],
        throughput_tok_per_s=total_tokens / total_time_s if total_time_s > 0 else 0,
        exact_match=exact_hits / max(n, 1),
        execution_match=(exec_hits / exec_total) if exec_total else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark quantized model variants")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model or model ID")
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["fp16", "gptq", "awq", "qlora-nf4"],
    )
    parser.add_argument("--dataset-name", type=str, default="b-mc2/sql-create-context")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = benchmark_variant(
        args.model_path, args.variant, args.dataset_name, args.n_samples
    )
    print(json.dumps(asdict(result), indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
