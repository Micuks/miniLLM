"""Collect teacher outputs and logprobs for offline knowledge distillation.

Supports two modes:
1. API mode: Use Together AI or OpenAI-compatible API (Qwen2.5-72B)
2. Local mode: Use a local GPTQ/AWQ quantized teacher model
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)


def collect_via_api(
    dataset_name: str,
    output_path: str,
    api_base: str,
    api_key: str,
    model: str = "Qwen/Qwen2.5-72B-Instruct",
    n_samples: int | None = None,
    top_logprobs: int = 20,
) -> None:
    """Collect teacher outputs via OpenAI-compatible API with logprobs."""
    import openai

    client = openai.OpenAI(base_url=api_base, api_key=api_key)

    from miniLLM.prompts import SYSTEM_PROMPT

    ds = load_dataset(dataset_name, split="train")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as fh:
        for idx, sample in enumerate(ds):
            schema = sample.get("context", "").strip()
            question = sample.get("question", "").strip()
            gold = sample.get("answer", "").strip()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"### Database Schema:\n{schema}\n\n### Question:\n{question}"},
            ]

            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=256,
                    logprobs=True,
                    top_logprobs=top_logprobs,
                )
                choice = resp.choices[0]
                teacher_text = choice.message.content.strip()

                # Extract top-K logprobs per token
                token_logprobs = []
                if choice.logprobs and choice.logprobs.content:
                    for token_info in choice.logprobs.content:
                        top_k = {}
                        if token_info.top_logprobs:
                            for lp in token_info.top_logprobs:
                                top_k[lp.token] = lp.logprob
                        token_logprobs.append({
                            "token": token_info.token,
                            "logprob": token_info.logprob,
                            "top_logprobs": top_k,
                        })

                record = {
                    "index": idx,
                    "context": schema,
                    "question": question,
                    "gold_sql": gold,
                    "teacher_sql": teacher_text,
                    "teacher_logprobs": token_logprobs,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            except Exception as e:
                logger.warning("Failed on sample %d: %s", idx, e)
                continue

            if (idx + 1) % 100 == 0:
                logger.info("Collected %d/%d samples", idx + 1, len(ds))

    logger.info("Teacher outputs saved to %s", out)


def collect_via_local(
    dataset_name: str,
    output_path: str,
    model_name_or_path: str,
    n_samples: int | None = None,
    top_k: int = 20,
) -> None:
    """Collect teacher outputs from a local model with logprobs."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from miniLLM.prompts import build_inference_prompt

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", trust_remote_code=True
    )
    model.eval()

    ds = load_dataset(dataset_name, split="train")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as fh:
        for idx, sample in enumerate(ds):
            schema = sample.get("context", "").strip()
            question = sample.get("question", "").strip()
            gold = sample.get("answer", "").strip()

            prompt = build_inference_prompt(schema, question, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.0,
                    return_dict_in_generate=True,
                    output_scores=True,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs.sequences[0][input_len:]
            teacher_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Extract top-K logprobs from scores
            token_logprobs = []
            for step_idx, score in enumerate(outputs.scores):
                log_probs = torch.log_softmax(score[0], dim=-1)
                top_vals, top_ids = torch.topk(log_probs, k=min(top_k, log_probs.shape[-1]))
                top_k_dict = {}
                for v, tid in zip(top_vals.tolist(), top_ids.tolist()):
                    token_str = tokenizer.decode([tid])
                    top_k_dict[token_str] = v

                token_logprobs.append({
                    "token": tokenizer.decode([generated_ids[step_idx].item()]),
                    "logprob": log_probs[generated_ids[step_idx].item()].item(),
                    "top_logprobs": top_k_dict,
                })

            record = {
                "index": idx,
                "context": schema,
                "question": question,
                "gold_sql": gold,
                "teacher_sql": teacher_text,
                "teacher_logprobs": token_logprobs,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % 50 == 0:
                logger.info("Collected %d/%d samples", idx + 1, len(ds))

    logger.info("Teacher outputs saved to %s", out)


def main():
    parser = argparse.ArgumentParser(description="Collect teacher outputs for KD")
    parser.add_argument("--mode", type=str, choices=["api", "local"], required=True)
    parser.add_argument("--dataset-name", type=str, default="b-mc2/sql-create-context")
    parser.add_argument("--output-path", type=str, default="outputs/teacher_outputs.jsonl")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    # API mode args
    parser.add_argument("--api-base", type=str, default="https://api.together.xyz/v1")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    # Local mode args
    parser.add_argument("--model-name-or-path", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.mode == "api":
        collect_via_api(
            args.dataset_name,
            args.output_path,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.teacher_model,
            n_samples=args.n_samples,
            top_logprobs=args.top_k,
        )
    else:
        if not args.model_name_or_path:
            parser.error("--model-name-or-path required for local mode")
        collect_via_local(
            args.dataset_name,
            args.output_path,
            model_name_or_path=args.model_name_or_path,
            n_samples=args.n_samples,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
