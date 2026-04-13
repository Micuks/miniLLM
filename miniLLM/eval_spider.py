"""Spider benchmark evaluation: EM and Execution Accuracy by difficulty level."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .data.spider import load_spider, SpiderExample
from .prompts import build_inference_prompt
from .sql_eval import normalize_sql, execution_match, execution_match_from_db


def parse_args():
    parser = argparse.ArgumentParser(description="Spider benchmark evaluation")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (None = all)")
    parser.add_argument(
        "--spider-db-dir", type=str, default=None,
        help="Path to Spider database directory for execution accuracy"
    )
    parser.add_argument(
        "--backend", type=str, choices=["hf", "vllm"], default="hf",
        help="Inference backend"
    )
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8001")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantization to save VRAM")
    parser.add_argument("--report-path", type=str, default="outputs/spider_eval_report.json")
    return parser.parse_args()


def _generate_hf(model, tokenizer, schema: str, question: str) -> str:
    """Generate SQL using HuggingFace model."""
    prompt = build_inference_prompt(schema, question, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def _generate_vllm(vllm_url: str, schema: str, question: str) -> str:
    """Generate SQL using vLLM endpoint."""
    import requests

    resp = requests.post(
        f"{vllm_url}/generate_sql",
        json={"schema_ddl": schema, "question": question},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["sql"]


def main() -> None:
    args = parse_args()

    examples = load_spider(split="validation", max_samples=args.max_samples)
    print(f"Loaded {len(examples)} Spider validation examples")

    # Set up model for HF backend
    model = None
    tokenizer = None
    if args.backend == "hf":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kwargs = dict(device_map="auto", trust_remote_code=True)
        if args.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, **load_kwargs
        )
        model = PeftModel.from_pretrained(base_model, args.adapter_path) if args.adapter_path else base_model
        model.eval()

    # Metrics accumulators
    by_difficulty: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "em": 0,
            "ex": 0,
            "ex_attempted": 0,
            "ex_failed": 0,
            "ex_unknown": 0,
        }
    )
    records: list[dict] = []

    for idx, ex in enumerate(examples):
        if args.backend == "hf":
            pred_sql = _generate_hf(model, tokenizer, ex.schema_ddl, ex.question)
        else:
            pred_sql = _generate_vllm(args.vllm_url, ex.schema_ddl, ex.question)

        # Exact match
        em = normalize_sql(pred_sql) == normalize_sql(ex.gold_sql)

        # Execution match
        ex_match = None
        ex_error = None
        if args.spider_db_dir:
            db_path = str(Path(args.spider_db_dir) / ex.db_id / f"{ex.db_id}.sqlite")
            if Path(db_path).exists():
                ex_match, ex_error = execution_match_from_db(db_path, pred_sql, ex.gold_sql)
        else:
            # Fall back to in-memory execution with reconstructed DDL
            ex_match, ex_error = execution_match(ex.schema_ddl, pred_sql, ex.gold_sql)

        diff = ex.difficulty
        by_difficulty[diff]["total"] += 1
        by_difficulty[diff]["em"] += int(em)
        if ex_match is not None:
            by_difficulty[diff]["ex_attempted"] += 1
            by_difficulty[diff]["ex"] += int(ex_match)
            by_difficulty[diff]["ex_failed"] += int(not ex_match)
        else:
            by_difficulty[diff]["ex_unknown"] += 1

        records.append({
            "index": idx,
            "db_id": ex.db_id,
            "difficulty": diff,
            "question": ex.question,
            "gold_sql": ex.gold_sql,
            "pred_sql": pred_sql,
            "exact_match": em,
            "execution_match": ex_match,
            "error": ex_error,
        })

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(examples)}")

    # Compute overall metrics
    total = len(records)
    total_em = sum(1 for r in records if r["exact_match"])
    total_ex = sum(1 for r in records if r["execution_match"] is True)
    total_ex_attempted = sum(1 for r in records if r["execution_match"] is not None)
    total_ex_failed = sum(1 for r in records if r["execution_match"] is False)
    total_ex_unknown = sum(1 for r in records if r["execution_match"] is None)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "backend": args.backend,
        "total_examples": total,
        "exact_match": total_em / max(total, 1),
        "execution_accuracy": total_ex / max(total_ex_attempted, 1) if total_ex_attempted else None,
        "execution_accuracy_all": total_ex / max(total, 1),
        "execution_counts": {
            "match": total_ex,
            "mismatch": total_ex_failed,
            "unknown": total_ex_unknown,
            "attempted": total_ex_attempted,
        },
        "by_difficulty": {},
    }

    for diff, counts in sorted(by_difficulty.items()):
        summary["by_difficulty"][diff] = {
            "total": counts["total"],
            "exact_match": counts["em"] / max(counts["total"], 1),
            "execution_accuracy": (
                counts["ex"] / max(counts["ex_attempted"], 1)
                if counts["ex_attempted"] else None
            ),
            "execution_accuracy_all": counts["ex"] / max(counts["total"], 1),
            "execution_counts": {
                "match": counts["ex"],
                "mismatch": counts["ex_failed"],
                "unknown": counts["ex_unknown"],
                "attempted": counts["ex_attempted"],
            },
        }

    # Output
    out_path = Path(args.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2))

    print("\n=== Spider Evaluation Summary ===")
    print(f"Total: {total}")
    print(f"Exact Match: {summary['exact_match']:.4f}")
    if summary["execution_accuracy"] is not None:
        print(
            "Execution Accuracy:"
            f" attempted={summary['execution_accuracy']:.4f}"
            f" all={summary['execution_accuracy_all']:.4f}"
            f" attempted_count={total_ex_attempted}/{total}"
        )
    print("\nBy Difficulty:")
    for diff, stats in summary["by_difficulty"].items():
        ex_str = (
            ","
            f" EX_attempted={stats['execution_accuracy']:.4f}"
            f" EX_all={stats['execution_accuracy_all']:.4f}"
            f" attempted={stats['execution_counts']['attempted']}/{stats['total']}"
            if stats["execution_accuracy"] is not None
            else f", EX_all={stats['execution_accuracy_all']:.4f} attempted=0/{stats['total']}"
        )
        print(f"  {diff}: n={stats['total']}, EM={stats['exact_match']:.4f}{ex_str}")
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
