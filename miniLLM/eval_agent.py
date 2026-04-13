"""Evaluate the Agent model in ReAct mode on Spider benchmark.

Supports both sql-create-context (simple) and Spider (hard, with difficulty breakdown).
Runs the model generating ReAct trajectories, optionally with interactive SQL execution.

Usage:
    # Spider evaluation (recommended — shows per-difficulty breakdown)
    python -m miniLLM.eval_agent \
        --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
        --adapter-path outputs/grpo-agent \
        --dataset spider --with-execution --interactive

    # sql-create-context evaluation
    python -m miniLLM.eval_agent \
        --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
        --dataset sql-create-context --with-execution
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .agent.env import SQLExecutionEnv
from .agent.react import (
    build_react_inference_prompt,
    extract_final_sql,
    parse_trajectory,
)
from .sql_eval import normalize_sql, execution_match, execution_match_from_db


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="spider",
                        choices=["spider", "sql-create-context"],
                        help="Evaluation dataset")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit samples (None = all for spider, 50 for sql-create-context)")
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--with-execution", action="store_true")
    parser.add_argument("--interactive", action="store_true",
                        help="Execute SQL actions mid-generation and inject real observations")
    parser.add_argument("--max-turns", type=int, default=5,
                        help="Max ReAct turns in interactive mode")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--report-path", type=str, default=None)
    return parser.parse_args()


def _load_eval_samples(args) -> list[dict]:
    """Load evaluation samples with schema, question, gold_sql, difficulty, db_path."""
    if args.dataset == "spider":
        from .data.spider import load_spider
        examples = load_spider(split="validation", max_samples=args.num_samples)
        return [
            {"schema": ex.schema_ddl, "question": ex.question,
             "gold_sql": ex.gold_sql, "difficulty": ex.difficulty,
             "db_id": ex.db_id, "db_path": ex.db_path}
            for ex in examples
        ]
    else:
        ds = load_dataset("b-mc2/sql-create-context", split="train")
        n = args.num_samples or 50
        end = min(args.eval_offset + n, len(ds))
        ds = ds.select(range(args.eval_offset, end))
        return [
            {"schema": s.get("context", ""), "question": s.get("question", ""),
             "gold_sql": s.get("answer", ""), "difficulty": "unknown", "db_id": ""}
            for s in ds
        ]


def generate_single_pass(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate a complete ReAct trajectory in one pass."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def generate_interactive(
    model, tokenizer, prompt: str, env: SQLExecutionEnv, max_turns: int = 5
) -> str:
    """Generate with real environment interaction.

    After each Action, pause, execute SQL, inject real Observation, continue.
    """
    full_text = prompt
    trajectory_parts: list[str] = []

    for _ in range(max_turns):
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            stop_strings=["Observation:"],
            tokenizer=tokenizer,
        )
        new_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        trajectory_parts.append(new_text)

        if "Answer:" in new_text:
            break

        action_m = re.search(r'Action:\s*execute_sql\["(.+?)"\]', new_text, re.DOTALL)
        if action_m:
            sql = action_m.group(1)
            result = env.execute(sql)
            observation = f"Observation: {result.format_observation(max_rows=10)}"
            trajectory_parts.append(observation)
            full_text = full_text + new_text + "\n" + observation + "\n"
        else:
            break

    return "\n".join(trajectory_parts)


def main() -> None:
    args = parse_args()

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
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    samples = _load_eval_samples(args)
    print(f"Loaded {len(samples)} evaluation samples ({args.dataset})")

    # Per-difficulty accumulators
    by_diff: dict[str, dict] = defaultdict(
        lambda: {
            "total": 0,
            "em": 0,
            "ex": 0,
            "ex_attempted": 0,
            "ex_failed": 0,
            "ex_unknown": 0,
            "turns": [],
        }
    )
    records: list[dict] = []

    for idx, sample in enumerate(samples):
        schema = sample["schema"]
        question = sample["question"]
        gold = sample["gold_sql"]
        diff = sample["difficulty"]

        prompt = build_react_inference_prompt(schema, question, tokenizer)

        if args.interactive:
            try:
                db_path = sample.get("db_path")
                if db_path:
                    # Use real Spider database for interactive execution
                    from .agent.env import SQLExecutionEnvFromDB
                    env = SQLExecutionEnvFromDB(db_path)
                else:
                    env = SQLExecutionEnv(schema)
                trajectory = generate_interactive(
                    model, tokenizer, prompt, env, max_turns=args.max_turns
                )
                env.close()
            except (ValueError, Exception):
                trajectory = generate_single_pass(model, tokenizer, prompt)
        else:
            trajectory = generate_single_pass(model, tokenizer, prompt)

        pred_sql = extract_final_sql(trajectory)
        parsed = parse_trajectory(trajectory)
        n_turns = len([s for s in parsed.steps if s.action_sql])

        em = False
        ex_match = None
        error = None
        if pred_sql:
            em = normalize_sql(pred_sql) == normalize_sql(gold)
            if args.with_execution:
                db_path = sample.get("db_path")
                if db_path:
                    ex_match, error = execution_match_from_db(db_path, pred_sql, gold)
                else:
                    ex_match, error = execution_match(schema, pred_sql, gold)

        by_diff[diff]["total"] += 1
        by_diff[diff]["em"] += int(em)
        by_diff[diff]["turns"].append(n_turns)
        if ex_match is not None:
            by_diff[diff]["ex_attempted"] += 1
            by_diff[diff]["ex"] += int(ex_match)
            by_diff[diff]["ex_failed"] += int(not ex_match)
        else:
            by_diff[diff]["ex_unknown"] += 1

        records.append({
            "index": idx,
            "db_id": sample.get("db_id", ""),
            "difficulty": diff,
            "question": question,
            "trajectory": trajectory,
            "pred_sql": pred_sql or "",
            "gold_sql": gold,
            "exact_match": em,
            "execution_match": ex_match,
            "num_turns": n_turns,
            "error": error,
        })

        status = f"EM={'Y' if em else 'N'} EX={'Y' if ex_match else ('N' if ex_match is False else '?')}"
        print(f"[{idx+1}/{len(samples)}] ({diff}) {status} turns={n_turns}")

    # Summary
    total = len(records)
    total_em = sum(r["exact_match"] for r in records)
    total_ex = sum(1 for r in records if r["execution_match"] is True)
    total_ex_att = sum(1 for r in records if r["execution_match"] is not None)
    total_ex_failed = sum(1 for r in records if r["execution_match"] is False)
    total_ex_unknown = sum(1 for r in records if r["execution_match"] is None)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "dataset": args.dataset,
        "mode": "interactive" if args.interactive else "single_pass",
        "num_samples": total,
        "exact_match": total_em / max(total, 1),
        "execution_match": total_ex / max(total_ex_att, 1) if total_ex_att else None,
        "execution_match_all": total_ex / max(total, 1),
        "execution_counts": {
            "match": total_ex,
            "mismatch": total_ex_failed,
            "unknown": total_ex_unknown,
            "attempted": total_ex_att,
        },
        "by_difficulty": {},
    }

    print("\n=== Agent Eval Summary ===")
    print(f"Overall: EM={summary['exact_match']:.4f}", end="")
    if summary["execution_match"] is not None:
        print(
            f" EX_attempted={summary['execution_match']:.4f}"
            f" EX_all={summary['execution_match_all']:.4f}"
            f" attempted={total_ex_att}/{total}",
            end="",
        )
    print()

    for diff in ["easy", "medium", "hard", "unknown"]:
        if diff not in by_diff:
            continue
        d = by_diff[diff]
        em_rate = d["em"] / max(d["total"], 1)
        ex_rate = d["ex"] / max(d["ex_attempted"], 1) if d["ex_attempted"] else None
        ex_all = d["ex"] / max(d["total"], 1)
        avg_turns = sum(d["turns"]) / max(len(d["turns"]), 1)
        summary["by_difficulty"][diff] = {
            "total": d["total"], "exact_match": em_rate,
            "execution_match": ex_rate,
            "execution_match_all": ex_all,
            "execution_counts": {
                "match": d["ex"],
                "mismatch": d["ex_failed"],
                "unknown": d["ex_unknown"],
                "attempted": d["ex_attempted"],
            },
            "avg_turns": avg_turns,
        }
        ex_str = (
            f" EX_attempted={ex_rate:.4f} EX_all={ex_all:.4f}"
            f" attempted={d['ex_attempted']}/{d['total']}"
            if ex_rate is not None
            else f" EX_all={ex_all:.4f} attempted=0/{d['total']}"
        )
        print(f"  {diff:8s}: n={d['total']:3d} EM={em_rate:.4f}{ex_str} avg_turns={avg_turns:.1f}")

    # Save report
    report_path = args.report_path or f"outputs/eval_agent_{args.dataset}.json"
    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
