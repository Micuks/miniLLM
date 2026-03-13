from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import build_inference_prompt
from .sql_eval import score_text2sql


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="b-mc2/sql-create-context")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--with-execution", action="store_true")
    parser.add_argument("--report-path", type=str, default="outputs/eval_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path) if args.adapter_path else base_model

    ds = load_dataset(args.dataset_name, split="train").select(range(args.num_samples))

    exact_hits = 0
    exec_hits = 0
    exec_total = 0
    records: list[dict] = []

    for idx, sample in enumerate(ds):
        schema = sample.get("context", "")
        question = sample.get("question", "")
        gold = sample.get("answer", "")
        prompt = build_inference_prompt(schema, question)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        pred = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

        result = score_text2sql(schema, pred, gold, with_execution=args.with_execution)
        exact_hits += int(result.exact_match)
        if result.execution_match is not None:
            exec_total += 1
            exec_hits += int(result.execution_match)

        records.append(
            {
                "index": idx,
                "question": question,
                "pred_sql": result.pred_sql,
                "gold_sql": result.gold_sql,
                "exact_match": result.exact_match,
                "execution_match": result.execution_match,
                "error": result.error,
            }
        )

        print("=== Question ===")
        print(question)
        print("=== Pred SQL ===")
        print(result.pred_sql)
        print("=== Gold SQL ===")
        print(result.gold_sql)
        print(f"=== Exact Match === {result.exact_match}")
        if args.with_execution:
            print(f"=== Execution Match === {result.execution_match}")
            if result.error:
                print(f"=== Execution Error === {result.error}")
        print()

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "dataset_name": args.dataset_name,
        "num_samples": args.num_samples,
        "exact_match": exact_hits / max(args.num_samples, 1),
        "execution_match": (exec_hits / exec_total) if exec_total else None,
        "execution_coverage": exec_total / max(args.num_samples, 1),
    }

    out_path = Path(args.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2))

    print("=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
