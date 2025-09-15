from __future__ import annotations

import argparse
from typing import List

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from .prompts import build_inference_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="b-mc2/sql-create-context")
    parser.add_argument("--num-samples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
    )

    ds = load_dataset(args.dataset_name, split="train").select(range(args.num_samples))

    for sample in ds:
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
        text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print("=== Question ===")
        print(question)
        print("=== Pred SQL ===")
        print(text.strip())
        print("=== Gold SQL ===")
        print(gold.strip())
        print()


if __name__ == "__main__":
    main()



