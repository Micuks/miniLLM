from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from .prompts import build_supervised_chat


@dataclass
class TrainArgs:
    model_name_or_path: str
    dataset_name: str
    output_dir: str
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    max_steps: int | None


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="b-mc2/sql-create-context")
    parser.add_argument("--output-dir", type=str, default="outputs/sft")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    return TrainArgs(
        model_name_or_path=args.model_name_or_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_steps=args.max_steps,
    )


def prepare_dataset(dataset_name: str, tokenizer, eval_holdout: int = 500) -> List[str]:
    dataset = load_dataset(dataset_name, split="train")
    # Hold out the first `eval_holdout` examples for evaluation to prevent data leakage
    if eval_holdout > 0 and eval_holdout < len(dataset):
        dataset = dataset.select(range(eval_holdout, len(dataset)))
    prompts: List[str] = [build_supervised_chat(sample, tokenizer) for sample in dataset]
    return prompts


def main() -> None:
    cfg = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Defer torch import to avoid making it a hard dependency for non-training usage
    import torch

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    sft_cfg = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        packing=False,
        max_steps=cfg.max_steps,
    )

    train_texts = prepare_dataset(cfg.dataset_name, tokenizer)
    train_ds = Dataset.from_dict({"text": train_texts})

    def formatting_func(example: Dict[str, str]) -> str:
        return example["text"]

    # TRL API compatibility: some versions use `processing_class` instead of `tokenizer`,
    # and may not accept `dataset_text_field`. Use `formatting_func` for robustness.
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            peft_config=lora_cfg,
            args=sft_cfg,
            formatting_func=formatting_func,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'tokenizer'" in str(exc):
            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=train_ds,
                peft_config=lora_cfg,
                args=sft_cfg,
                formatting_func=formatting_func,
            )
        else:
            raise

    trainer.train()
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    main()


