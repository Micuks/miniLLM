"""Knowledge distillation training entry point.

Train a student model (Qwen2.5-7B) using teacher outputs via:
L = alpha * L_SFT + beta * L_KD_word + gamma * L_KD_seq
"""
from __future__ import annotations

import argparse
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from .distill.kd_trainer import KDTrainer, KDDataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    # Model
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--teacher-data", type=str, required=True, help="Path to teacher .jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/distill")
    # LoRA
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    # Training
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    # KD loss weights
    parser.add_argument("--alpha", type=float, default=1.0, help="SFT loss weight")
    parser.add_argument("--beta", type=float, default=0.5, help="KD word-level loss weight")
    parser.add_argument("--gamma", type=float, default=0.5, help="KD seq-level loss weight")
    parser.add_argument("--kd-temperature", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.info("Loading student model: %s", args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    logger.info("Loading KD dataset from: %s", args.teacher_data)
    dataset = KDDataset(args.teacher_data, tokenizer, max_length=args.max_length)
    logger.info("Dataset size: %d", len(dataset))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        max_steps=args.max_steps if args.max_steps else -1,
        remove_unused_columns=False,
    )

    trainer = KDTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        kd_temperature=args.kd_temperature,
    )

    logger.info("Starting KD training (alpha=%.2f, beta=%.2f, gamma=%.2f, T=%.1f)",
                args.alpha, args.beta, args.gamma, args.kd_temperature)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Distilled model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
