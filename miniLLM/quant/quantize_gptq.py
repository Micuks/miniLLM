"""Quantize a model to GPTQ 4-bit using auto_gptq."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def load_calibration_data(
    tokenizer,
    dataset_name: str = "b-mc2/sql-create-context",
    n_samples: int = 128,
    max_length: int = 512,
) -> list:
    """Load and tokenize calibration samples for GPTQ quantization."""
    from miniLLM.prompts import build_supervised_chat

    ds = load_dataset(dataset_name, split="train").select(range(n_samples))
    texts = [build_supervised_chat(sample, tokenizer) for sample in ds]
    return [
        tokenizer(t, return_tensors="pt", max_length=max_length, truncation=True)["input_ids"]
        for t in texts
    ]


def quantize_gptq(
    model_name_or_path: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    n_calibration: int = 128,
    dataset_name: str = "b-mc2/sql-create-context",
) -> Path:
    """Quantize a model with GPTQ and save to output_dir."""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
    )

    logger.info("Loading model for GPTQ quantization: %s", model_name_or_path)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name_or_path,
        quantize_config=quantize_config,
        trust_remote_code=True,
    )

    calibration_data = load_calibration_data(
        tokenizer, dataset_name=dataset_name, n_samples=n_calibration
    )
    logger.info("Quantizing with %d calibration samples...", len(calibration_data))
    model.quantize(calibration_data)

    model.save_quantized(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("GPTQ model saved to %s", out)
    return out


def main():
    parser = argparse.ArgumentParser(description="GPTQ quantization")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--n-calibration", type=int, default=128)
    parser.add_argument("--dataset-name", type=str, default="b-mc2/sql-create-context")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    quantize_gptq(
        args.model_name_or_path,
        args.output_dir,
        bits=args.bits,
        group_size=args.group_size,
        n_calibration=args.n_calibration,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
