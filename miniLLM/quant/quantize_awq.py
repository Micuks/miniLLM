"""Quantize a model to AWQ 4-bit using autoawq."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def load_calibration_texts(
    tokenizer,
    dataset_name: str = "b-mc2/sql-create-context",
    n_samples: int = 128,
) -> list[str]:
    """Load calibration texts for AWQ quantization."""
    from miniLLM.prompts import build_supervised_chat

    ds = load_dataset(dataset_name, split="train").select(range(n_samples))
    return [build_supervised_chat(sample, tokenizer) for sample in ds]


def quantize_awq(
    model_name_or_path: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    n_calibration: int = 128,
    dataset_name: str = "b-mc2/sql-create-context",
) -> Path:
    """Quantize a model with AWQ and save to output_dir."""
    from awq import AutoAWQForCausalLM

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model for AWQ quantization: %s", model_name_or_path)
    model = AutoAWQForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    calib_texts = load_calibration_texts(
        tokenizer, dataset_name=dataset_name, n_samples=n_calibration
    )

    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",
    }

    logger.info("Quantizing with AWQ (%d calibration samples)...", len(calib_texts))
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)

    model.save_quantized(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("AWQ model saved to %s", out)
    return out


def main():
    parser = argparse.ArgumentParser(description="AWQ quantization")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--n-calibration", type=int, default=128)
    parser.add_argument("--dataset-name", type=str, default="b-mc2/sql-create-context")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    quantize_awq(
        args.model_name_or_path,
        args.output_dir,
        bits=args.bits,
        group_size=args.group_size,
        n_calibration=args.n_calibration,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
