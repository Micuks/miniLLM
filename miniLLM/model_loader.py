"""Unified loaders for base model + LoRA adapters, shared by SFT and GRPO.

Supports two quantization modes:
  - "none": bf16 base, LoRA trainable in bf16 (legacy behaviour).
  - "bnb4": 4-bit NF4 base via bitsandbytes, LoRA trainable in bf16 (QLoRA).

QLoRA path is designed for 3B / 24GB: the 4-bit base keeps actor + frozen
reference model resident in VRAM without merging adapters into quantized
weights (which cannot be done losslessly).
"""

from __future__ import annotations

import logging
import os
from typing import Literal

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM

log = logging.getLogger(__name__)

QuantMode = Literal["none", "bnb4"]


def _bnb4_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _default_device_map() -> dict:
    """Pin the whole model to the local rank's GPU.

    bitsandbytes quantizes weights during `from_pretrained`, so the target
    device must be declared up-front — 4-bit tensors cannot be cleanly moved
    CPU→GPU afterward. For DeepSpeed single-GPU-per-rank training this is
    the correct placement; `device_map="auto"` would delegate to accelerate
    and risk splitting layers or leaving shards on CPU.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {"": local_rank}


def load_base_model(
    model_name_or_path: str,
    *,
    quant_mode: QuantMode = "none",
    trust_remote_code: bool = True,
    device_map=None,
) -> AutoModelForCausalLM:
    """Load the backbone in bf16 or bnb 4-bit NF4.

    `device_map` only applies to the bnb4 path. Pass an explicit mapping to
    override; `None` routes the whole model to the local rank's GPU.
    """
    kwargs: dict = {"trust_remote_code": trust_remote_code}
    if quant_mode == "bnb4":
        kwargs["quantization_config"] = _bnb4_config()
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = device_map if device_map is not None else _default_device_map()
    else:
        kwargs["dtype"] = torch.bfloat16
    log.info(
        "Loading base model %s (quant=%s, device_map=%s)",
        model_name_or_path, quant_mode, kwargs.get("device_map"),
    )
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def _default_lora_cfg(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )


def _finalize_training_graph(model, *, gradient_checkpointing: bool):
    """Enable input-embedding grads on the PEFT-wrapped model.

    Called *after* adapter attachment so we hook whichever embedding module
    PeftModel exposes — if a future PEFT version swapped the embedding at
    wrap time this would still target the right module. For bnb4 the
    equivalent call already ran inside prepare_model_for_kbit_training, but
    repeating it on the wrapped model is a cheap no-op when the hook is
    already installed.
    """
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # Some PEFT versions expose this on the wrapper; if the base already
        # had checkpointing on this is idempotent.
        try:
            model.gradient_checkpointing_enable()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def attach_fresh_lora(
    model,
    *,
    quant_mode: QuantMode,
    gradient_checkpointing: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    """Attach a brand-new trainable LoRA to a base model.

    Used by SFT when no prior adapter is provided.
    """
    if quant_mode == "bnb4":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, _default_lora_cfg(lora_r, lora_alpha, lora_dropout))
    return _finalize_training_graph(model, gradient_checkpointing=gradient_checkpointing)


def load_adapter_for_training(
    model,
    adapter_path: str,
    *,
    quant_mode: QuantMode,
    gradient_checkpointing: bool = True,
):
    """Load an existing LoRA adapter onto `model` and keep it trainable.

    This is the QLoRA-friendly continuation path: we do NOT call
    merge_and_unload (which is lossy on 4-bit weights). Instead the SFT
    adapter remains the single trainable LoRA and GRPO continues updating it.
    """
    if quant_mode == "bnb4":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
    log.info("Attaching trainable adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    return _finalize_training_graph(model, gradient_checkpointing=gradient_checkpointing)


def load_adapter_frozen(
    model,
    adapter_path: str,
):
    """Load an adapter for reference (KL) use: frozen, eval mode, no grad."""
    log.info("Attaching frozen reference adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
