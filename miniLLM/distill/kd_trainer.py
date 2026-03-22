"""Custom Trainer subclass for knowledge distillation with QLoRA."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments, AutoTokenizer

from .kd_loss import combined_kd_loss


class KDDataset(TorchDataset):
    """Dataset that loads teacher outputs from a .jsonl file alongside tokenized inputs."""

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path) as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        from miniLLM.prompts import _build_messages

        # Build messages for gold answer
        messages = _build_messages(
            schema=sample["context"],
            question=sample["question"],
            answer=sample["gold_sql"],
        )
        gold_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        gold_enc = self.tokenizer(
            gold_text, max_length=self.max_length, truncation=True, return_tensors="pt"
        )

        # Tokenize teacher sequence
        teacher_sql = sample.get("teacher_sql", "")
        teacher_enc = self.tokenizer(
            teacher_sql, max_length=self.max_length, truncation=True, return_tensors="pt"
        )

        # Parse sparse teacher logprobs: convert string token keys to token IDs
        teacher_logprobs_raw = sample.get("teacher_logprobs", [])
        teacher_top_k = []
        for pos_info in teacher_logprobs_raw:
            top_k_dict = {}
            for token_str, logprob in pos_info.get("top_logprobs", {}).items():
                tid = self.tokenizer.convert_tokens_to_ids(token_str)
                if tid != self.tokenizer.unk_token_id:
                    top_k_dict[tid] = logprob
            teacher_top_k.append(top_k_dict)

        return {
            "input_ids": gold_enc["input_ids"].squeeze(0),
            "attention_mask": gold_enc["attention_mask"].squeeze(0),
            "labels": gold_enc["input_ids"].squeeze(0).clone(),
            "teacher_token_ids": teacher_enc["input_ids"].squeeze(0),
            "teacher_top_k_logprobs": teacher_top_k,
        }


def kd_collate_fn(features: list[dict]) -> dict:
    """Custom collate that handles teacher_top_k_logprobs (list of dicts)."""
    import torch

    batch = {}
    for key in ("input_ids", "attention_mask", "labels", "teacher_token_ids"):
        if key in features[0]:
            batch[key] = torch.stack([f[key] for f in features])

    # Keep teacher_top_k_logprobs as a list (not tensorizable)
    if "teacher_top_k_logprobs" in features[0]:
        batch["teacher_top_k_logprobs"] = [f["teacher_top_k_logprobs"] for f in features]

    return batch


class KDTrainer(Trainer):
    """Trainer with custom KD loss: alpha*SFT + beta*KD_word + gamma*KD_seq."""

    def __init__(
        self,
        *args,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        kd_temperature: float = 4.0,
        **kwargs,
    ):
        # Force our custom collate
        kwargs.setdefault("data_collator", kd_collate_fn)
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kd_temperature = kd_temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        teacher_token_ids = inputs.pop("teacher_token_ids", None)
        teacher_top_k_logprobs = inputs.pop("teacher_top_k_logprobs", None)

        outputs = model(**inputs)
        logits = outputs.logits

        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # For batch size > 1, we process each sample individually
        batch_loss = torch.tensor(0.0, device=logits.device)
        batch_size = shift_logits.shape[0]

        for b in range(batch_size):
            b_logits = shift_logits[b]  # (seq_len, vocab_size)
            b_labels = shift_labels[b]  # (seq_len,)

            b_teacher_ids = teacher_token_ids[b] if teacher_token_ids is not None else None
            b_teacher_topk = teacher_top_k_logprobs[b] if teacher_top_k_logprobs is not None else None

            loss = combined_kd_loss(
                student_logits=b_logits,
                gold_labels=b_labels,
                teacher_token_ids=b_teacher_ids,
                teacher_top_k_logprobs=b_teacher_topk,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                temperature=self.kd_temperature,
            )
            batch_loss = batch_loss + loss

        batch_loss = batch_loss / batch_size

        return (batch_loss, outputs) if return_outputs else batch_loss
