"""Knowledge distillation loss functions.

L = alpha * L_SFT + beta * L_KD_word + gamma * L_KD_seq

- L_SFT: cross-entropy vs gold labels
- L_KD_seq: cross-entropy vs teacher-generated sequences
- L_KD_word: KL divergence on sparse top-K logprobs with temperature scaling
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_word_loss(
    student_logits: torch.Tensor,
    teacher_top_k_logprobs: list[dict[int, float]],
    temperature: float = 4.0,
) -> torch.Tensor:
    """Word-level KD loss: KL divergence on sparse top-K teacher logprobs.

    Args:
        student_logits: (seq_len, vocab_size) - student model logits for generated tokens
        teacher_top_k_logprobs: list of dicts mapping token_id -> logprob (one per position)
        temperature: temperature scaling factor
    """
    device = student_logits.device
    total_loss = torch.tensor(0.0, device=device)
    n_positions = 0

    for pos, teacher_topk in enumerate(teacher_top_k_logprobs):
        if pos >= student_logits.shape[0] or not teacher_topk:
            continue

        token_ids = list(teacher_topk.keys())
        teacher_logprobs_vals = torch.tensor(
            [teacher_topk[tid] for tid in token_ids], device=device
        )

        # Temperature-scaled teacher probs (re-normalize over top-K)
        teacher_probs = F.softmax(teacher_logprobs_vals / temperature, dim=-1)

        # Student log-probs for the same tokens
        student_log_probs = F.log_softmax(student_logits[pos] / temperature, dim=-1)
        student_selected = student_log_probs[torch.tensor(token_ids, device=device)]

        # KL(teacher || student) on the sparse top-K support
        kl = F.kl_div(student_selected, teacher_probs, reduction="batchmean", log_target=False)
        total_loss = total_loss + kl * (temperature ** 2)
        n_positions += 1

    if n_positions > 0:
        total_loss = total_loss / n_positions

    return total_loss


def kd_seq_loss(
    student_logits: torch.Tensor,
    teacher_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Sequence-level KD loss: cross-entropy vs teacher-generated sequence.

    Args:
        student_logits: (seq_len, vocab_size) - student logits
        teacher_token_ids: (seq_len,) - teacher-generated token IDs
    """
    seq_len = min(student_logits.shape[0], teacher_token_ids.shape[0])
    return F.cross_entropy(
        student_logits[:seq_len],
        teacher_token_ids[:seq_len],
    )


def combined_kd_loss(
    student_logits: torch.Tensor,
    gold_labels: torch.Tensor,
    teacher_token_ids: torch.Tensor | None,
    teacher_top_k_logprobs: list[dict[int, float]] | None,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.5,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Combined KD loss: alpha*SFT + beta*KD_word + gamma*KD_seq.

    Args:
        student_logits: (seq_len, vocab_size) raw logits
        gold_labels: (seq_len,) gold token IDs
        teacher_token_ids: (seq_len,) teacher-generated token IDs (for L_KD_seq)
        teacher_top_k_logprobs: sparse teacher logprobs (for L_KD_word)
        alpha, beta, gamma: loss weights
        temperature: KD temperature
    """
    # L_SFT: cross-entropy vs gold
    seq_len_gold = min(student_logits.shape[0], gold_labels.shape[0])
    l_sft = F.cross_entropy(student_logits[:seq_len_gold], gold_labels[:seq_len_gold])

    total = alpha * l_sft

    # L_KD_word
    if teacher_top_k_logprobs and beta > 0:
        l_kd_word = kd_word_loss(student_logits, teacher_top_k_logprobs, temperature)
        total = total + beta * l_kd_word

    # L_KD_seq
    if teacher_token_ids is not None and gamma > 0:
        l_kd_seq = kd_seq_loss(student_logits, teacher_token_ids)
        total = total + gamma * l_kd_seq

    return total
