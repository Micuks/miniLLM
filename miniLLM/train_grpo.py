"""GRPO (Group Relative Policy Optimization) with native DeepSpeed.

Hand-written GRPO training loop:
  1. For each prompt, sample G completions from the current policy
  2. Score each completion with environment-based reward
  3. Normalize rewards within each group (advantage = (r - mean) / std)
  4. Compute policy gradient loss weighted by advantages
  5. Update via deepspeed engine.backward() / engine.step()

No TRL, no Trainer — full control over every step.

Usage:
    deepspeed --num_gpus=1 --module miniLLM.train_grpo \
        --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
        --adapter-path outputs/react-sft \
        --deepspeed configs/ds_zero2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import deepspeed
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

import re

from .agent.env import SQLExecutionEnv, SQLExecutionEnvFromDB
from .agent.react import build_react_inference_prompt, parse_trajectory
from .agent.reward import resolve_reward_weights, reward_breakdown
from .model_loader import (
    attach_fresh_lora,
    load_adapter_for_training,
    load_adapter_frozen,
    load_base_model,
)

_ACTION_SQL_RE = re.compile(r'Action:\s*execute_sql\["(.+?)"\]', re.DOTALL)


# ---------------------------------------------------------------------------
# v5 helpers: adaptive compute, span weighting, turn-delta credit
# ---------------------------------------------------------------------------

def _adaptive_max_turns(difficulty: str, base_max_turns: int) -> int:
    """Idea 4: fewer turns for easy, more for hard."""
    multipliers = {"easy": 0.4, "medium": 0.8, "hard": 1.4, "unknown": 1.0}
    return max(1, int(base_max_turns * multipliers.get(difficulty, 1.0)))


def _build_turn_weights(turn_info: list[dict], n_tokens: int, device: torch.device) -> torch.Tensor:
    """Idea 1: per-token weights based on turn contribution.

    Turns with successful SQL execution or Answer tags get higher weight,
    so the policy gradient pushes harder on high-value decisions.
    """
    weights = torch.ones(n_tokens, device=device)
    for turn in turn_info:
        if not turn.get("is_model", False):
            continue
        start, end = turn["start"], turn["end"]
        if turn.get("has_answer", False):
            w = 1.5
        elif turn.get("action_success") is True:
            w = 1.2
        elif turn.get("action_success") is False:
            w = 0.5
        else:
            w = 0.8
        weights[start:end] = w
    # Normalize so mean weight over model tokens ≈ 1.0
    if (weights > 0).any():
        weights = weights / weights.mean().clamp(min=0.1)
    return weights


def _entropy_weights(logits: torch.Tensor, gen_mask: torch.Tensor) -> torch.Tensor:
    """Idea 5: per-token entropy weighting.

    High-entropy tokens (uncertain decisions like table/column choice)
    get higher gradient weight than low-entropy boilerplate.
    """
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(-1)
    # Normalize to [0.5, 2.0] range over model-generated tokens
    mask = gen_mask.float()
    ent_on_model = entropy * mask
    ent_mean = ent_on_model.sum() / mask.sum().clamp(min=1.0)
    ent_w = 0.5 + 1.5 * (entropy / ent_mean.clamp(min=1e-8)).clamp(0, 2.0)
    return ent_w

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset: list of (schema, question, gold_sql, prompt_ids)
# ---------------------------------------------------------------------------

def prepare_prompts(
    source: str,
    tokenizer,
    num_samples: int = 500,
    max_prompt_len: int = 1024,
) -> list[dict]:
    """Prepare tokenized prompts + metadata for GRPO.

    Supports 'spider' (with real db_path) or 'sql-create-context'.
    """
    if source == "spider":
        from .data.spider import load_spider
        examples = load_spider(split="train", max_samples=num_samples)
        raw = [{"schema": ex.schema_ddl, "question": ex.question,
                "gold_sql": ex.gold_sql, "db_path": ex.db_path,
                "difficulty": ex.difficulty}
               for ex in examples]
    else:
        ds = load_dataset("b-mc2/sql-create-context", split="train")
        ds = ds.select(range(500, min(500 + num_samples, len(ds))))
        raw = [{"schema": s.get("context", ""), "question": s.get("question", ""),
                "gold_sql": s.get("answer", ""), "db_path": None,
                "difficulty": "unknown"}
               for s in ds]

    prompts = []
    for item in raw:
        prompt_text = build_react_inference_prompt(item["schema"], item["question"], tokenizer)
        encoded = tokenizer(
            prompt_text, max_length=max_prompt_len, truncation=True, return_tensors="pt"
        )
        prompts.append({
            "schema": item["schema"],
            "question": item["question"],
            "gold_sql": item["gold_sql"],
            "db_path": item["db_path"],
            "difficulty": item["difficulty"],
            "prompt_ids": encoded["input_ids"].squeeze(0),
            "prompt_mask": encoded["attention_mask"].squeeze(0),
        })
    return prompts


# ---------------------------------------------------------------------------
# GRPO core: sample, reward, policy gradient
# ---------------------------------------------------------------------------

def _make_rollout_env(schema: str, db_path: str | None):
    if db_path:
        return SQLExecutionEnvFromDB(db_path)
    return SQLExecutionEnv(schema)


@torch.no_grad()
def _rollout_one(
    engine, tokenizer, prompt_ids: torch.Tensor, schema: str, db_path: str | None,
    *, do_sample: bool, temperature: float, top_p: float,
    max_new_tokens_per_turn: int, max_turns: int,
) -> tuple[torch.Tensor, torch.Tensor, str, list[dict]]:
    """One interactive rollout. Returns (gen_ids, gen_mask, gen_text, turn_info).

    gen_mask: 1 for model-generated tokens, 0 for env-injected Observation tokens.
    turn_info: per-segment metadata for turn-delta credit assignment.
    """
    device = engine.device
    current_ids = prompt_ids.to(device).unsqueeze(0)
    gen_ids_parts: list[torch.Tensor] = []
    gen_mask_parts: list[torch.Tensor] = []
    text_parts: list[str] = []
    turn_info: list[dict] = []
    token_offset = 0

    env = _make_rollout_env(schema, db_path)
    try:
        for _turn in range(max_turns):
            kw = {
                "input_ids": current_ids,
                "max_new_tokens": max_new_tokens_per_turn,
                "do_sample": do_sample,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "stop_strings": ["Observation:"],
                "tokenizer": tokenizer,
            }
            if do_sample:
                kw["temperature"] = temperature
                kw["top_p"] = top_p
            output_ids = engine.module.generate(**kw)
            new_tokens = output_ids[0, current_ids.shape[1]:]
            if new_tokens.numel() == 0:
                break
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            n_new = new_tokens.numel()
            gen_ids_parts.append(new_tokens)
            gen_mask_parts.append(torch.ones_like(new_tokens))
            text_parts.append(new_text)

            has_answer = "Answer:" in new_text
            action_m = _ACTION_SQL_RE.search(new_text)
            action_success = None

            if has_answer:
                turn_info.append({"start": token_offset, "end": token_offset + n_new,
                                  "is_model": True, "action_success": action_success,
                                  "has_answer": True})
                token_offset += n_new
                break

            if not action_m:
                turn_info.append({"start": token_offset, "end": token_offset + n_new,
                                  "is_model": True, "action_success": None,
                                  "has_answer": False})
                token_offset += n_new
                break

            sql = action_m.group(1)
            result = env.execute(sql)
            action_success = result.success
            turn_info.append({"start": token_offset, "end": token_offset + n_new,
                              "is_model": True, "action_success": action_success,
                              "has_answer": False})
            token_offset += n_new

            obs_text = " " + result.format_observation(max_rows=10) + "\n"
            obs_ids = tokenizer(
                obs_text, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0].to(device)
            if obs_ids.numel() == 0:
                break
            n_obs = obs_ids.numel()
            gen_ids_parts.append(obs_ids)
            gen_mask_parts.append(torch.zeros_like(obs_ids))
            text_parts.append(obs_text)
            turn_info.append({"start": token_offset, "end": token_offset + n_obs,
                              "is_model": False, "action_success": None,
                              "has_answer": False})
            token_offset += n_obs
            current_ids = torch.cat([current_ids[0], new_tokens, obs_ids]).unsqueeze(0)
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    if not gen_ids_parts:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, "", []
    gen_ids = torch.cat(gen_ids_parts)
    gen_mask = torch.cat(gen_mask_parts)
    gen_text = "".join(text_parts)
    return gen_ids, gen_mask, gen_text, turn_info


@torch.no_grad()
def sample_completions(
    engine, tokenizer, prompt_ids: torch.Tensor, schema: str, db_path: str | None,
    num_generations: int, max_new_tokens_per_turn: int, max_turns: int,
    temperature: float, top_p: float, greedy_generations: int = 0,
) -> list[tuple[torch.Tensor, torch.Tensor, str, list[dict]]]:
    """Sample G interactive rollouts from the current policy for one prompt.

    Returns list of (gen_ids, gen_mask, gen_text, turn_info) tuples.
    """
    engine.eval()
    results: list[tuple[torch.Tensor, torch.Tensor, str, list[dict]]] = []
    greedy_generations = min(max(greedy_generations, 0), num_generations)
    for i in range(num_generations):
        do_sample = i >= greedy_generations and temperature > 0.0
        results.append(_rollout_one(
            engine, tokenizer, prompt_ids, schema, db_path,
            do_sample=do_sample, temperature=temperature, top_p=top_p,
            max_new_tokens_per_turn=max_new_tokens_per_turn, max_turns=max_turns,
        ))
    engine.train()
    return results


def _retokenize_vllm_rollout(
    tokenizer, text_segments: list[tuple[str, bool]],
    turn_info: list[dict], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Convert vLLM text segments into (gen_ids, gen_mask, updated_turn_info)."""
    gen_ids_parts: list[torch.Tensor] = []
    gen_mask_parts: list[torch.Tensor] = []
    # Rebuild turn_info with token-level start/end boundaries
    updated_turn_info: list[dict] = []
    offset = 0
    ti_idx = 0
    for text, is_model in text_segments:
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        n = ids.numel()
        gen_ids_parts.append(ids)
        gen_mask_parts.append(
            torch.ones_like(ids) if is_model else torch.zeros_like(ids)
        )
        # Match with turn_info entry
        if ti_idx < len(turn_info):
            entry = dict(turn_info[ti_idx])
            entry["start"] = offset
            entry["end"] = offset + n
            updated_turn_info.append(entry)
            ti_idx += 1
        offset += n
    if not gen_ids_parts:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, []
    return torch.cat(gen_ids_parts), torch.cat(gen_mask_parts), updated_turn_info


def sample_completions_vllm(
    vllm_engine, tokenizer, prompt_text: str, prompt_ids: torch.Tensor,
    schema: str, db_path: str | None,
    num_generations: int, max_tokens_per_turn: int, max_turns: int,
    temperature: float, top_p: float, device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, str, list[dict]]]:
    """vLLM-based sample_completions — same output format as HF version."""
    from .agent.vllm_rollout import VLLMRolloutEngine
    results = vllm_engine.sample_completions(
        prompt_text=prompt_text,
        schema=schema,
        db_path=db_path,
        num_generations=num_generations,
        max_tokens_per_turn=max_tokens_per_turn,
        max_turns=max_turns,
        temperature=temperature,
        top_p=top_p,
    )
    completions = []
    for r in results:
        gen_ids, gen_mask, fixed_turn_info = _retokenize_vllm_rollout(
            tokenizer, r.text_segments, r.turn_info, device
        )
        completions.append((gen_ids, gen_mask, r.gen_text, fixed_turn_info))
    return completions


def _completion_token_log_probs(
    model, full_ids: torch.Tensor, prompt_len: int, gen_ids: torch.Tensor,
    *, return_logits: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=full_ids)
    logits = outputs.logits[0, prompt_len - 1:-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    token_lps = log_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)
    if return_logits:
        return token_lps, logits
    return token_lps


def maybe_disable_adapter(model):
    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter):
        return disable_adapter()
    return nullcontext()


def compute_single_grpo_loss(
    engine,
    prompt_ids: torch.Tensor,
    gen_ids: torch.Tensor,
    gen_mask: torch.Tensor,
    advantage: float,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor | None = None,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
    *,
    turn_info: list[dict] | None = None,
    use_turn_delta: bool = False,
    use_span_weighting: bool = False,
) -> torch.Tensor:
    """Compute clipped surrogate loss for ONE interactive rollout.

    Supports three v5 enhancements (composable):
    - turn_delta: weight tokens by per-turn contribution (successful exec, Answer)
    - span_weighting: weight tokens by entropy (uncertain decisions get more gradient)
    """
    device = engine.device
    prompt_ids_d = prompt_ids.to(device)
    gen_ids_d = gen_ids.to(device)
    gen_mask_d = gen_mask.to(device).float()
    full_ids = torch.cat([prompt_ids_d, gen_ids_d]).unsqueeze(0)
    prompt_len = prompt_ids_d.shape[0]

    need_logits = use_span_weighting
    if need_logits:
        token_log_probs, logits = _completion_token_log_probs(
            engine, full_ids, prompt_len, gen_ids_d, return_logits=True
        )
    else:
        token_log_probs = _completion_token_log_probs(engine, full_ids, prompt_len, gen_ids_d)

    # Build composite token weights: gen_mask × turn_weights × entropy_weights
    weights = gen_mask_d.clone()

    if use_turn_delta and turn_info:
        tw = _build_turn_weights(turn_info, gen_ids_d.shape[0], device)
        weights = weights * tw

    if use_span_weighting:
        ew = _entropy_weights(logits, gen_mask_d)
        weights = weights * ew

    n_eff = weights.sum().clamp(min=1.0)
    new_lp_mean = (token_log_probs * weights).sum() / n_eff

    ratio = torch.exp(new_lp_mean - old_log_prob.to(device))
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    loss = -torch.min(surr1, surr2)

    if kl_coef > 0.0 and ref_log_prob is not None:
        n_gen = gen_mask_d.sum().clamp(min=1.0)
        log_ratio_mean = torch.clamp(new_lp_mean - ref_log_prob.to(device), min=-5.0, max=5.0)
        approx_kl = torch.exp(log_ratio_mean) - 1.0 - log_ratio_mean
        loss = loss + kl_coef * approx_kl * n_gen

    return loss


@torch.no_grad()
def compute_log_probs(
    engine,
    prompt_ids: torch.Tensor,
    gen_ids: torch.Tensor,
    gen_mask: torch.Tensor,
    *,
    disable_adapter: bool = False,
    ref_model=None,
) -> torch.Tensor:
    """Mean per-token log-prob over model-generated tokens (excludes injected obs).

    When `ref_model` is provided, the forward pass runs through it directly —
    used by the dual-model KL path (actor + frozen-SFT ref). Otherwise falls
    back to the legacy `engine.module` + disable_adapter(...) approximation
    for bf16 runs that still merge SFT into the base.
    """
    if ref_model is not None:
        module = ref_model
        device = next(module.parameters()).device
    else:
        module = engine.module
        device = engine.device
    prompt_ids_d = prompt_ids.to(device)
    gen_ids_d = gen_ids.to(device)
    gen_mask_d = gen_mask.to(device).float()
    full_ids = torch.cat([prompt_ids_d, gen_ids_d]).unsqueeze(0)
    prompt_len = prompt_ids_d.shape[0]
    context = (
        maybe_disable_adapter(module)
        if disable_adapter and ref_model is None
        else nullcontext()
    )
    with context:
        token_log_probs = _completion_token_log_probs(
            module, full_ids, prompt_len, gen_ids_d
        )
    n_gen = gen_mask_d.sum().clamp(min=1.0)
    return ((token_log_probs * gen_mask_d).sum() / n_gen).cpu()


def annealed_temperature(progress: float, start: float, end: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return start + (end - start) * progress


def build_prompt_buckets(prompts: list[dict]) -> dict[str, list[dict]]:
    buckets = {"easy": [], "medium": [], "hard": [], "unknown": []}
    for prompt in prompts:
        buckets.setdefault(prompt.get("difficulty", "unknown"), []).append(prompt)
    return buckets


def curriculum_weights(progress: float) -> dict[str, float]:
    if progress < 0.25:
        return {"easy": 0.55, "medium": 0.35, "hard": 0.10, "unknown": 0.0}
    if progress < 0.70:
        return {"easy": 0.25, "medium": 0.45, "hard": 0.30, "unknown": 0.0}
    return {"easy": 0.15, "medium": 0.35, "hard": 0.50, "unknown": 0.0}


def select_prompt(
    prompts: list[dict],
    prompt_idx: int,
    *,
    use_curriculum: bool,
    progress: float,
    buckets: dict[str, list[dict]],
    bucket_cursors: dict[str, int],
    rng: random.Random,
) -> tuple[dict, int]:
    if not use_curriculum:
        return prompts[prompt_idx % len(prompts)], prompt_idx + 1

    weights = curriculum_weights(progress)
    labels = [label for label, weight in weights.items() if weight > 0 and buckets.get(label)]
    if not labels:
        return prompts[prompt_idx % len(prompts)], prompt_idx + 1

    sampled_label = rng.choices(labels, weights=[weights[label] for label in labels], k=1)[0]
    bucket = buckets[sampled_label]
    cursor = bucket_cursors[sampled_label]
    bucket_cursors[sampled_label] = cursor + 1
    return bucket[cursor % len(bucket)], prompt_idx


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def cosine_lr(step: int, total_steps: int, lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO RL with native DeepSpeed")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--source", type=str, default="spider",
                        choices=["spider", "sql-create-context"])
    parser.add_argument("--output-dir", type=str, default="outputs/grpo-agent")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512,
                        help="Max new tokens per ReAct turn during training rollouts.")
    parser.add_argument("--max-turns", type=int, default=5,
                        help="Max ReAct turns per interactive rollout.")
    parser.add_argument("--advantage-threshold", type=float, default=0.3,
                        help="Skip the group if max |advantage| is below this (prevents "
                             "noise updates on zero-signal hard prompts).")
    parser.add_argument("--rvds-threshold", type=float, default=0.0,
                        help="RVDS (Rollout-Variance Dynamic Sampling): skip group if "
                             "raw reward variance across rollouts is below this. "
                             "Catches both dense-hacking (all rollouts lock to template) "
                             "and sparse-collapse (all rollouts fail). 0 = disabled.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy-generations", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--reward-profile", type=str, default="dense",
                        choices=["legacy", "dense", "sparse", "exec_only", "cgfr"])
    parser.add_argument("--curriculum", dest="curriculum", action="store_true",
                        help="Use Spider difficulty curriculum during GRPO")
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false",
                        help="Disable Spider difficulty curriculum during GRPO")
    parser.set_defaults(curriculum=True)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--quant-mode", type=str, default="none",
                        choices=["none", "bnb4"],
                        help="'bnb4' enables QLoRA-GRPO: actor = 4-bit base + "
                             "trainable SFT LoRA; a separate frozen 4-bit base + "
                             "SFT LoRA snapshot serves as the KL reference. "
                             "'none' keeps the legacy bf16 merge path.")
    # v5 enhancements
    parser.add_argument("--adaptive-turns", action="store_true",
                        help="Vary max_turns by difficulty (easy=0.4x, hard=1.4x)")
    parser.add_argument("--span-weighting", action="store_true",
                        help="Weight token losses by entropy (uncertain tokens get more gradient)")
    parser.add_argument("--turn-delta", action="store_true",
                        help="Per-turn credit assignment (successful actions/Answer get more weight)")
    # vLLM acceleration
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM for batched rollout generation (~10x faster)")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.4,
                        help="GPU memory fraction for vLLM (rest goes to training)")
    parser.add_argument("--vllm-sync-steps", type=int, default=10,
                        help="Sync LoRA weights to vLLM every N steps")
    parser.add_argument("--deepspeed", type=str, default="configs/ds_zero2.json")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model: two paths ----
    #
    # quant-mode bnb4 (QLoRA-GRPO):
    #   actor = 4-bit base + trainable SFT LoRA (same adapter continues training)
    #   ref   = 4-bit base + frozen SFT LoRA snapshot, outside DeepSpeed
    #
    # quant-mode none (legacy):
    #   base is bf16; the SFT adapter is merged into base and a fresh RL LoRA
    #   is stacked on top. KL uses `disable_adapter(...)` on the actor.
    ref_model = None
    _merged_dir = None
    vllm_engine = None

    if args.quant_mode == "bnb4":
        if not args.adapter_path:
            raise ValueError(
                "--quant-mode bnb4 requires --adapter-path (SFT adapter to continue "
                "training). QLoRA-GRPO does not support a cold-start fresh LoRA in "
                "this script — run SFT first with --quant-mode bnb4 and feed its "
                "adapter here."
            )
        if args.use_vllm:
            raise ValueError(
                "--use-vllm is not supported together with --quant-mode bnb4 yet: "
                "vLLM export would require re-saving a merged base, and 4-bit "
                "merges are lossy. Disable one or the other."
            )
        actor_base = load_base_model(args.model_name_or_path, quant_mode="bnb4")
        model = load_adapter_for_training(
            actor_base, args.adapter_path, quant_mode="bnb4",
        )
        ref_base = load_base_model(args.model_name_or_path, quant_mode="bnb4")
        ref_model = load_adapter_frozen(ref_base, args.adapter_path)
    else:
        model = load_base_model(args.model_name_or_path, quant_mode="none")
        if args.adapter_path:
            from peft import PeftModel
            log.info("Loading and merging SFT adapter from %s", args.adapter_path)
            model = PeftModel.from_pretrained(model, args.adapter_path)
            model = model.merge_and_unload()

        # ---- Save merged base for vLLM BEFORE applying LoRA ----
        if args.use_vllm:
            import tempfile
            # Use output dir instead of /tmp to avoid filling small /tmp partition
            _merged_base_root = str(Path(args.output_dir) / "_vllm_merged_tmp")
            os.makedirs(_merged_base_root, exist_ok=True)
            _merged_dir = tempfile.mkdtemp(prefix="grpo_merged_", dir=_merged_base_root)
            log.info("Saving merged base model for vLLM to %s", _merged_dir)
            model.save_pretrained(_merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(_merged_dir)

        model = attach_fresh_lora(
            model,
            quant_mode="none",
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable params: %d (quant=%s, ref=%s)",
             trainable, args.quant_mode,
             "separate" if ref_model is not None else "disable_adapter")

    # ---- Prompts ----
    prompts = prepare_prompts(
        args.source, tokenizer, num_samples=args.num_samples,
    )
    log.info("Loaded %d prompts for GRPO", len(prompts))
    prompt_buckets = build_prompt_buckets(prompts)
    bucket_cursors = {label: 0 for label in prompt_buckets}
    rng = random.Random(42)

    # ---- Initialize vLLM from saved merged base ----
    if args.use_vllm and _merged_dir is not None:
        from .agent.vllm_rollout import VLLMRolloutEngine
        vllm_engine = VLLMRolloutEngine(
            _merged_dir,
            gpu_memory_utilization=args.vllm_gpu_util,
            max_model_len=2048,
        )
        log.info("vLLM rollout engine ready (gpu_util=%.2f)", args.vllm_gpu_util)

    # ---- DeepSpeed config ----
    with open(args.deepspeed) as f:
        ds_config = json.load(f)

    # gradient_accumulation_steps = num_generations so DeepSpeed accumulates
    # across per-completion backward calls and reduces once per prompt.
    ds_config["train_micro_batch_size_per_gpu"] = 1
    ds_config["gradient_accumulation_steps"] = args.num_generations
    ds_config["train_batch_size"] = args.num_generations
    ds_config["gradient_clipping"] = 1.0
    ds_config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": args.learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "torch_adam": True,
        },
    }
    ds_config.pop("scheduler", None)

    # ---- deepspeed.initialize() ----
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=ds_config,
    )

    # ---- GRPO Training Loop ----
    warmup_steps = int(args.max_steps * args.warmup_ratio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    prompt_idx = 0
    running_loss = 0.0
    running_reward = 0.0
    running_reward_components = {
        "format_score": 0.0,
        "validity_score": 0.0,
        "structure_score": 0.0,
        "execution_score": 0.0,
        "correctness_score": 0.0,
        "error_penalty": 0.0,
    }
    skipped_groups = 0

    log.info(
        "Starting GRPO: %d steps, G=%d, temp=%.2f->%.2f, greedy=%d, clip=%.2f, kl=%.3f, reward=%s",
        args.max_steps,
        args.num_generations,
        args.temperature,
        args.min_temperature,
        args.greedy_generations,
        args.clip_eps,
        args.kl_coef,
        args.reward_profile,
    )

    while global_step < args.max_steps:
        progress = global_step / max(args.max_steps - 1, 1)
        data, prompt_idx = select_prompt(
            prompts,
            prompt_idx,
            use_curriculum=args.curriculum and args.source == "spider",
            progress=progress,
            buckets=prompt_buckets,
            bucket_cursors=bucket_cursors,
            rng=rng,
        )

        prompt_ids = data["prompt_ids"]
        schema = data["schema"]
        gold_sql = data["gold_sql"]
        db_path = data.get("db_path")
        current_temperature = annealed_temperature(
            progress, args.temperature, args.min_temperature
        )
        reward_weights = resolve_reward_weights(args.reward_profile, progress)

        # 1. Sample G interactive rollouts (model <-> real DB)
        # Idea 4: adapt max_turns by difficulty
        difficulty = data.get("difficulty", "unknown")
        effective_max_turns = (
            _adaptive_max_turns(difficulty, args.max_turns)
            if args.adaptive_turns else args.max_turns
        )

        if vllm_engine is not None:
            prompt_text = build_react_inference_prompt(schema, data["question"], tokenizer)
            completions = sample_completions_vllm(
                vllm_engine, tokenizer, prompt_text, prompt_ids,
                schema, db_path,
                num_generations=args.num_generations,
                max_tokens_per_turn=args.max_completion_length,
                max_turns=effective_max_turns,
                temperature=current_temperature,
                top_p=args.top_p,
                device=engine.device,
            )
        else:
            completions = sample_completions(
                engine, tokenizer, prompt_ids, schema, db_path,
                num_generations=args.num_generations,
                max_new_tokens_per_turn=args.max_completion_length,
                max_turns=effective_max_turns,
                temperature=current_temperature,
                top_p=args.top_p,
                greedy_generations=args.greedy_generations,
            )

        # 2. Compute rewards (use real DB when available)
        rewards = []
        reward_parts = []
        for _, _, gen_text, _ in completions:
            breakdown = reward_breakdown(
                gen_text,
                gold_sql,
                schema,
                db_path=db_path,
                profile=args.reward_profile,
                progress=progress,
                weights=reward_weights,
            )
            reward_parts.append(breakdown)
            rewards.append(breakdown.total)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        # RVDS: skip this group if raw reward variance collapsed. Catches both
        # dense-hacking (all rollouts at same template reward) and sparse-collapse
        # (all rollouts fail with reward 0). Advantage-threshold alone misses the
        # first case because normalized advantages can still have high max |adv|
        # even when the underlying reward variance is tiny.
        if args.rvds_threshold > 0.0 and rewards_t.var().item() < args.rvds_threshold:
            skipped_groups += 1
            continue

        # 3. Compute advantages (group-relative normalization)
        if rewards_t.std() > 1e-6:
            advantages = (rewards_t - rewards_t.mean()) / rewards_t.std()
        else:
            advantages = torch.zeros_like(rewards_t)

        # Skip if group signal is too weak (prevents hard-tier noise updates).
        if advantages.abs().max() < args.advantage_threshold:
            skipped_groups += 1
            continue

        # 4. Compute old log-probs (before parameter update)
        valid_indices = [
            i for i in range(len(completions))
            if completions[i][0].numel() > 0
            and completions[i][1].sum().item() > 0
            and abs(advantages[i].item()) > 1e-6
        ]
        if not valid_indices:
            skipped_groups += 1
            continue

        old_log_probs = {}
        ref_log_probs = {}
        for i in valid_indices:
            gen_ids, gen_mask, _, _ = completions[i]
            old_log_probs[i] = compute_log_probs(engine, prompt_ids, gen_ids, gen_mask)
            if args.kl_coef > 0.0:
                if ref_model is not None:
                    # QLoRA path: dedicated frozen reference model (base+SFT LoRA).
                    ref_log_probs[i] = compute_log_probs(
                        engine, prompt_ids, gen_ids, gen_mask, ref_model=ref_model,
                    )
                else:
                    # Legacy bf16 path: ref = actor with LoRA disabled.
                    ref_log_probs[i] = compute_log_probs(
                        engine, prompt_ids, gen_ids, gen_mask, disable_adapter=True,
                    )

        # 5. Per-completion forward+backward (DeepSpeed accumulates across G calls)
        step_loss = 0.0
        for i in valid_indices:
            gen_ids, gen_mask, _, turn_info = completions[i]
            loss_i = compute_single_grpo_loss(
                engine, prompt_ids, gen_ids, gen_mask, advantages[i].item(),
                old_log_probs[i],
                ref_log_prob=ref_log_probs.get(i),
                clip_eps=args.clip_eps,
                kl_coef=args.kl_coef,
                turn_info=turn_info,
                use_turn_delta=args.turn_delta,
                use_span_weighting=args.span_weighting,
            )
            # Scale for accumulation across valid completions
            scaled_loss = loss_i / max(len(valid_indices), 1)
            engine.backward(scaled_loss)
            engine.step()  # DeepSpeed only truly steps at accumulation boundary
            step_loss += loss_i.item() / max(len(valid_indices), 1)

        # Pad remaining accumulation steps if valid < num_generations
        # so DeepSpeed reaches its gradient_accumulation boundary
        for _ in range(args.num_generations - len(valid_indices)):
            dummy = torch.tensor(0.0, device=engine.device, requires_grad=True)
            engine.backward(dummy)
            engine.step()

        global_step += 1
        running_loss += step_loss
        running_reward += sum(rewards) / len(rewards)
        for component_name in running_reward_components:
            running_reward_components[component_name] += (
                sum(getattr(b, component_name) for b in reward_parts) / len(reward_parts)
            )

        # Manual LR scheduling
        new_lr = cosine_lr(global_step, args.max_steps, args.learning_rate, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        # Logging
        if global_step % args.logging_steps == 0:
            avg_loss = running_loss / args.logging_steps
            avg_reward = running_reward / args.logging_steps
            avg_components = {
                name: value / args.logging_steps
                for name, value in running_reward_components.items()
            }
            log.info(
                "step=%d/%d loss=%.4f reward=%.3f fmt=%.2f val=%.2f struct=%.2f exec=%.2f corr=%.2f err=%.2f temp=%.2f lr=%.2e skipped=%d rewards=%s",
                global_step,
                args.max_steps,
                avg_loss,
                avg_reward,
                avg_components["format_score"],
                avg_components["validity_score"],
                avg_components["structure_score"],
                avg_components["execution_score"],
                avg_components["correctness_score"],
                avg_components["error_penalty"],
                current_temperature,
                new_lr,
                skipped_groups,
                [round(r, 2) for r in rewards],
            )
            running_loss = 0.0
            running_reward = 0.0
            running_reward_components = {name: 0.0 for name in running_reward_components}
            skipped_groups = 0

        # Save + vLLM weight sync
        if global_step % args.save_steps == 0:
            engine.save_checkpoint(str(output_dir), tag=f"checkpoint-{global_step}")
            model.save_pretrained(str(output_dir / f"checkpoint-{global_step}"))
            tokenizer.save_pretrained(str(output_dir / f"checkpoint-{global_step}"))
            log.info("Saved checkpoint at step %d", global_step)

        # Periodic vLLM weight sync (re-init with updated merged weights).
        #
        # Reversible merge: write merged base to disk for vLLM reload, then
        # unmerge so the live PeftModel and the DeepSpeed engine/optimizer
        # keep pointing at the same trainable LoRA parameters. A destructive
        # merge_and_unload() here would strand `engine.module` and the
        # optimizer state on a now-detached module, and the freshly wrapped
        # replacement would carry random LoRA — silently wiping training
        # progress. Only reached in quant-mode=none.
        if (vllm_engine is not None
                and args.vllm_sync_steps > 0
                and global_step % args.vllm_sync_steps == 0):
            from .agent.vllm_rollout import VLLMRolloutEngine as _VLLMRE
            log.info("Syncing weights to vLLM at step %d", global_step)
            vllm_engine.shutdown()
            model.merge_adapter()
            try:
                inner_base = model.get_base_model()
                inner_base.save_pretrained(_merged_dir, safe_serialization=True)
            finally:
                model.unmerge_adapter()
            vllm_engine = _VLLMRE(
                _merged_dir,
                gpu_memory_utilization=args.vllm_gpu_util,
                max_model_len=2048,
            )
            log.info("vLLM weights synced at step %d", global_step)

    # ---- Final save ----
    if vllm_engine is not None:
        vllm_engine.shutdown()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log.info("GRPO complete. Model saved to %s", output_dir)


if __name__ == "__main__":
    main()
