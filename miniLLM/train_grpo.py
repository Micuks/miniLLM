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
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

from .agent.react import build_react_inference_prompt
from .agent.reward import resolve_reward_weights, reward_breakdown

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

@torch.no_grad()
def sample_completions(
    engine, tokenizer, prompt_ids: torch.Tensor, num_generations: int,
    max_new_tokens: int, temperature: float, top_p: float,
    greedy_generations: int = 0,
) -> list[tuple[torch.Tensor, str]]:
    """Sample G completions from the current policy for one prompt."""
    engine.eval()
    device = engine.device
    input_ids = prompt_ids.unsqueeze(0).to(device)
    results = []
    greedy_generations = min(max(greedy_generations, 0), num_generations)
    for i in range(num_generations):
        do_sample = i >= greedy_generations and temperature > 0.0
        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        output_ids = engine.module.generate(**generate_kwargs)
        # Extract only the generated part
        gen_ids = output_ids[0, input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append((gen_ids, gen_text))
    engine.train()
    return results


def _completion_token_log_probs(model, full_ids: torch.Tensor, prompt_len: int, gen_ids: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=full_ids)
    logits = outputs.logits[0, prompt_len - 1:-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)


def maybe_disable_adapter(model):
    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter):
        return disable_adapter()
    return nullcontext()


def compute_single_grpo_loss(
    engine,
    prompt_ids: torch.Tensor,
    gen_ids: torch.Tensor,
    advantage: float,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor | None = None,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
) -> torch.Tensor:
    """Compute clipped surrogate loss for ONE completion.

    ZeRO-2 requires exactly one forward per backward. So we compute loss
    for each completion separately and use DeepSpeed gradient accumulation
    to aggregate across the group.
    """
    device = engine.device
    prompt_ids_d = prompt_ids.to(device)
    gen_ids_d = gen_ids.to(device)
    full_ids = torch.cat([prompt_ids_d, gen_ids_d]).unsqueeze(0)
    prompt_len = prompt_ids_d.shape[0]

    token_log_probs = _completion_token_log_probs(engine, full_ids, prompt_len, gen_ids_d)
    new_lp = token_log_probs.sum()

    ratio = torch.exp(new_lp - old_log_prob.to(device))
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    loss = -torch.min(surr1, surr2)

    if kl_coef > 0.0 and ref_log_prob is not None:
        # Approximate per-token KL against the frozen merged-SFT policy by
        # disabling the trainable LoRA adapter on the same PEFT model.
        mean_log_ratio = (new_lp - ref_log_prob.to(device)) / max(gen_ids_d.numel(), 1)
        mean_log_ratio = torch.clamp(mean_log_ratio, min=-5.0, max=5.0)
        approx_kl = torch.exp(mean_log_ratio) - 1.0 - mean_log_ratio
        loss = loss + kl_coef * approx_kl

    return loss


@torch.no_grad()
def compute_log_probs(
    engine,
    prompt_ids: torch.Tensor,
    gen_ids: torch.Tensor,
    *,
    disable_adapter: bool = False,
) -> torch.Tensor:
    """Compute total log-prob of gen_ids given prompt_ids under current policy."""
    device = engine.device
    prompt_ids_d = prompt_ids.to(device)
    gen_ids_d = gen_ids.to(device)
    full_ids = torch.cat([prompt_ids_d, gen_ids_d]).unsqueeze(0)
    prompt_len = prompt_ids_d.shape[0]
    context = maybe_disable_adapter(engine.module) if disable_adapter else nullcontext()
    with context:
        token_log_probs = _completion_token_log_probs(
            engine.module, full_ids, prompt_len, gen_ids_d
        )
    return token_log_probs.sum().cpu()


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
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy-generations", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--reward-profile", type=str, default="dense",
                        choices=["legacy", "dense"])
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

    # ---- Model: load base + merge SFT adapter + apply new LoRA ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if args.adapter_path:
        log.info("Loading and merging SFT adapter from %s", args.adapter_path)
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable params: %d", trainable)

    # ---- Prompts ----
    prompts = prepare_prompts(
        args.source, tokenizer, num_samples=args.num_samples,
    )
    log.info("Loaded %d prompts for GRPO", len(prompts))
    prompt_buckets = build_prompt_buckets(prompts)
    bucket_cursors = {label: 0 for label in prompt_buckets}
    rng = random.Random(42)

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

        # 1. Sample G completions
        completions = sample_completions(
            engine, tokenizer, prompt_ids,
            num_generations=args.num_generations,
            max_new_tokens=args.max_completion_length,
            temperature=current_temperature,
            top_p=args.top_p,
            greedy_generations=args.greedy_generations,
        )

        # 2. Compute rewards (use real DB when available)
        rewards = []
        reward_parts = []
        for _, gen_text in completions:
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

        # 3. Compute advantages (group-relative normalization)
        if rewards_t.std() > 1e-6:
            advantages = (rewards_t - rewards_t.mean()) / rewards_t.std()
        else:
            advantages = torch.zeros_like(rewards_t)

        # Skip if all rewards are identical (no signal)
        if advantages.abs().max() < 1e-6:
            skipped_groups += 1
            continue

        # 4. Compute old log-probs (before parameter update)
        valid_indices = [i for i in range(len(completions))
                         if completions[i][0].numel() > 0 and abs(advantages[i].item()) > 1e-6]
        if not valid_indices:
            skipped_groups += 1
            continue

        old_log_probs = {}
        ref_log_probs = {}
        for i in valid_indices:
            gen_ids, _ = completions[i]
            old_log_probs[i] = compute_log_probs(engine, prompt_ids, gen_ids)
            if args.kl_coef > 0.0:
                ref_log_probs[i] = compute_log_probs(
                    engine, prompt_ids, gen_ids, disable_adapter=True
                )

        # 5. Per-completion forward+backward (DeepSpeed accumulates across G calls)
        # Filter to completions with non-zero advantage and non-empty gen

        step_loss = 0.0
        for vi, i in enumerate(valid_indices):
            gen_ids, _ = completions[i]
            loss_i = compute_single_grpo_loss(
                engine, prompt_ids, gen_ids, advantages[i].item(),
                old_log_probs[i],
                ref_log_prob=ref_log_probs.get(i),
                clip_eps=args.clip_eps,
                kl_coef=args.kl_coef,
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
                "step=%d/%d loss=%.4f reward=%.3f fmt=%.2f val=%.2f struct=%.2f exec=%.2f corr=%.2f temp=%.2f lr=%.2e skipped=%d rewards=%s",
                global_step,
                args.max_steps,
                avg_loss,
                avg_reward,
                avg_components["format_score"],
                avg_components["validity_score"],
                avg_components["structure_score"],
                avg_components["execution_score"],
                avg_components["correctness_score"],
                current_temperature,
                new_lr,
                skipped_groups,
                [round(r, 2) for r in rewards],
            )
            running_loss = 0.0
            running_reward = 0.0
            running_reward_components = {name: 0.0 for name in running_reward_components}
            skipped_groups = 0

        # Save
        if global_step % args.save_steps == 0:
            engine.save_checkpoint(str(output_dir), tag=f"checkpoint-{global_step}")
            model.save_pretrained(str(output_dir / f"checkpoint-{global_step}"))
            tokenizer.save_pretrained(str(output_dir / f"checkpoint-{global_step}"))
            log.info("Saved checkpoint at step %d", global_step)

    # ---- Final save ----
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log.info("GRPO complete. Model saved to %s", output_dir)


if __name__ == "__main__":
    main()
