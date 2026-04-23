"""vLLM-accelerated interactive rollouts for GRPO training.

Replaces sequential HF generate calls with batched vLLM generation.
For G=8 rollouts with max_turns=5:
  - HF: 40 sequential generate calls (G × turns)
  - vLLM: 5 batched calls (1 per turn, G prompts in parallel)

Usage:
    engine = VLLMRolloutEngine("Qwen/Qwen2.5-3B-Instruct", gpu_memory_utilization=0.4)
    completions = engine.sample_completions(prompt_text, schema, db_path, ...)
    engine.shutdown()
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .env import SQLExecutionEnv, SQLExecutionEnvFromDB
from .react import extract_final_sql

log = logging.getLogger(__name__)

_ACTION_SQL_RE = re.compile(r'Action:\s*execute_sql\["(.+?)"\]', re.DOTALL)


@dataclass
class RolloutResult:
    """Result from one interactive rollout."""
    gen_text: str = ""
    turn_info: list[dict] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    # For re-tokenization into gen_ids/gen_mask: list of (text, is_model_generated)
    text_segments: list[tuple[str, bool]] = field(default_factory=list)


def _make_env(schema: str, db_path: str | None):
    if db_path:
        return SQLExecutionEnvFromDB(db_path)
    return SQLExecutionEnv(schema)


class VLLMRolloutEngine:
    """Batched interactive rollouts using vLLM."""

    def __init__(
        self,
        model_name: str,
        *,
        adapter_path: str | None = None,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 4096,
        enforce_eager: bool = True,
    ):
        llm_kwargs = dict(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
        )
        if adapter_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 64
            self._adapter_path = adapter_path
        else:
            self._adapter_path = None

        log.info("Initializing vLLM: model=%s, gpu_util=%.2f, adapter=%s",
                 model_name, gpu_memory_utilization, adapter_path)
        self.llm = LLM(**llm_kwargs)
        self._lora_request = None
        if self._adapter_path:
            self._lora_request = LoRARequest("sft", 1, self._adapter_path)

    def shutdown(self):
        """Release GPU memory."""
        del self.llm
        torch.cuda.empty_cache()

    def generate_single_pass(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> str:
        """One-shot (non-interactive) greedy completion — parity with
        eval_agent.generate_single_pass.

        Does NOT stop on `Observation:` and does NOT cap at a single ReAct
        turn, so direct-eval runs can produce a full `Answer: <sql>` tail
        rather than being truncated after a partial Thought/Action.
        """
        params = SamplingParams(
            n=1,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        kwargs = {}
        if self._lora_request:
            kwargs["lora_request"] = self._lora_request
        outputs = self.llm.generate([prompt_text], params, **kwargs)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text

    def sample_completions(
        self,
        prompt_text: str,
        schema: str,
        db_path: str | None,
        num_generations: int,
        max_tokens_per_turn: int,
        max_turns: int,
        temperature: float,
        top_p: float,
    ) -> list[RolloutResult]:
        """Generate G interactive rollouts in parallel using vLLM.

        Turn 1: 1 prompt × n=G → G completions (one vLLM call)
        Turn 2+: K active prompts × n=1 (one vLLM call, K ≤ G)
        """
        # Turn 1: batch-generate G completions from the same prompt
        params = SamplingParams(
            n=num_generations,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            max_tokens=max_tokens_per_turn,
            stop=["Observation:"],
        )
        generate_kwargs = {}
        if self._lora_request:
            generate_kwargs["lora_request"] = self._lora_request

        outputs = self.llm.generate([prompt_text], params, **generate_kwargs)
        if not outputs:
            return [RolloutResult() for _ in range(num_generations)]

        # Initialize per-rollout state
        rollouts: list[dict] = []
        for i, completion in enumerate(outputs[0].outputs):
            text = completion.text
            token_ids = list(completion.token_ids)
            rollouts.append({
                "full_prompt": prompt_text + text,
                "text_parts": [text],
                "text_segments": [(text, True)],  # (text, is_model)
                "token_ids": token_ids,
                "turn_info": [],
                "done": False,
            })
            has_answer = "Answer:" in text
            action_m = _ACTION_SQL_RE.search(text)
            action_success = None

            if has_answer:
                rollouts[i]["turn_info"].append({
                    "is_model": True, "action_success": None, "has_answer": True,
                })
                rollouts[i]["done"] = True
            elif action_m:
                sql = action_m.group(1)
                try:
                    env = _make_env(schema, db_path)
                    result = env.execute(sql)
                    action_success = result.success
                    obs_text = "Observation: " + result.format_observation(max_rows=10) + "\n"
                    env.close()
                except Exception:
                    action_success = False
                    obs_text = "Observation: Error executing SQL\n"
                rollouts[i]["turn_info"].append({
                    "is_model": True, "action_success": action_success, "has_answer": False,
                })
                rollouts[i]["turn_info"].append({
                    "is_model": False, "action_success": None, "has_answer": False,
                })
                rollouts[i]["text_parts"].append(obs_text)
                rollouts[i]["text_segments"].append((obs_text, False))
                rollouts[i]["full_prompt"] += obs_text
            else:
                rollouts[i]["turn_info"].append({
                    "is_model": True, "action_success": None, "has_answer": False,
                })
                rollouts[i]["done"] = True

        # Subsequent turns: batch only active rollouts
        single_params = SamplingParams(
            n=1,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            max_tokens=max_tokens_per_turn,
            stop=["Observation:"],
        )

        for _turn in range(1, max_turns):
            active_indices = [i for i, r in enumerate(rollouts) if not r["done"]]
            if not active_indices:
                break

            active_prompts = [rollouts[i]["full_prompt"] for i in active_indices]
            turn_outputs = self.llm.generate(active_prompts, single_params, **generate_kwargs)

            for j, idx in enumerate(active_indices):
                if j >= len(turn_outputs) or not turn_outputs[j].outputs:
                    rollouts[idx]["done"] = True
                    continue

                text = turn_outputs[j].outputs[0].text
                token_ids = list(turn_outputs[j].outputs[0].token_ids)
                rollouts[idx]["text_parts"].append(text)
                rollouts[idx]["text_segments"].append((text, True))
                rollouts[idx]["token_ids"].extend(token_ids)
                rollouts[idx]["full_prompt"] += text

                has_answer = "Answer:" in text
                action_m = _ACTION_SQL_RE.search(text)

                if has_answer:
                    rollouts[idx]["turn_info"].append({
                        "is_model": True, "action_success": None, "has_answer": True,
                    })
                    rollouts[idx]["done"] = True
                elif action_m:
                    sql = action_m.group(1)
                    try:
                        env = _make_env(schema, db_path)
                        result = env.execute(sql)
                        action_success = result.success
                        obs_text = "Observation: " + result.format_observation(max_rows=10) + "\n"
                        env.close()
                    except Exception:
                        action_success = False
                        obs_text = "Observation: Error executing SQL\n"
                    rollouts[idx]["turn_info"].append({
                        "is_model": True, "action_success": action_success, "has_answer": False,
                    })
                    rollouts[idx]["turn_info"].append({
                        "is_model": False, "action_success": None, "has_answer": False,
                    })
                    rollouts[idx]["text_parts"].append(obs_text)
                    rollouts[idx]["text_segments"].append((obs_text, False))
                    rollouts[idx]["full_prompt"] += obs_text
                else:
                    rollouts[idx]["turn_info"].append({
                        "is_model": True, "action_success": None, "has_answer": False,
                    })
                    rollouts[idx]["done"] = True

        # Convert to RolloutResult
        results = []
        for r in rollouts:
            gen_text = "".join(r["text_parts"])
            results.append(RolloutResult(
                gen_text=gen_text,
                turn_info=r["turn_info"],
                token_ids=r["token_ids"],
                text_segments=r["text_segments"],
            ))
        return results
