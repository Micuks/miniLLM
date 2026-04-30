---
purpose: Published SOTA numbers for comparison table in paper
date: 2026-04-19
---

# SOTA Baselines for Text-to-SQL Comparison

## Published numbers (papers we'd cite)

### Spider-dev (full 1034 samples)

| Method | Model size | EX_att | EM | Source | Notes |
|---|---|---|---|---|---|
| **Arctic-Text2SQL-R1** | 7B | 79.4% | — | [arxiv:2505.20315](https://arxiv.org/abs/2505.20315) | GRPO + simple exec reward |
| **Arctic-Text2SQL-R1** | 32B | 86.7% | — | same | SOTA open-source |
| **MARS-SQL** | 7B | **89.75%** on test | — | [arxiv:2511.01008](https://arxiv.org/abs/2511.01008) | 3-agent GRPO, dev≈90% |
| **MTIR-SQL** | 7B | 84.6% | — | [arxiv:2510.25510](https://arxiv.org/html/2510.25510v1) | Multi-turn tool invocation |
| **Alpha-SQL** | 32B (frozen) | — | — | [arxiv:2502.17248](https://arxiv.org/abs/2502.17248) | MCTS at inference, BIRD only |
| **ReFoRCE** | ? | — | — | [arxiv:2502.00675](https://arxiv.org/abs/2502.00675) | Spider 2.0 top, different benchmark |
| DIN-SQL (GPT-4) | — | 85.3% | — | Pourreza & Rafiei 2023 | Prompt-based, no training |
| DAIL-SQL (GPT-4) | — | 86.6% | — | Gao et al. 2024 | Prompt-based |

### BIRD-dev (1534 samples, in-domain knowledge required)

| Method | Model size | EX (%) | VES | Source | Notes |
|---|---|---|---|---|---|
| **Arctic-Text2SQL-R1** | 7B | **68.47%** | — | arxiv:2505.20315 | = 70B SFT perf |
| **Arctic-Text2SQL-R1** | 32B | **71.83%** (test) | — | same | **SOTA test** |
| **MARS-SQL** | 7B | 77.84% dev | — | arxiv:2511.01008 | |
| **MCTS-SQL** (GPT-4o) | — | 69.40% | — | arxiv:2501.16607 | |
| **Alpha-SQL** | 32B | 69.7% | — | arxiv:2502.17248 | MCTS, zero-shot |
| **Reasoning-SQL** | ? | — | — | OpenReview HbwkIDWQgN | GRPO + partial rewards |
| **Reward-SQL** | 7B | — | — | arxiv:2505.04671 | PRM for CTE decomposition |
| **Graph-Reward-SQL** | ? | — | — | arxiv:2505.12380v1 | Execution-free RL |

### SFT baselines at 3B scale (for our comparison)

| Method | Model | Spider EX | BIRD EX |
|---|---|---|---|
| Qwen2.5-3B + ReAct SFT (ours) | 3B | 61.0% | ~25% (est.) |
| Qwen2.5-3B + GRPO v3 (ours) | 3B | 48.7% | — |
| Qwen2.5-3B + GRPO v4 (ours) | 3B | 52.0% | — |
| Qwen2.5-3B + **GRPO v5 (ours)** | 3B | *TBD* | *TBD* |

## Paper positioning strategy

Our narrative: "At the 3B model size, we show that careful credit assignment in GRPO closes a significant fraction of the gap to 7B+ methods that use larger models or test-time search."

Comparison framing options:

### Option A: "Match 7B with 3B"
- "Our GRPO v5 achieves X% on Spider-dev, within Y pp of Arctic-R1-7B (79.4%) at half the model size."
- Needs: our EX near 70-75% on Spider → ambitious but possible with 300+ steps

### Option B: "Efficient RL frontier"
- Focus on **training efficiency**: compute per % improvement vs alternatives
- Metric: "step-adjusted EX gain" = (final EX - SFT EX) / training_steps
- This works even if absolute EX is modest — shows our methods extract more per step

### Option C: "Credit assignment ablation study"
- Focus purely on the 3 new techniques (turn-delta, span-weighting, adaptive-turns)
- "What does each contribute? How do they compose?"
- Paper becomes methodology-focused rather than SOTA-chasing
- Safest for limited compute

## Feasibility for EMNLP/ACL at 3B scale

**Realistic 3B EX ceiling on Spider-dev (full)**:
- Current SFT: 61%
- v4 GRPO: 52% (regressed from SFT on interactive)
- v5 @ 100 steps: *TBD, likely 55-65%*
- v5 @ 300 steps: *hopefully 65-72%*

**Realistic 3B EX ceiling on BIRD-dev**:
- BIRD is much harder, requires evidence-based reasoning
- 3B without extensive tuning likely: 15-25%
- With our methods: maybe 25-35%
- Still far from Arctic-R1-7B (68.47%)

**Recommended narrative for submission**: **Option C** (ablation study) because:
1. Doesn't require beating SOTA (hard at 3B)
2. Our contribution is the method, not the absolute number
3. Feature ablation × 3 seeds is rigorous and publishable
4. Can position as "understanding what makes GRPO work for multi-turn agent RL"

## What we need to run

### Absolute minimum for paper (~100 GPU-h)
- 3 seeds × (v4, v5-all) = 6 runs × 200 spider samples
- Single full dev eval (1034 samples) for best config
- Single BIRD dev eval (sanity check)

### Target (~400 GPU-h)
- 3 seeds × 5 configs (above)
- Full Spider-dev + full BIRD-dev for best config
- No SOTA reproduction (cite only)

### Ambitious (~900 GPU-h, needs 2 months)
- Full Phase C from ACL_EMNLP_PLAN.md
- Include Arctic-R1 reproduction at 3B
