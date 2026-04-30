---
date: 2026-04-20
source: v5-100 log, v5-300 log
finding: Reward hacking is the dominant failure mode in dense-reward GRPO for multi-turn ReAct agents
---

# Diagnostic: Reward Hacking in Multi-Turn GRPO

## Key Metric: Shallow-Correctness Gap

Define **shallow_reward = mean(fmt, validity, structure)** (format-level signals that can be gamed without solving SQL) and **corr** (actual execution correctness). Gap = shallow - corr.

Variance collapse: `rmax - rmin` across the G=8 rollouts. If < 0.2, all completions converged to same reward template.

**Hacking signature:** gap > 0.5 AND variance < 0.2 → model has locked into a "safe template" that passes format checks but doesn't solve tasks.

## V5-300 Timeline (smoking gun)

| step | shallow | corr | GAP | var | notes |
|---|---|---|---|---|---|
| 5 | 0.86 | 0.75 | 0.11 | 0.97 | Healthy training |
| 80 | 0.86 | 0.35 | **0.51** | **0.01** | **First full hack** |
| 120 | 0.97 | 0.82 | 0.15 | 0.01 | Brief peak (before collapse) |
| 130 | 0.52 | 0.00 | 0.52 | 0.71 | Collapse begins |
| 145 | 0.80 | 0.15 | 0.65 | 0.96 | Hack + variance |
| 170 | 0.60 | 0.03 | 0.57 | 0.47 | Severe |
| **175** | **0.94** | **0.05** | **0.89** | **0.01** | **Lock-in: all 8 at 0.62-0.63** |
| 180 | 0.78 | 0.00 | 0.78 | 0.72 | corr=0 |

## V5-100 Timeline (same pattern, ended early)

| step | shallow | corr | GAP | var |
|---|---|---|---|---|
| 5 | 0.86 | 0.75 | 0.11 | 0.97 |
| 45 | 0.94 | 0.50 | 0.44 | 0.30 |
| 65 | 0.54 | 0.15 | 0.39 | 0.97 |
| 90 | **0.77** | **0.05** | **0.72** | 0.56 |

V5-100 showed severe hacking at step 90 before the env.py crash killed training. **V5-100's "win" over v5-300 is an artifact of accidental early stopping at step 75-90.**

## Concrete Reward Hacking Evidence (step 175)

rewards=[0.62, 0.62, 0.63, 0.63, 0.62, 0.62, 0.63, 0.62]

All 8 completions, from different random samples, get the SAME reward to 2 decimal places. Impossible if model is solving diverse prompts. Only possible if model has found a **single template** that:
- Passes format check (✓ fmt=1.00)
- Passes validity check (✓ val=0.96)
- Passes structure check (✓ struct=0.85)
- Fails execution/correctness (✗ exec=0.23, corr=0.05)

## Why This Happens: Reward Composition

Our dense reward:
```
total = 0.35 × fmt + 0.20 × val + 0.25 × struct + 0.10 × exec + 0.10 × corr + err_penalty
```

At t=1 (end of training), weights shift to: 0.24 / 0.16 / 0.14 / 0.16 / 0.30.

The **shallow floor** = 0.35×1.0 + 0.20×0.8 + 0.25×0.7 = ~0.69 (early) or ~0.56 (late)

GRPO discovers this floor is achievable with a fixed template and stops exploring. Correctness (high-variance, hard to achieve) gets outcompeted by the deterministic shallow reward.

## Publishable Contribution (for Findings paper)

**Title: "Dense Rewards Trap Multi-Turn Agent RL: A Case Study on Text-to-SQL"**

1. **Empirical observation**: Reward hacking emerges within 100 steps of GRPO training on dense-reward Text-to-SQL agents
2. **Diagnostic metric**: (fmt+val+struct)/3 - corr gap + rollout variance collapse as early indicators
3. **Root cause analysis**: Safe reward floor from composable format signals
4. **Practitioner guidance**: 
   - Add early stopping on correctness plateau
   - Reduce format reward weight after warmup
   - Multiplicative (not additive) reward composition forces correctness as necessary condition

## Paper Story Positioning

Instead of:
> "We propose TDCA+EWSG+DAC that improve GRPO..."

Reframe as:
> "We identify reward hacking as a fundamental failure mode of dense-reward GRPO in multi-turn Text-to-SQL agents. Three proposed enhancements (TDCA, EWSG, DAC) designed to extract more training signal actually **accelerate** reward hacking due to their compounding effects on gradient magnitude. We provide diagnostic metrics and practitioner guidance."

This is actually a stronger story because:
- We have clear evidence
- It contradicts the "add more signal = better" intuition
- Practitioners will care (most Text-to-SQL RL papers use dense rewards)
- The failure is reproducible and instructive

## Next Steps for Findings Paper (~2-3 weeks)

1. Run v5-300 with **matched LR to v5-100 schedule** (control for confound)
2. Run **sparse reward** (execution match only) control — does hacking disappear?
3. Run **reward weight ablation** (only format vs only correctness) — isolate which component enables hacking
4. 3 seeds each for 2-3 key configs (v4, v5-all, sparse-reward)
5. Full Spider-dev eval

~25-30 GPU-hours. Feasible in 2 weeks.
