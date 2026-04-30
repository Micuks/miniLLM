---
date: 2026-04-20
benchmark: Spider-dev 200 samples, interactive mode
---

# GRPO v5 Final Results & Lessons Learned

## Headline

**V5-100 (100 steps, aggressive LR decay) is the winner. Extending to 300 steps hurts performance due to LR scheduling and reward hacking.**

## Full Comparison Table

| Method | EM | EX_att | EX_all | attempted |
|---|---|---|---|---|
| SFT baseline | 0.165 | 0.610 | — | — |
| GRPO v3 | 0.110 | 0.487 | 0.375 | 154/200 |
| GRPO v4 | 0.120 | 0.520 | 0.385 | 148/200 |
| **GRPO v5-100 (winner)** | **0.130** | **0.574** | **0.410** | 143/200 |
| GRPO v5-300 ck75 | 0.105 | 0.442 | 0.345 | 156/200 |
| GRPO v5-300 ck150 | 0.100 | 0.448 | 0.320 | 143/200 |

## Per-Difficulty (V5-300 ck75 vs V5-100)

| Tier | v5-100 EX_att | v5-300 ck75 EX_att | Δ |
|---|---|---|---|
| easy | **0.877** | 0.709 | -16.8pp |
| medium | 0.389 | 0.269 | -12.0pp |
| hard | 0.360 | 0.327 | -3.3pp |

## Training Curve Analysis

The v5-300 training log showed:
- Step 5-20: rapid improvement (similar to v5-100)
- Step 25-75: peak corr 0.68 at step 75 checkpoint
- Step 120: training peak reward=0.89, corr=0.82 (!)
- Step 130-180: **reward hacking**: rewards cluster at 0.62-0.75 (well-formatted but wrong SQL), corr drops to 0.00-0.15
- Killed at step 180

Key finding: best *training* checkpoint was around step 120, but eval showed step 75 was better than step 150 (both worse than v5-100 at ck75).

## Root Cause: LR Scheduling Mismatch

| | v5-100 (100 steps) | v5-300 (300 steps) |
|---|---|---|
| LR at 25% progress | 2.68e-6 | 2.68e-6 |
| LR at 50% progress | 1.62e-6 | **2.89e-6** |
| LR at 75% progress | 0.82e-6 | **2.10e-6** |
| LR at step=75 | **0.82e-6** | **2.68e-6** |

V5-300 ck75 had 3.3× higher LR than v5-100 ck75. Policy at v5-300 ck75 is still aggressively updating, not settled. By the time LR decays (step 225+), reward hacking has already set in.

## Reward Hacking Pattern (v5-300)

Starting step 130-180:
```
rewards=[0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.03, 0.75]  ← 7/8 get same ~0.75
rewards=[0.62×8]  ← all 8 converge to identical reward
rewards=[0.66×6, 0.2, 0.67]  ← tight cluster, fmt=0.59, corr=0.00
```

**Interpretation:** Model learned to exploit the dense reward — produce perfect format + partially-correct SQL template = steady 0.62-0.75. This is reward-hacking on our "safe reward floor".

The compound effect of:
- span-weighting (×2 on high-entropy tokens)
- turn-delta (×1.5 on Answer tokens)
- Long sustained high LR (×3 vs v5-100)

produces unstable updates that exploit the reward structure instead of solving SQL.

## Recommended Fixes for Future Work

1. **Linear warmup + step decay** (instead of cosine): LR peak for 10% then drop to 0.2×, 0.5×, etc. at milestones
2. **Early stopping**: monitor rolling corr; stop when corr hasn't improved in K=20 windows
3. **Lower KL tolerance at peak LR**: increase KL coef during peak LR period
4. **Reward structure fix**: cap the sum of format+validity+structure to prevent safe-reward floor > 0.5

## For ACL/EMNLP Submission

**Current positioning:**
- V5-100 achieves 57.4% EX_att on Spider interactive = best so far
- Gap to SFT: -3.6pp (v4 had -9pp)
- Gap to Arctic-R1-7B: ~22pp (but at 3B scale)

**What we can claim:**
- 3 new GRPO techniques (turn-delta + span-weighting + adaptive-turns) improve multi-turn ReAct agents
- At 3B: v5 > v4 > v3, narrowing gap to SFT
- Training efficiency: 100 steps sufficient, longer hurts (novel observation)

**Ablation still needed (Phase C):**
- Individual feature contributions (v5-adaptive, v5-span, v5-turn alone)
- 3 seeds per config
- Both Spider and BIRD benchmarks
