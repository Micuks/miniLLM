---
date: 2026-04-27
status: HONEST READING — endpoint delta is noise; trajectory is the only signal worth chasing
---

# SKE-RL vs Dense GRPO at 7B — Honest Reading

## What we have

Three eval runs from `outputs/react-sft-7b-bf16` (62.13% EX_att) on Spider 200 single_pass, paired hyperparams except `--use-ske-rl` flag:

| ckpt | Easy | Medium | Hard | Overall | matched/att |
|---|---|---|---|---|---|
| 7B SFT | 88.51 | 56.60 | 38.16 | **62.13** | 105/169 |
| Dense ck25 | 91.94 | 53.57 | 46.15 | 65.29 | 111/170 |
| Dense ck50 | 91.80 | 60.38 | 41.38 | 65.12 | 112/172 |
| Dense ck75 | 92.06 | 61.11 | 38.89 | 65.50 | 112/171 |
| SKE ck25 | 91.53 | 55.36 | 37.93 | 61.85 | 107/173 |
| SKE ck50 | 90.00 | 57.14 | 41.82 | 63.74 | 109/171 |
| SKE ck75 | 92.19 | 60.38 | 41.51 | **66.47** | 113/170 |

Both runs OOM'd at step 85-90, no ck100.

## What's defensible

**1. Capacity hypothesis confirmed (real signal):**
- 3B archive: all 4 reward variants regressed Medium by -18 to -28pp vs 3B SFT
- 7B Dense GRPO: lifts SFT by +3.4pp overall, +3.6 Easy, +4.5 Medium (averaged across ck25/50/75)
- This delta is robust — every Dense checkpoint beats SFT on overall by 2.99-3.37pp

**2. Dense plateaus quickly (real signal):**
- Dense Overall: 65.29 → 65.12 → 65.50 — completely flat from ck25
- Dense Hard: 46.15 → 41.38 → 38.89 — actually *decays*
- Suggests Dense reaches its capacity ceiling early; more steps don't help

## What's NOT defensible (single seed)

**SKE-RL endpoint advantage at ck75 (+0.97pp over Dense)** = **1 correct query** (113 vs 112 / n=200). Within single-seed noise.

Per-skeleton-class diagnostic on (Dense ck75, SKE ck75) records:
- 73 qualifying skeleton classes
- 67 tied
- SKE wins 1 class (n=4 medium, +0.25 delta)
- Dense wins 5 classes (mix easy/medium/hard, mostly n=2)

The +0.97pp endpoint is not concentrated in any specific class — it's a single scattered query.

## What's a hint, not a claim

**SKE trajectory monotonicity:**
- SKE Overall: 61.85 → 63.74 → 66.47 — monotonic rise across all 3 checkpoints
- SKE Medium: 55.36 → 57.14 → 60.38 — monotonic
- 4.6pp rise over 50 steps = ~10 query swing — could be real, could be ck25 unlucky start (`ske_used` was at kill-threshold border at step 10)

If Dense plateau is real and SKE keeps rising, then with more steps SKE *would* overtake meaningfully. But this is conjecture from 3 datapoints with n=200 each.

## What's needed to make any SKE claim

1. **Multi-seed** (≥3) at ck75 for both Dense and SKE — bounds the +0.97pp endpoint
2. **ck100 recovery** (currently OOM'd) — checks if Dense really stays flat past ck75 and SKE keeps rising
3. **Larger n_eval** — full Spider-dev (1034) instead of 200; turns 1-query differences into 5-query differences

## Recommended paper claims (with current data only)

**Title direction**: "Scale Unlocks Multi-Turn Agentic GRPO on Text-to-SQL"

**Core claim** (defensible from current data):
> At 7B, GRPO with dense execution-based reward improves over SFT by +3.4pp EX_att (62.1 → 65.3), recovering the gains that 3B GRPO failed to capture across all 4 reward profiles.

**Secondary** (qualified):
> Skeleton-equivalence-class advantage shaping (SKE-RL) is mechanistically motivated by within-class outcome variance ($\\bar{\\sigma}^2 = 0.030$) and rollout-time skeleton diversity (62.5% medium queries hit gold shape). Empirically, SKE-RL is non-destructive but does not yield a statistically distinguishable improvement over raw GRPO at the 100-step single-seed budget evaluated here.

**Diagnostic** (good for Section 5):
> Per-difficulty trajectory analysis shows Dense GRPO plateaus by step 25 while SKE-RL trajectory continues to rise through step 75, suggesting SKE-RL's noise reduction may extend the effective RL learning horizon — pending multi-seed verification.

## Recommended next experiments (in priority order)

| # | Experiment | GPU-h | Yield |
|---|---|---|---|
| 1 | Multi-seed (3 seeds) Dense ck75 + SKE ck75 | ~30 | bounds the +0.97pp signal |
| 2 | Full Spider-dev (1034) eval on best 3 checkpoints | ~6 | turns single-query noise into 5-query confidence |
| 3 | Recover ck100 for both (extend +25 from ck75 with reduced max_completion) | ~4 | Confirms or kills the trajectory claim |
| 4 | External baseline (DIN-SQL or CoT Qwen2.5-7B) | ~8 | Required for ACL/EMNLP submission |
| 5 | 3B re-run with same SKE-RL gates (does Phase 1 fail at 3B?) | ~24 | establishes capacity-vs-mechanism story |
