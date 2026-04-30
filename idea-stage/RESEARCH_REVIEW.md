---
date: 2026-04-20
reviewer: GPT-5.4 xhigh (via Codex MCP)
score: 3/10 (current state)
thread: a7146f47854319e9f
---

# Senior Reviewer Critique (NeurIPS/ACL level)

## Summary of Concerns

| # | Issue | Severity |
|---|---|---|
| 1 | TDCA "causal" naming overclaims — it's heuristic weighting | HIGH |
| 2 | EWSG overlaps with GTPO/GRPO-S (generic token entropy) | HIGH |
| 3 | DAC uses oracle difficulty labels at test time | HIGH (methodological) |
| 4 | V5-100 **below SFT baseline** (EX_att 0.574 vs 0.610) | **CRITICAL** |
| 5 | Single seed + 200-sample subset = 11 extra examples advantage, no significance | CRITICAL |
| 6 | "Longer training hurts" claim confounded by 3.3× LR difference | HIGH |
| 7 | "Span" in EWSG name doesn't match per-token mechanism | MEDIUM |
| 8 | Missing SFT EX_all metric (looks selective) | MEDIUM |

## Ranked Concerns (from review)

1. **RL doesn't beat SFT** — "A paper proposing new RL training techniques that remains below the supervised baseline lacks a convincing core motivation"
2. **Evaluation is insufficient** — 200 samples, single seed, 5.4pp gap ≈ 11 examples
3. **Novelty overstated** — all 3 ideas are heuristic adaptations, not principled innovations
4. **Method names overclaim** — "causal", "span" will antagonize reviewers
5. **Degradation finding confounded** — can't distinguish "dense reward decoupling" from "bad LR schedule"

## Minimum Viable Experiment Package

From review (priority order):

| Priority | Experiment | GPU-h |
|---|---|---|
| 1 | **Full Spider-dev (1034)** eval: SFT, v4, v5-100, v5-300 ck75/ck150 | ~10h (eval only) |
| 2 | **2 extra seeds** for v4 + v5-100 = 4 more training runs | ~30h |
| 3 | **-TDCA, -EWSG ablations** (1 seed each) on full Spider-dev | ~15h |
| 4 | **Matched-LR long-training control** (v5-300 with v5-100's cosine) | ~25h |
| 5 | **BIRD-dev transfer test** (1 seed, best checkpoint) | ~5h |

**Minimum if compute-constrained**: 4 runs (2 extra seeds × 2 configs: v4, v5-100) = ~30h

## Recommended Narrative (Option D)

> "Turn-aware credit assignment for multi-turn Text-to-SQL agent RL, with evidence that dense reward optimization can decouple from execution quality under prolonged training."

**Changes needed:**
- Rename TDCA to avoid "causal" (e.g., "Turn-Weighted GRPO")
- Rename EWSG to avoid "span" (e.g., "Entropy-Weighted Token Loss")
- Move DAC to appendix (or remove) — oracle difficulty is fatal
- Frame reward-hacking observation as supporting analysis, NOT main finding
- Add matched-LR control

## Positive Signals

- Setting is meaningful (interactive multi-turn SQL agent)
- Per-difficulty breakdown provided
- Degradation observation could be publishable with proper controls

## Path to Acceptance

**Minimum bar (review quote):**
> "If these changes show a genuine improvement over the SFT baseline with statistical support, the paper becomes a credible empirical contribution to multi-turn agent RL for Text-to-SQL."

**Must-have:**
1. Full Spider-dev + SFT EX_all missing column filled
2. Multi-seed (≥3) on v4 vs v5-100 with paired t-test
3. Beat SFT on at least one primary metric
4. Matched-LR control for degradation claim
5. BIRD-dev transfer (even 1 seed)

## Decision Matrix for Phase C

Given single GPU + intermittent GRASS contention:

| Strategy | Time | Outcome |
|---|---|---|
| A. Do minimum 4 runs + full Spider eval | ~40h | Credible main result if v5-100 beats v4 with p<0.05 |
| B. Full ablation × 3 seeds (original plan) | ~400h | Ambitious, may not finish before ACL deadline |
| C. Pivot to different framing (degradation study) | ~100h | Requires matched-LR control + more analysis |
| D. Abandon v5 approach, rethink | — | — |

**Recommended: Strategy A** — minimum viable package, preserves option to expand.
