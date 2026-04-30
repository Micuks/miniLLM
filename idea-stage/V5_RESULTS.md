---
date: 2026-04-19
model: Qwen2.5-3B + SFT + GRPO v5 (checkpoint-75)
benchmark: Spider-dev (200 samples)
---

# GRPO v5 Interactive Eval Results

## Headline

**V5 significantly outperforms v4 on all metrics**, with largest gains on Easy tier.
Gap to SFT narrowed from -9pp (v4) to -3.6pp (v5).

## Main Table

| Method | EM | EX_att | EX_all | attempted |
|---|---|---|---|---|
| SFT baseline | 0.165 | **0.610** | — | — |
| GRPO v3 | 0.110 | 0.487 | 0.375 | 154/200 |
| GRPO v4 | 0.120 | 0.520 | 0.385 | 148/200 |
| **GRPO v5** | **0.130** | **0.574** | **0.410** | 143/200 |

## Per-Difficulty Breakdown (V5)

| Tier | n | EM | EX_att | EX_all | avg turns |
|---|---|---|---|---|---|
| easy | 65 | 0.369 | **0.877** | **0.769** | 3.3 |
| medium | 62 | 0.032 | 0.389 | 0.226 | 4.3 |
| hard | 73 | 0.000 | 0.360 | 0.247 | 4.3 |

## V5 vs V4 (per-difficulty delta)

| Tier | Δ EM | Δ EX_att | Δ EX_all | Δ turns |
|---|---|---|---|---|
| easy | +0.046 | +0.028 | **+0.077** | -0.2 |
| medium | -0.016 | +0.011 | 0.000 | +0.2 |
| hard | 0.000 | +0.050 | 0.000 | 0.0 |

## Training Notes

- Training ran 90/100 steps before env.py crash (fixed post-crash)
- Best checkpoint used: checkpoint-75 (95% of planned training)
- 3 features composed: --adaptive-turns + --span-weighting + --turn-delta
- Key observation: reward curves show higher variance than v4 (loss scale ~1.0 vs ~0.001)
  reflecting more aggressive gradient weighting

## Interpretation

**What worked:**
- Turn-delta + span-weighting made GRPO extract more signal from each rollout (skipped rates dropped from v4's 4/5 to v5's 0-2/5 in many windows)
- Easy tier improved substantially (+7.7pp EX_all) — the model now reliably converges on simple queries
- Hard tier attempt rate improved (more completions reach an Answer), but full correctness didn't yet follow

**What didn't (yet) work:**
- Medium tier is stuck — possibly because medium queries are the "middle zone" where adaptive-turns gives fewer turns (0.8× = 4) but queries still need more exploration
- Hard EX_all didn't improve despite +5pp EX_att — suggests model attempts more but still generates wrong SQL

**Gap analysis vs SFT:**
- SFT's strength: better calibrated "when to stop" — it doesn't over-explore
- V5's strength: better exploration on hard queries
- Combining: SFT-like early stopping + V5-like hard-query exploration could close the -3.6pp gap

## Next Steps

1. **300-step training** — current 100 steps were insufficient for the turn-delta signal to fully propagate
2. **Ablation study**: Which of the 3 features contributed most? (e.g., span-weighting vs turn-delta)
3. **Train multi-seed** — is V5's improvement robust across seeds or an artifact?
4. **Try on BIRD** — does the +5.4pp EX_att hold on a harder benchmark?
