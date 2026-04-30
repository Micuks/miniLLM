---
date: 2026-04-23
status: archived — 3B experiments paused, pivoting to 7B
---

# 3B Experiments Archive (Spider 200 Interactive ReAct)

## Summary (EX_att)

**Baseline**: ReAct-SFT v1 single_pass = 57.0% (Easy 73.7%, Med 51.4%, Hard 39.5%)

| Config | Best ckpt | Overall | Easy | Med | Hard | Notes |
|---|---|---|---|---|---|---|
| Dense v5 (original) | ck300 | ~57% | — | — | — | reward hacks at ck175 |
| **Sparse** | ck25 | 50.3% | 81.8% | 33.3% | 32.1% | only config that lifts easy vs SFT |
| CGFR | ck100 | 46.2% | 71.9% | 31.4% | 32.7% | no hack, stable decay |
| **CGFR+RVDS** | ck100 | **46.5%** | 70.9% | 32.0% | 34.6% | best medium+hard of RL variants |
| DAPO (killed step 35) | ck25 | — | — | — | — | unfinished, archived |

Per-difficulty detail: see `PER_DIFFICULTY_DIAGNOSTIC_3B.md`.

## Key Findings

1. **No 3B RL config beats SFT** (max RL = 50.3% vs SFT 57.0%)
2. **Medium queries consistently fail**: all RL configs -18 to -28 pp vs SFT on medium
3. **Easy gains require early checkpoint**: sparse_ck25 (+8.1) unique; later checkpoints regress
4. **Reward engineering (CGFR, RVDS) did not rescue**: same medium-query failure across dense/sparse/gated/filtered rewards
5. **Implication**: likely a capacity limit, not a reward-shape problem

## Open Question: Strategy or Scale?

If our recipe (CGFR+RVDS, DAPO-style) applied to **7B** lifts medium queries above 7B SFT:
→ **Capacity hypothesis confirmed**. Paper: "at-3B negative + at-7B positive, same recipe"

If 7B still fails on medium:
→ **Structural problem in multi-turn ReAct**. Paper: "RL over multi-turn ReAct agents has a fundamental medium-query failure mode"

Both outcomes publishable.

## Compute Budget Spent

- Dense v5-100/300: ~12h
- Sparse 100 steps: ~12h (hit env.py crash at step 90)
- CGFR 100 steps: ~10h
- CGFR+RVDS 100 steps: ~16h (RVDS skipping slowed training)
- DAPO 3B (partial, 35 steps): ~3h
- All Spider 200 evals: ~8h total
- **Total 3B spend: ~60 GPU-h**

## Pivot

Paused 3B. Next: **7B SFT (QLoRA bnb4) → 7B Dense GRPO → per-difficulty eval**. Same pipeline.
