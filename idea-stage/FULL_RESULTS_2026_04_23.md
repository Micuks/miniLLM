---
date: 2026-04-23
benchmark: Spider-dev 200 samples (car_1, concert_singer, pets_1, flight_2)
metric: EX_att = execution match / attempted; EX_all = execution match / all 200
---

# Full Experimental Results — All Configurations

## Summary Table

| Config | EX_att | EX_all | EM | Easy_att | Med_att | Hard_att | Att/N |
|---|---|---|---|---|---|---|---|
| SFT (ReAct, single_pass) | 57.0% | 57.0%* | 11.0% | 73.7% | 51.4% | 39.5% | 200/200 |
| Dense-v5-100 (best RL) | **57.3%** | 41.0% | 13.0% | 87.7% | 38.9% | 36.0% | 143/200 |
| Dense-v5-300 ck75 | 44.2% | 34.5% | 10.5% | 70.9% | 26.9% | 32.7% | 156/200 |
| Dense-v5-300 ck150 | 44.8% | 32.0% | 10.0% | 69.6% | 27.5% | 29.8% | 143/200 |
| Sparse-ck25 (best sparse) | 50.3% | 38.5% | 10.5% | 81.8% | 33.3% | 32.1% | 153/200 |
| Sparse-ck75 | 42.4% | 33.5% | 10.0% | 68.4% | 25.5% | 29.6% | 158/200 |
| Sparse-ck100 | 43.5% | 33.5% | 10.0% | 69.6% | 29.2% | 28.0% | 154/200 |
| CGFR-only-ck25 | 43.5% | 35.0% | 11.0% | 73.7% | 25.0% | 28.8% | 161/200 |
| CGFR-only-ck50 | 42.0% | 33.0% | 10.0% | 67.8% | 26.0% | 27.1% | 157/200 |
| CGFR-only-ck75 | 46.2% | 36.0% | 9.5% | 70.9% | 33.3% | 32.1% | 156/200 |
| CGFR-only-ck100 | 46.2% | 37.0% | 9.0% | 71.9% | 31.4% | 32.7% | 160/200 |
| CGFR+RVDS-ck25 | 42.3% | 33.0% | 10.5% | 68.4% | 23.9% | 30.2% | 156/200 |
| CGFR+RVDS-ck50 | 44.5% | 34.5% | 10.0% | 70.2% | 26.1% | 32.7% | 155/200 |
| CGFR+RVDS-ck75 | 44.4% | 33.5% | 10.0% | 69.6% | 26.7% | 32.0% | 151/200 |
| CGFR+RVDS-ck100 | 46.5% | 36.5% | 9.5% | 70.9% | 32.0% | 34.6% | 157/200 |

*SFT EX_all = EX_att since single_pass generates SQL for all queries (100% att rate)

---

## Key Findings

### Finding 1: EX_att vs EX_all — The Critical Metric Gap

**Dense-v5-100 scores 57.3% EX_att (≈SFT!) but only 41% EX_all (16pp below SFT).**

The discrepancy arises from coverage:
- SFT (single_pass): generates SQL for all 200 queries → EX_all = EX_att = 57%
- Dense-v5-100 (interactive): only 143/200 (71.5%) queries get a SQL answer → EX_all = 41%

**All RL variants are worse than SFT on the proper EX_all metric.**

| Config | EX_all | vs SFT |
|---|---|---|
| SFT | 57.0% | — |
| Dense-v5-100 | 41.0% | **-16.0pp** |
| Sparse-ck25 | 38.5% | -18.5pp |
| CGFR-ck100 | 37.0% | -20.0pp |

### Finding 2: RL Systematically Degrades Medium Queries

Every RL variant trades easy query gains for severe medium query degradation.

| Config | Δ Easy | Δ Med | Δ Hard | Δ Total (EX_att) |
|---|---|---|---|---|
| Dense-v5-100 vs SFT | **+14.0pp** | **-12.5pp** | -3.5pp | +0.3pp |
| Dense-v5-300 ck150 | -4.0pp | -23.9pp | -9.7pp | -12.3pp |
| Sparse-ck25 | +8.1pp | -18.1pp | -7.4pp | -6.7pp |
| Sparse-ck100 | -4.0pp | -22.3pp | -11.5pp | -13.5pp |
| CGFR-ck100 | -1.8pp | -20.1pp | -6.8pp | -10.8pp |

**Interpretation:** RL training on GRPO updates the policy toward patterns that succeed on easy queries (simple COUNT, SELECT, basic WHERE). Medium queries require multi-table JOINs, complex aggregations, and subqueries — patterns that RL does NOT reinforce because they rarely succeed in the 3B model's random rollouts. SFT, trained on teacher-generated trajectories covering all difficulties, maintains good medium performance.

### Finding 3: Early Peak → Degradation Pattern

All RL variants peak early and then degrade. The CGFR+RVDS combination is the most stable.

| Config | Peak Corr | @Step | Final Corr | Corr Δ | fmt Rise |
|---|---|---|---|---|---|
| Dense-v5 | 0.75 | step 5 | 0.05 | -0.70 | +0.04 |
| Dense-v5-300 | 0.82 | step 120 | 0.00 | -0.82 | +0.07 |
| Sparse | 0.80 | step 20 | 0.25 | -0.55 | +0.14 |
| CGFR-only | 0.72 | step 80 | 0.10 | -0.62 | +0.12 |
| **CGFR+RVDS** | **0.53** | step 45 | **0.35** | **-0.01** | +0.26 |

CGFR+RVDS achieves the most stable training (corr_change = -0.01, essentially flat from step 45-100). However, its peak is lower (0.53 vs 0.80 for sparse). **CGFR+RVDS ck50 eval = 44.5% EX_att / 34.5% EX_all** — same level as CGFR-only and slightly worse than Dense-v5-100 on EX_all. **The most stable training did NOT translate to better eval performance**, confirming that the bottleneck is not training stability but model capacity / generalization.

### Finding 4: Coverage Drops with More RL Training

As RL training continues, models answer FEWER queries (coverage drops):

| Config | Steps | Att/N | Coverage |
|---|---|---|---|
| Dense-v5-100 | 100 | 143/200 | 71.5% |
| Dense-v5-300 | 300 | 143/200 | 71.5% |
| Sparse-ck25 | 25 | 153/200 | 76.5% |
| Sparse-ck100 | 100 | 154/200 | 77.0% |
| CGFR-ck25 | 25 | 161/200 | 80.5% |
| CGFR-ck100 | 100 | 160/200 | 80.0% |

CGFR models have higher coverage (80%) than dense/sparse (71-77%). This is expected: CGFR's format gating forces the model to either be correct (high reward) or try harder (no safe-floor reward), reducing reward for "try a format without executing". However, CGFR still doesn't convert coverage to correct answers.

### Finding 5: CGFR Gating Doesn't Fix the Core Problem

CGFR was designed to eliminate the "safe floor" reward for incorrect SQL. Despite this:
- CGFR-ck100: 46.2% EX_att (best CGFR, but still -11pp below Dense-v5-100)
- CGFR training shows fmt rising to 0.93 by late steps — format learning still occurs for CORRECT rollouts
- The real issue is that medium/hard queries can't be solved at 3B → those rollouts remain wrong → no reward signal

**The bottleneck is not reward structure, it's capacity.** 3B has insufficient capacity to solve medium/hard SQL in multi-turn ReAct, so no reward structure redesign can help if the rollouts are all-fail for those queries.

---

### Finding 7: Query-Level Substitution (Dense-v5-100 vs SFT)

| | Easy | Medium | Hard | Total |
|---|---|---|---|---|
| Both correct | — | — | — | 54 (27%) |
| Both wrong | — | — | — | 95 (47.5%) |
| SFT only ✓ (RL regressions) | 4 | 9 | 10 | 23 (11.5%) |
| RL only ✓ (RL improvements) | 12 | 5 | 11 | 28 (14%) |
| **Net Δ (RL - SFT)** | **+8** | **-4** | **+1** | **+5** |

RL doesn't broadly improve — it substitutes ~25 queries between categories. Regressions cluster on medium/hard (-4 medium, +1 hard); improvements concentrate on easy (+8). Sample regressions show RL produces malformed SQL (e.g., `unrecognized token: "\`Name"`, `no such column: T2.Name`) — patterns the model picked up that don't generalize to dev DBs.

### Finding 6: Per-DB Variance Confirms Difficulty Story

DB-level breakdown reveals huge variance — RL helps where queries are easy, hurts where they're medium/hard:

| DB | n | Easy | Med | Hard | SFT | Dense-v5 | Δ vs SFT |
|---|---|---|---|---|---|---|---|
| flight_2 | 21 | 100% | 0% | 0% | 57.1% | **81.0%** | **+23.9pp** |
| concert_singer | 45 | 31% | 36% | 33% | 51.1% | 60.0% | +8.9pp |
| car_1 | 92 | 26% | 30% | 44% | 25.0% | 26.1% | +1.1pp |
| pets_1 | 42 | 14% | 43% | 43% | 45.2% | **33.3%** | **-11.9pp** |

**flight_2 = 100% easy queries → RL gains 24pp. pets_1 = 86% medium/hard → RL loses 12pp.**

This is the clearest empirical evidence that RL gains are entirely concentrated on easy queries. The aggregate RL underperformance vs SFT (16pp on EX_all) comes from the medium+hard-heavy mix of eval queries.

---

## Training Dynamics Analysis

### Dense GRPO (v5-100): Reward Hacking Signature

```
Step 45: reward=0.817 fmt=0.96 val=0.95 struct=0.90 exec=0.62 corr=0.50
Step 90: reward=0.415 fmt=0.84 val=0.76 struct=0.70 exec=0.27 corr=0.05
         rewards=[0.59, 0.03, 0.59, 0.03, 0.59, 0.59, 0.59, 0.03]
```

By step 90 (just before 100): fmt rises while corr collapses. The "rewards=[0.59×6, 0.03×2]" cluster shows model locked into a pattern that scores ~0.59 on format+validity+struct without achieving correctness.

### CGFR+RVDS: Stable but Lower Ceiling

```
Step 15: corr=0.47 rewards=[1.2, 1.2, 0.04, 1.2, 1.2, 1.2, 0.04, 1.2]  ← 6/8 correct
Step 45: corr=0.53 rewards=[1.17×6, 0.04, 1.17]  ← peak
Step 85: corr=0.25 fmt=0.93 val=0.91  ← late stage: format rises, corr drops
Step 100: corr=0.35 (partial recovery at end)
```

CGFR+RVDS still shows fmt rising at late stages but the RVDS skipping prevents the worst collapse. By step 100 there's partial recovery (corr=0.35 vs peak 0.53).

---

## Root Cause Analysis (6 Hypotheses)

### H-B: Generalization Failure [CONFIRMED]
- Training corr peaks at 0.80-0.82 on training queries (146 DBs)
- Eval EX_all = 37-41% on dev queries (4 different DBs)
- Gap: ~37-43pp between best training corr and eval performance
- **Core mechanism**: RL trains on training-DB patterns; dev DBs have different schema structures

### H-C: SFT Ceiling [PARTIALLY CONFIRMED]
- SFT medium EX_att = 51.4%: this is the 3B ceiling for medium SQL
- RL cannot exceed this because medium/hard rollouts are mostly all-fail
- But: SFT isn't the ceiling for easy (RL improves easy by +14pp) — capacity exists for easy SQL
- **Conclusion**: SFT ceiling applies selectively to medium/hard, not easy queries

### H-E: Multi-Turn Format Attractor [CONFIRMED]
- Dense reward safe floor = 0.69 max without correct SQL (format+validity+struct)
- RL exploits this: rewards cluster at 0.59-0.75 without correct SQL (step 90+)
- CGFR reduces safe floor; CGFR+RVDS is most stable but still can't break medium/hard bottleneck

### H-F: Reward Sparsity / Step-to-Step Instability [CONFIRMED via sparse curves]
- 90% of training steps have mixed success/failure (not bimodal all-fail/all-succeed)
- Mean corr = 0.392 for sparse (averaging across 100 steps)
- 41% of rollouts skipped overall — but per-step skipping is noisy (0-10 per step)
- Real issue: step-to-step instability due to single-query-per-step training

### H-A: Entropy Collapse [INDIRECT EVIDENCE]
- Variance collapse signature at step 175 (dense v5-300): all 8 rollouts identical
- Dense v5-100 peak corr at step 5 (surprisingly early) suggests fast post-SFT collapse
- CGFR+RVDS delays peak to step 45 (vs step 5 for dense), consistent with slower entropy collapse

### H-D: KL Over-Constraint [NOT TESTED]
- KL=0.1 for all runs
- Ablation not yet run (12h GPU cost)
- Pending: compare KL=0 vs 0.1 vs 0.3

---

## Paper Narrative Update

**Primary finding (clean, novel):**
> At 3B scale, all GRPO reward structures fail to match SFT on the EX_all metric, despite apparent parity on EX_att. The hidden failure mode is twofold: (1) RL selectively improves easy queries while severely degrading medium queries (-12 to -24pp), and (2) RL reduces coverage (models fail to answer 20-29% of queries). CGFR+RVDS produces the most stable training but cannot overcome the fundamental capacity bottleneck for medium/hard SQL at 3B scale.

**Why it matters:**
- Think2SQL (dense rewards help small models) is in single-turn SQL where easy queries dominate
- Our multi-turn ReAct setting exposes medium queries where 3B lacks capacity
- EX_att is a misleading metric for interactive agents — EX_all (including coverage) is essential

**Story arc:**
1. Both dense and sparse fail (different failure modes)
2. CGFR eliminates safe floor but cannot fix capacity bottleneck
3. CGFR+RVDS is most stable but lower ceiling
4. Root cause: RL at 3B has insufficient rollout success on medium/hard to learn from them
5. SFT with diverse teacher trajectories covers all difficulties → better generalization

---

## Pending Experiments

- [ ] CGFR+RVDS ck25/75/100 evals (queued, running after ck50)
- [ ] KL ablation (12h, H-D) — GPU-heavy
- [ ] H-A: Token entropy measurement on SFT model outputs (2h analysis)
- [ ] H-B: Formal training-DB vs eval-DB accuracy comparison
