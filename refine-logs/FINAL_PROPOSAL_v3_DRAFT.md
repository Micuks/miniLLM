# Research Proposal v3 (DRAFT, R2-revised) — Stopping Policy vs Conditional Solver

> ⚠️ **STATUS: SUPERSEDED by `FINAL_PROPOSAL_v4.md` (2026-04-30)**.
>
> v3's "SKE-specific" framing was empirically **falsified** by the B4 Dense
> G=8 confound run also completed on 2026-04-30. Result:
> Dense G=8 reproduces SKE-G8's commitment shift AND per-turn regression
> (`per_turn_sym = −1.85 [−3.69, −0.13] ✗`), so the mechanism is generic
> to group size G, not SKE-specific. See
> `B4_RESULT_v3_PREDICTION_FALSIFIED_2026-04-30.md` and
> `FINAL_PROPOSAL_v4.md` for the live thesis.
>
> This document is **kept verbatim** (with R2 revisions: Shapley-sym
> primary, R_same paired analysis, honest-pivot framing) as the audit
> trail of (a) the v3 thesis we predicted, and (b) the round-2 reviewer-
> required robustness work that carries forward to v4. The Shapley-sym
> decomposition formulas, the R_same composition-decontamination check,
> and the honest-pivot framing language all remain valid and were applied
> to v4's 5-condition data in `V4_ROBUSTNESS_PACK_2026-04-30.md`.
>
> **Do not cite this file as the live proposal.** Cite `FINAL_PROPOSAL_v4.md`.

---

**Original status**: DRAFT — supersedes `FINAL_PROPOSAL.md` (R4 9.00/10) on
the basis that v2's preregistered protocol-vs-semantic axis was empirically
falsified on this dataset. Round-2 reviewer fixes (Shapley-sym decomposition
primary, R_same paired analysis, honest-pivot framing) applied 2026-04-30.

**Date**: 2026-04-30 (R2 revision)
**Original v3 draft**: 2026-04-29
**Prerequisite reads**:
- `A0_PRE_AUDIT_FINDING_2026-04-29.md` — v2 axis falsification.
- `A0_OPTION_2_3_SPIKE_2026-04-29.md` — Option-2 exec-aware lenient rejected.
- `TURN_DECOMP_FINDING_2026-04-29.md` — Option-3 results that motivate v3.
- `RESEARCH_REVIEW_v3_2026-04-29.md` — round-1+2 reviewer findings.
- `MAIN_CONF_PLAN_2026-04-30.md` — round-3 main-conf plausibility analysis.
- `outputs/robustness_pack.json` — A/B/sym + R_same + paired-ΔEX numbers.

---

## Problem Anchor (immutable from v2)

Multi-turn ReAct GRPO succeeds modestly at 7B (+2.32 pp full Spider-dev),
fails at 3B; SKE-RL fails at G=4 / G=8. The unexplained part remains: what
does the +2 pp actually change inside the agent, and why does SKE-RL not
help?

## What v2 got wrong (preregistered hypothesis falsified)

v2 hypothesized **protocol-vs-semantic** decomposition: "agentic GRPO mostly
cleans up trajectory format, not SQL semantics". After implementing the
leakage-safe lenient extractor and running it against all 4 locked eval
JSONs, strict and lenient extractors returned the **same SQL string in
≥99.9% of records** (1034/1034 SFT, 1033/1034 Dense, 1034/1034 SKE-G4,
1033/1034 SKE-G8). The `protocol_gain` bucket is empty by construction.
This falsifies the v2 thesis.

The analysis below is **exploratory**, not confirmatory of v2. We do not
borrow the credibility of v2's preregistration to imply v3 is confirmatory.

## What v3 does instead (turn-progression decomposition)

Agentic GRPO improves aggregate EX through two operationally distinct
mechanisms that are invisible to aggregate metrics alone:

1. **Conditional solver capability**: at fixed turn count, the model's SQL
   is more often correct.
2. **Stopping policy shift**: the model resolves more queries in a single
   turn (where SFT's accuracy is already higher) without improving accuracy
   at any individual turn count.

Decompose `total_ΔEX` accordingly:

```
total_ΔEX = share_shift_term + per_turn_gain_term       (exact, per resample)
```

**Shapley-symmetric average** is the **primary** reported quantity (binary
turn binning `k ∈ {1, 2+}`):

```
share_shift_sym = (s_T - s_B) · ((b1 - b2) + (t1 - t2)) / 2
per_turn_sym    = ((s_B + s_T)/2) · (t1 - b1)
                + (1 - (s_B + s_T)/2) · (t2 - b2)
```

Ordering A (baseline-weighted share-shift) and Ordering B (treatment-weighted
share-shift) are reported in the appendix as a sensitivity check. The
qualitative conclusion is invariant to ordering choice.

**R_same paired ΔEX**: paired execution-match difference restricted to
records where `turn_T(i) = turn_B(i)`. Within R_same, per-turn case mix is
held fixed by construction; a non-zero R_same delta cannot be a composition
artifact of share-shift.

## Empirical result on existing data (Spider-dev 1034)

Bootstrap 1000 resamples, paired by record id, percentile-95 CIs.
✓ = CI excludes 0 positive · ✗ = CI excludes 0 negative.

### Aggregate decomposition (Shapley-symmetric primary)

| Pair | total ΔEX | share_shift_sym | per_turn_sym | R_same paired ΔEX |
|---|---|---|---|---|
| **Dense75 − SFT** | +2.32 [+0.39, +4.35] ✓ | +0.83 [+0.07, +1.67] ✓ | +1.49 [−0.51, +3.66] | **+2.22 [+0.37, +4.06] ✓** |
| SKE-G4-75 − SFT | +1.64 [−0.39, +3.87] | +1.20 [+0.45, +1.98] ✓ | +0.44 [−1.87, +2.70] | **+2.47 [+0.37, +4.57] ✓** |
| **SKE-G8-75 − SFT** | +0.39 [−1.06, +1.94] | **+2.52 [+1.71, +3.50] ✓** | **−2.13 [−3.87, −0.46] ✗** | +0.78 [−0.56, +2.22] (overlaps 0) |

### Decomposition-ordering sensitivity (Appendix)

| Pair | share_A | share_B | share_sym | A−B gap |
|---|---:|---:|---:|---:|
| Dense G=4 | +0.90 | +0.77 | +0.83 | 0.13 pp |
| SKE G=4 | +1.32 | +1.08 | +1.20 | 0.24 pp |
| SKE G=8 | +2.64 | +2.40 | +2.52 | 0.25 pp |

A and B agree on sign, qualitative ranking, and CI-relative-to-zero in every
pair. The qualitative conclusion is invariant to decomposition ordering.

### Per-difficulty Shapley-sym decomposition (Easy=333, Medium=377, Hard=324)

| Pair | Easy | Medium | Hard |
|---|---|---|---|
| Dense − SFT | total +1.50; share +0.66; per_turn +0.84 | **total +3.98 ✓; share +0.68; per_turn +3.30 ✓** | total +1.23; per_turn +0.86 (no sig) |
| SKE-G4 − SFT | total +1.20 (no sig) | total +1.86 (no sig) | share +0.85 ✓; per_turn +1.00 |
| SKE-G8 − SFT | total +1.80 ✓; share +2.03 ✓; per_turn ~0 | share +1.25 ✓; per_turn −0.72 | total −1.23; per_turn **−3.38** (CI just overlaps 0 at +0.16) |

## Method Thesis (one sentence — honest-pivot framing)

> After a preregistered extractor hypothesis failed, we show that agentic
> GRPO's Spider gains arise mainly from earlier commitment, while SKE-RL
> amplifies this shift without improving conditional SQL accuracy.

## Refined narrative (R_same disambiguates the per-turn finding)

The combination of (a) `per_turn_sym` 95% CI excluding 0 in the negative
direction for SKE-G8, AND (b) `R_same` paired ΔEX 95% CI **overlapping 0**
for SKE-G8, is more informative than either result alone:

- SKE-G8 does **not** show evidence of conditional SQL improvement on
  records where its turn count matched SFT's (R_same paired ΔEX = +0.78
  [−0.56, +2.22]).
- SKE-G8's negative `per_turn_sym` term is concentrated in cells where the
  turn count changed (especially `1→1` over-commit, where SKE-G8's same-turn
  EX is 70.0% vs SFT's 73.6% on identical records — a paired drop of 3.6pp
  on 824 records).
- Reading: **SKE-G8 reinforces stopping behavior more aggressively than
  Dense, but provides no signal that improves conditional SQL writing**.
  Aggregate gain washes out because the over-commitment cells produce
  paired EX losses that cancel the share-shift benefit.

By contrast, Dense and SKE-G4 both show **R_same paired ΔEX with CI strictly
above 0** — these conditions improve SQL quality on records they didn't
move. Dense's improvement is concentrated on Medium difficulty.

## Contribution Focus

**Dominant** — a deterministic, portable, leakage-safe decomposition of any
agentic-RL aggregate metric into (stopping-policy shift) and (conditional
solver capability), reported with paired-record bootstrap 95% CIs and
hardened by (i) Shapley-symmetric ordering, (ii) R_same paired same-turn
analysis as composition decontamination, (iii) per-difficulty stratification.
Applied to our 7B Dense GRPO, the conditional-solver term is concentrated on
Medium difficulty (`per_turn_sym` +3.30 pp, CI excludes 0).

**Supporting** — a mechanistic explanation of SKE-RL's failure: SKE
amplifies the stopping-policy shift more than Dense (SKE-G8 share_sym +2.52
vs Dense +0.83) but provides no signal that improves conditional SQL
writing (R_same paired ΔEX +0.78, CI overlaps 0). The aggregate gain
saturates and reverses precisely because over-commitment in the `1→1` and
`1→2+` cells produces paired EX losses that cancel the share-shift benefit.

**Non-contributions** (unchanged): SOTA on Spider/BIRD; new RL training
method; toolkit-as-lead.

## Headline numbers (signed pp; bootstrap 1000 resamples; percentile-95 CI; paired)

| # | Metric | Value | Use |
|---:|---|---|---|
| 1 | `share_shift_sym` (Dense vs SFT) | +0.83 [+0.07, +1.67] | PRIMARY |
| 2 | `per_turn_sym` (Dense vs SFT) | +1.49 [−0.51, +3.66] | PRIMARY |
| 3 | `R_same paired ΔEX` (Dense vs SFT) | +2.22 [+0.37, +4.06] ✓ | **decontaminated quality signal** |
| 4 | `share_shift_sym` (SKE-G8 vs SFT) | +2.52 [+1.71, +3.50] ✓ | mechanism |
| 5 | `per_turn_sym` (SKE-G8 vs SFT) | −2.13 [−3.87, −0.46] ✗ | mechanism |
| 6 | `R_same paired ΔEX` (SKE-G8 vs SFT) | +0.78 [−0.56, +2.22] | **mechanism (R_same null)** |
| 7 | per-difficulty `per_turn_sym` (Dense, Medium) | +3.30 [+0.22, +6.38] ✓ | localizes Dense's quality gain |

## Composition-caveat appendix paragraph (verbatim, copyable)

> Interpretation caveat. The term `per_turn_gain` is an exact accounting
> component, but not a pure causal "SQL quality" effect, because RL can
> change which records appear in each turn bin. To assess whether the sign
> of `per_turn_gain` is driven only by such case-mix changes, we add two
> robustness checks. First, we repeat the decomposition within Spider
> difficulty strata (easy / medium / hard), which reduces heterogeneity
> inside each turn bin. Second, we compute paired execution deltas on the
> no-shift subset R_same = {i : turn_T(i) = turn_B(i)}, where turn-bin
> reassignment is absent by construction. We treat agreement in sign across
> the full-sample decomposition, the difficulty-stratified decomposition,
> and the no-shift paired subset as evidence that the reported
> `per_turn_gain` pattern is not solely a composition artifact.

## Pre-registered launch path (per `PRE_REGISTRATION.md` v3 addendum)

| Block | Content | GPU-h | Status |
|---|---|---:|---|
| 0 | Zero-GPU robustness (Shapley-sym + R_same + difficulty-stratified + 3×3 paired-ΔEX cells) | 0 | ✓ done |
| B4 | Dense G=8 confound run, ckpt75 primary | ~28 | prereg locked, queued |
| Block A | Dense G=4, Dense G=8, SKE G=8 multi-seed (3 seeds each) | ~140 | queued |
| Block B | Stop-calibrated reward intervention (pilot + 3 seeds) + zero-train inference gate | ~112 | queued |
| Block C | BIRD ports (4 conditions × 2 seeds) | ~190 | queued |
| Block D | Spider-Realistic eval-only | ~10 | queued |

**Total**: ~480 GPU-h. Single-GPU constraint (Quadro RTX 6000, 24GB) means
calendar time ≈ 20 wall-days minimum if exclusively dedicated.

## Failure modes (revised; aligned with PRE_REGISTRATION v3-9)

| Failure mode | Detection | Fallback |
|---|---|---|
| B4 `share_shift_sym` outside [+1.8, +2.4] | bootstrap CI | report measured value; mechanism story survives if CI > 0 |
| B4 `per_turn_sym` 95% CI excludes 0 negative | reviewer-defined falsifier | drop SKE-specific claim; rewrite as "generic large-G quality regression" |
| SKE-G8 `per_turn_sym` not stably negative across 3 seeds | multi-seed CI | demote to workshop note |
| BIRD share-shift-dominant pattern absent | per-pair Shapley-sym CI | demote to workshop note |
| Stop-calibrated intervention doesn't preserve share-shift while reducing per-turn regression | reviewer-defined success criterion | drop intervention claim, retain diagnostic-only contribution |

## Why this pivot is principled (not HARK)

v2 was rejected because the lenient extractor produced empirically the same
SQL as strict on this dataset, making `protocol_gain` empty by construction
on a fully preregistered run. We report v2 falsification explicitly rather
than retrofit it. v3 uses an axis (turn count) that is part of the data
already (`num_turns` field in every eval JSON record). The decomposition
arithmetic is mathematically forced once the axis is chosen; only the
ordering choice has interpretive freedom and we report A/B/sym side by side
to surface that freedom.

The SKE-G8 `per_turn_sym` CI excluding 0 is a **post-hoc finding on the same
data**, not pre-data signal — bootstrap survival ≠ removing the HARK concern.
Therefore: we frame it as "evidence consistent with over-commitment" and
gate the stronger causal language on the pre-registered B4 confound run +
multi-seed replication.

## Recommendation

Adopt v3 (R2-revised) as the working proposal. Update `EXPERIMENT_PLAN.md`
Week 1 to mark zero-GPU work as complete. Week 2 launches B4 + Block A
multi-seed. Week 3 launches Block B (intervention). Week 4 launches Block C
(BIRD) + writing.

Target venue: **ICLR 2027 main** (deadline ~Sep 2026).
Fallback: **NeurIPS 2026 workshop** (TRL / RL-for-LLMs).
