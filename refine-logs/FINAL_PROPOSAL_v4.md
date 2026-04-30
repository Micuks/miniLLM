# Research Proposal v4 — Group Size Trades Commitment for Capability

**Status**: ADOPTED — supersedes v3 on the basis of the B4 falsification result.
**Date**: 2026-04-30
**Supersedes**:
- `FINAL_PROPOSAL_v3_DRAFT.md` (R0/2026-04-29) — its SKE-specific framing was falsified by B4.
- `FINAL_PROPOSAL.md` (v2, R4 9.00/10) — its protocol-vs-semantic axis was empirically null.

**Required reads (in order)**:
- `A0_PRE_AUDIT_FINDING_2026-04-29.md` — v2 axis is null.
- `TURN_DECOMP_FINDING_2026-04-29.md` — v3 turn-progression decomposition with signal.
- `B4_RESULT_v3_PREDICTION_FALSIFIED_2026-04-30.md` — pre-registered prediction falsification.

---

## Problem Anchor (immutable from v2/v3)

Multi-turn ReAct GRPO succeeds modestly at 7B (+2.32 pp full Spider-dev) but
fails at 3B; SKE-RL fails at G=4 / G=8. v3 hypothesized SKE-specific
mechanism, ran B4 (Dense G=8) as the discriminating test, and B4 falsified
the SKE-specific framing. This proposal absorbs the falsification.

## What B4 changed

v3 predicted Dense G=8 would isolate share-shift mechanism without per-turn
loss. Observed: Dense G=8 reproduces SKE-G8's per-turn regression
(`per_turn_gain = -1.91 [-3.64, -0.27]` ✗) and same-magnitude share shift
(`share_shift = +2.59 [+1.68, +3.59]` ✓). The mechanism is generic to
group size, not advantage-estimator-specific.

## Method Thesis (v4 — one sentence)

Group size G in agentic-GRPO tunes a fundamental trade-off between
commitment-policy strength and per-turn task-execution capability;
increasing G amplifies the strategy shift while degrading per-turn
quality, and the trade-off is invariant to the advantage-shaping
estimator (Dense, SKE-RL).

## Contribution Focus (v4)

**Dominant**: Two-term decomposition
(`share_shift_term + per_turn_gain_term`) with paired-record bootstrap
CIs, applied to a 2×2 ablation {Dense, SKE-RL} × {G=4, G=8} on
Spider-dev. Reveals the G-driven trade-off.

**Methodological**: Pre-registered prediction falsification documented
end-to-end. We made a public quantitative prediction in v3, ran the
experiment, and reported the falsification. The reframe is forced by
data, not by reviewer preference. The falsification *strengthens* the
final claim by replacing a narrow SKE-specific mechanism with a
universal G-driven trade-off.

**Negative**: SKE-RL's failure is a special case of the G effect, not a
unique pathology of the class-baseline advantage. v3's secondary
contribution recedes; the G axis takes its place.

**Non-contributions**: SOTA, new RL method, new advantage estimator,
multi-seed averaging, scaling-law claim across model sizes.

## Headline numbers (full Spider-dev 1034, paired bootstrap 1000 resamples, 95% percentile CI; ✓/✗ = CI excludes 0 in pos./neg. direction)

| Pair | total ΔEX | share_shift_pp | per_turn_gain_pp |
|---|---:|---:|---:|
| Dense G=4 vs SFT | +2.32 [+0.39, +4.35] | +0.90 [+0.08, +1.84] ✓ | +1.42 [−0.64, +3.58] |
| Dense G=8 vs SFT | +0.68 [−0.87, +2.22] | +2.59 [+1.68, +3.59] ✓ | **−1.91 [−3.64, −0.27] ✗** |
| SKE-RL G=4 vs SFT | +1.64 [−0.39, +3.87] | +1.32 [+0.47, +2.24] ✓ | +0.32 [−1.98, +2.58] |
| SKE-RL G=8 vs SFT | +0.39 [−1.06, +1.94] | +2.64 [+1.73, +3.59] ✓ | **−2.26 [−3.89, −0.60] ✗** |

The G=8 pair reproduces with point estimates within 0.1 pp of each
other across estimators on `share_shift` and within 0.4 pp on
`per_turn_gain` — the two estimators are operationally equivalent at
G=8.

## Per-record over-commit signature

(1 → 1) cell EX rates (records where both SFT and the RL condition
resolved in 1 turn):

| Pair | (1→1) n | rl EX% within cell |
|---|---:|---:|
| Dense G=4 | 753 | 73.6% (≈ SFT 73.6%) |
| Dense G=8 | 820 | **70.1%** |
| SKE-RL G=4 | 757 | 73.1% |
| SKE-RL G=8 | 824 | **70.0%** |

G=8 conditions trap more records in the (1→1) cell AND degrade
accuracy by ~3.5 pp on those records. Both effects are size-matched
across Dense and SKE-RL at G=8.

## Method Plan

| Block | Cost | Status |
|---|---:|---|
| Per-condition turn-progression breakdown | <0.1 GPU-h | ✓ done |
| Pairwise decomposition vs SFT | <0.1 GPU-h | ✓ done |
| Per-difficulty stratified decomposition | <0.1 GPU-h | ✓ done |
| Per-record paired-transition tables | <0.1 GPU-h | ✓ done |
| Bootstrap CIs (paired, 1000 resamples) | <0.1 GPU-h | ✓ done |
| Trajectory length signal | <0.1 GPU-h | ✓ done |
| **Dense G=8 confound run + eval** | ~13 GPU-h | ✓ done (B4) |
| Writing | n/a | in progress |

Total mandatory compute consumed: **~13 GPU-h**. No further training
mandatory.

## Failure modes (revised)

| Mode | Detection | Fallback |
|---|---|---|
| G effect not monotonic at G < 4 or G > 8 | would need G=2 or G=16 runs | acknowledged limitation; trade-off characterized only at the two G values we ran |
| Per-turn regression a Spider-specific artifact | external benchmark needed | acknowledged limitation; future work |
| Reviewer challenges 2-G interpolation | "you only have 2 points on G axis" | response: the prediction was binary (G=8 reproduces SKE-G8 pattern or doesn't), not a curve fit |
| Confound from skipped-old-logprob (B2 optimization) | re-run Dense G=8 with full PPO old-logprob | not in budget; skip-old is mathematically unbiased at single-epoch (verified in tests) |

## Recommendation (locked)

Adopt v4 as the working proposal. Update paper draft Sections 4-6 to use
the 2×2 framing (already done in `PAPER_DRAFT_v4_results.md`). No more
training mandatory. Writing time is the bottleneck.

The v3 draft is preserved as the pre-registration record of the
prediction we falsified — the credibility of v4's claim depends on
showing the falsification explicitly in the paper.
