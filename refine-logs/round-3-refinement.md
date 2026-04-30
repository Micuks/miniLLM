# Round 3 Refinement

## Problem Anchor (verbatim)
Same as round 0; not repeated.

## Anchor Check
All 3 R3 fixes (E3 effect-size, bucket denominator clarity, signed attribution) tighten measurement quality without changing the question. No drift.

## Simplicity Check
- Dominant contribution unchanged.
- One subtle simplification: `observation_repair_share` officially demoted to subanalysis row in appendix table, not in main figure caption. This was already the intent; now explicit.
- Dominant table now leads with `lenient_attribution_gap` (raw, signed, in pp); `attribution_ratio` reported alongside but with explicit caveat when |strict_gain| < 0.5 pp.

## Changes Made

### 1. E3 statistical criterion replaced with equivalence test (IMPORTANT)
- **Reviewer said**: `p > 0.05` is absence of significance, not equivalence.
- **Action**: E3 now uses **Cliff's delta** with 95% bootstrap CI and a pre-registered equivalence margin **|δ| ≤ 0.147** (Vargha-Delaney's "small effect" threshold). E3 is accepted only if the entire 95% CI of |δ| lies within [0, 0.147]. Reported alongside the median per-bucket SKE class-baseline shift magnitude for transparency.
- **Why this margin**: 0.147 is the standard literature threshold for "small effect" under the Vargha-Delaney convention; pre-registering avoids garden-of-forking-paths concerns.
- **Failure mode**: if CI exceeds the margin, E3 is FAILED — the SKE single-mechanism story is then weakened, and we report it honestly as "we cannot rule out a non-trivial relationship between SKE class-baseline shifts and dev semantic outcomes".

### 2. Bucket-share denominators explicit (MINOR)
- **Reviewer said**: `semantic_gain` denominator ambiguous.
- **Action**: report **two** quantities for each bucket:
  - `transition_bucket_share[B]` = |records in B| / |records where SFT.final_correct_strict=False AND RL.final_correct_strict=True|
  - `global_bucket_rate[B]` = |records in B| / |all Spider-dev records|
- **Why both**: transition share answers "of the wins, how many came from B?"; global rate answers "what fraction of the dev set saw an outcome of type B?". Reporting both prevents readers from misreading either in isolation.

### 3. Signed attribution + raw gap as primary (MINOR)
- **Reviewer said**: ratio can go negative or > 1; misleading to claim [0, 1].
- **Action**: the primary headline number is now **`lenient_attribution_gap = ΔEX_strict − ΔEX_lenient`** in absolute pp (signed). The `attribution_ratio` is reported as a secondary diagnostic, computed only if `|ΔEX_strict| ≥ 0.5 pp`, and reported as a **signed** ratio without clipping. Out-of-[0,1] values are flagged as "lenient gain exceeds strict gain — protocol noise hypothesis weakened" or "tiny strict gain — denominator unstable".

## Revised Proposal (delta-only)

The proposal body changes only in the metric definitions and the E3 evidence statement. Full proposal text remains as in `round-2-refinement.md` with these substitutions:

### Headline numbers (replaces R2 §Core Mechanism > Headline numbers)

All values reported with 95% bootstrap CIs (1000 resamples over Spider-dev records):

| # | Metric | Formula | Primary use |
|---:|---|---|---|
| 1 | `lenient_attribution_gap` | ΔEX_strict − ΔEX_lenient (signed, in pp) | **PRIMARY** — large positive = gain mostly protocol; ~0 = mostly semantic; negative = lenient extracts find more semantic gain than strict measured (anomaly flag) |
| 2 | `attribution_ratio` | gap_1 / max(\|ΔEX_strict\|, 0.5pp); signed; no clipping | Secondary, suppressed when \|ΔEX_strict\| < 0.5 pp (denominator unstable) |
| 3 | `transition_bucket_share[B]` | \|B\| / \|{SFT-strict-wrong → RL-strict-right}\| | "of strict-gain transitions, fraction in bucket B" |
| 4 | `global_bucket_rate[B]` | \|B\| / 1034 | "fraction of dev set with outcome B" |
| 5 | `observation_repair_share` | \|protocol_gain ∩ obs_repair_flag\| / \|protocol_gain\| | Appendix subanalysis only |

### E3 (replaces R2 §Optional Supporting Component > E3)

**E3 (revised):** SKE training-time per-class advantage shift is statistically equivalent across dev `skeleton_class`-buckets stratified by outcome density.

- **Test statistic**: Cliff's δ comparing per-`skeleton_class` mean SKE advantage-shift magnitude between two strata: high-`semantic_gain`-density classes (top-quartile by `semantic_gain` rate) vs high-`protocol_gain`-density classes (top-quartile by `protocol_gain` rate).
- **Equivalence margin (pre-registered)**: |Cliff's δ| ≤ 0.147 (Vargha-Delaney "small effect").
- **Acceptance**: E3 PASS iff entire 95% bootstrap CI of |δ| ⊆ [0, 0.147]. E3 FAIL otherwise (reported honestly as inconclusive equivalence).
- **Reported alongside**: median per-bucket SKE class-baseline shift magnitude in absolute pp of advantage-normalized units.
- **Tier-3 fallback** (join coverage <50%): downgrade to aggregate descriptive comparison, no equivalence claim.

### Failure modes (additions)

| Failure | Detect | Fallback |
|---|---|---|
| `lenient_attribution_gap` is negative outside CI | Lenient gain > strict gain | Anomaly: extractor or extraction is over-recovering. Re-validate extractor; if confirmed, the diagnostic still tells a story (lenient view picks up semantic gain hidden by strict's protocol-failure miss) but the protocol/semantic decomposition is non-monotone. Report and discuss. |
| `\|ΔEX_strict\| < 0.5 pp` (denominator unstable) | Bootstrap CI of ΔEX_strict crosses 0 | Suppress `attribution_ratio`; report only `lenient_attribution_gap` and bucket counts. Possible "Dense gain is mostly noise" finding. |
| E3 inconclusive (CI of \|δ\| straddles 0.147) | Bootstrap interval extends past margin | Honestly report: "we cannot rule out a small but non-trivial relationship between SKE class-baseline shifts and semantic outcomes". SKE single-mechanism claim weakened to "consistent with but not proving" the protocol-bottleneck thesis. |

(All other sections of the R2 proposal unchanged.)

## Compute & Timeline (unchanged from R2)

Mandatory: ~40 GPU-h (1.7 run-eq) + 3 days CPU + 1 week writing.
