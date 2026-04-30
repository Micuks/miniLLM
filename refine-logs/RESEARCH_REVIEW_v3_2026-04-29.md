---
date: 2026-04-29
reviewer: gpt-5.4 (xhigh reasoning)
thread_id: 019dd931-50bd-7511-94d8-6ce507348142
rounds: 2
verdict: WORKSHOP-WITH-FIXES
recommended_decision: option 2 (launch B4)
---

# Round-2 review on the v3 pivot — consolidated findings

Senior reviewer (gpt-5.4 xhigh) was given the full v3 context: SUMMARY,
v3 draft, paper intro, A0 finding, full numbers. Below are the binding
takeaways. Raw traces at `.aris/traces/research-review/2026-04-29_run01/`.

## Scores

| Dimension | Score |
|---|---:|
| Logical soundness of v3 pivot and decomposition | 6/10 |
| Empirical strength on existing data | 5/10 |
| Workshop venue readiness | 6/10 |
| Main-conference plausibility (with reasonable additions) | 4/10 |
| **Overall** | **6/10** |

**Verdict**: WORKSHOP-WITH-FIXES.
**Recommended decision**: option 2 — launch B4 while drafting.

## What the reviewer agrees with

- Arithmetic identity `total = share_shift + per_turn_gain` is sound.
- Bootstrap (1000 resamples, paired by record id, percentile-95, fixed seed)
  is the right uncertainty protocol.
- The negative-result-plus-diagnostic framing is **a legitimate workshop contribution**
  if framed honestly.
- SKE-G8's per-turn loss CI excluding 0 is real signal worth surfacing —
  but it's post-hoc on this data.

## What the reviewer pushes back on

### (1) "Principled pivot vs HARK-ed rationalization" — the central risk

> Principled **if and only if** you present it as an exploratory pivot
> after a falsified preregistered hypothesis. Not principled if you
> write v3 as though it was the intended thesis all along.

The SKE-G8 per_turn_gain CI excluding 0 is **not** pre-data signal. It
is a post-hoc finding on the same data that motivated the new story.
Bootstrap survival ≠ removing the HARK concern.

**Required mitigation**: explicit "preregistered v2 failed → exploratory v3"
language in both abstract and §1. (Verbatim copy below.)

### (2) Decomposition is reference-dependent (Oaxaca-Blinder issue)

The v3 draft uses Ordering A (baseline-weighted share-shift,
treatment-weighted per-turn). Ordering B (treatment-weighted share-shift,
baseline-weighted per-turn) gives different magnitudes. Ordering choice
is interpretively non-unique.

**Computed gap on our data** (reviewer-derived from existing table):

| Pair vs SFT | share_A | share_B | share_sym | A−B gap |
|---|---:|---:|---:|---:|
| Dense G=4   | 0.90 | 0.77 | 0.84 | 0.13 pp |
| SKE G=4     | 1.34 | 1.09 | 1.22 | 0.24 pp |
| SKE G=8     | 2.65 | 2.40 | 2.53 | 0.25 pp |

(I verified these arithmetically — all three orderings track. Sign and
qualitative ranking are stable; magnitudes differ enough to merit
disclosure.)

**Required fix**: report the symmetric Shapley average as primary; A and B
in appendix; replace "39%/80%/100% of gain" language with "the qualitative
conclusion is invariant to decomposition ordering".

### (3) `per_turn_gain` is NOT pure SQL quality

Once RL changes which records land in turn=1 vs turn≥2, the within-turn
EX terms no longer compare like with like. The `per_turn` term blends
(quality change) with (case-mix change).

**Required mitigation**:
- Stratify by difficulty AND by turn count.
- Paired same-turn analysis on the no-shift subset
  `R_same = {i : turn_T(i) = turn_B(i)}`.
- Treat sign agreement across (full-sample / difficulty-stratified /
  no-shift subset) as the evidence threshold for the pattern claim.

### (4) Single-seed variance is the real attack vector

Stronger objection than the deterministic SQL evaluator or Spider-only
scope. A single-seed +2.32 pp story is fragile. If only one extension is
affordable: **multi-seed replication of Dense G=4, Dense G=8, SKE G=8**
on training. NOT BIRD, NOT 14B, NOT 3B archaeology.

## The Shapley-symmetric decomposition (closed form, binary k=2)

Notation: `s_B := s(B,1)`, `s_T := s(T,1)`, `b1 := EX(B,1)`,
`b2 := EX(B,2+)`, `t1 := EX(T,1)`, `t2 := EX(T,2+)`.

**Ordering A**:
```
share_shift_A = (s_T - s_B) · (b1 - b2)
per_turn_A    = s_T · (t1 - b1) + (1 - s_T) · (t2 - b2)
```

**Ordering B**:
```
share_shift_B = (s_T - s_B) · (t1 - t2)
per_turn_B    = s_B · (t1 - b1) + (1 - s_B) · (t2 - b2)
```

**Symmetric Shapley average** (use as primary):
```
share_shift_sym = (s_T - s_B) · ((b1 - b2) + (t1 - t2)) / 2
per_turn_sym    = ((s_B + s_T)/2) · (t1 - b1)
                + (1 - (s_B + s_T)/2) · (t2 - b2)
```

Useful identity:
```
share_shift_A − share_shift_B = (s_T − s_B) · ((b1 - b2) − (t1 - t2))
```

So binary binning does **not** make ordering-dependence vanish; it is
only mild when the turn-gap `(EX(1) − EX(2+))` is similar under
baseline and treatment.

## Composition-vs-quality decontamination — copyable appendix paragraph

```
Interpretation caveat. The term `per_turn_gain` is an exact accounting
component, but not a pure causal "SQL quality" effect, because RL can
change which records appear in each turn bin. To assess whether the sign
of `per_turn_gain` is driven only by such case-mix changes, we add two
robustness checks. First, we repeat the decomposition within Spider
difficulty strata (easy / medium / hard), which reduces heterogeneity
inside each turn bin. Second, we compute paired execution deltas on the
no-shift subset R_same = {i : turn_T(i) = turn_B(i)}, where turn-bin
reassignment is absent by construction. We treat agreement in sign
across the full-sample decomposition, the difficulty-stratified
decomposition, and the no-shift paired subset as evidence that the
reported `per_turn_gain` pattern is not solely a composition artifact.
```

## Baseline-turn-conditioned paired analysis — required tabular form

| SFT turn | RL turn | n | share | EX_SFT (same records) | EX_RL (same records) | paired ΔEX | contribution = (n/N) × paired ΔEX |
|---|---:|---:|---:|---:|---:|---:|---:|

If only one extra column fits: `paired ΔEX`. Defensible appendix sentence:

```
To separate transition frequency from transition consequence, each
(turn_B → turn_T) cell reports not only its count but also the paired
execution-rate difference on the same records, EX_T − EX_B, and that
cell's contribution to the overall execution-match delta.
```

## B4 pre-registration — copy verbatim into git BEFORE launching

```
Experiment ID: B4 Dense G=8 confound

Primary analysis. Binary turn binning k ∈ {1, 2+} on full Spider-dev
(N=1034), using the symmetric Shapley decomposition:
  total_ΔEX = share_shift_sym + per_turn_sym.
Uncertainty is estimated with 1000 paired bootstrap resamples by record
id, percentile 95% confidence intervals, fixed bootstrap seed = 20260429.
Ordering-A and Ordering-B decompositions are reported as appendix
sensitivity checks only.

Hypothesis. Relative to the 7B SFT baseline, a Dense GRPO run with G=8
will reproduce the positive commitment/turn-share shift associated with
larger group size, but will not reproduce the negative within-turn
component observed for SKE-RL G=8; therefore, if observed, the SKE-RL
G=8 per-turn regression is not a generic G=8 effect.

Predicted bounds for share_shift_sym. Point estimate in the range +1.8
pp to +2.4 pp, with 95% CI entirely above 0.

Predicted bounds for per_turn_sym. Point estimate in the range −0.5 pp
to +0.5 pp, with 95% CI overlapping 0.

Falsification condition. The SKE-specific-degradation claim is falsified
if Dense G=8 yields per_turn_sym < 0 with a 95% CI excluding 0. A
secondary falsification of the generic high-G commitment-shift story is
share_shift_sym failing to be positive with a 95% CI excluding 0.

Decision rule. If both predictions hold, we will claim that larger group
size generically increases commitment shift, while the negative per-turn
component observed for SKE-RL G=8 is not reproduced by Dense and is
therefore consistent with an SKE-specific quality regression. If Dense
G=8 shows a negative per_turn_sym with 95% CI excluding 0, we will drop
the SKE-specific claim and rewrite the result as evidence for a generic
large-G over-commit/quality tradeoff. If the share_shift_sym prediction
fails, we will drop the generic high-G interpretation and present B4 as
inconclusive with respect to mechanism.
```

**Training config**:
- Same launcher / data / optimizer / LR schedule / reward / prompt format / rollout limits.
- Same training seed if possible.
- Change only `G: 4 → 8`.
- Train the same 100-step schedule; **ckpt75 as primary endpoint** (matches existing comparison set).
- ckpt100 reported as secondary sensitivity.
- Do **not** scale steps to equalize total sampled rollouts — this is a mechanism-control run, not a compute-equalized scaling study.
- If compute tightens, stop at ckpt75.

## B4 results-to-claims matrix

| Dense G=8 share_shift_sym | per_turn_sym CI overlaps 0 | per_turn_sym CI excludes 0 negative | per_turn_sym CI excludes 0 positive |
|---|---|---|---|
| **In window [+1.8, +2.4] and CI > 0** | Allowed: generic high-G mainly changes commitment; SKE-specific degradation remains plausible. Forbidden: "Dense proves no quality effect exists" or any causal claim. | Allowed: negative per-turn can be generic to G=8; drop SKE-specific degradation claim. Forbidden: "SKE-specific quality regression." | Allowed: Dense G=8 improves both commitment and within-turn; SKE-G8 underperformance becomes more plausibly SKE-specific. Forbidden: "Large G is generically harmful." |
| **Above window and CI > 0** | Allowed: generic high-G commitment effect stronger than expected; SKE-specific degradation still plausible. Forbidden: "Prediction matched exactly." | Allowed: high G likely induces both stronger commitment AND generic quality loss; SKE-specific claim fails. Forbidden: "Only SKE causes the regression." | Allowed: Dense benefits from larger G more than expected; SKE-specific degradation story strengthened. Forbidden: "High G generically hurts quality." |
| **Below window or CI not strictly > 0** | Allowed: B4 weakens or renders inconclusive the generic commitment-shift story; any SKE-specific claim must be tentative. Forbidden: "Generic large-G commitment shift confirmed." | Allowed: evidence points toward generic G=8 quality regression without predicted commitment benefit. Forbidden: "SKE-specific degradation" and "generic positive commitment-shift mechanism." | Allowed: Dense does not support preregistered generic-G story; SKE-specific degradation remains possible but unconfirmed. Forbidden: "Larger G generically causes over-commitment." |

## Honest-pivot framing — copy verbatim into the paper

### Abstract version
```
Our preregistered attribution hypothesis was falsified: after
implementing a leakage-safe lenient SQL extractor, we found that it
recovered virtually no additional outputs beyond the strict parser on
full Spider-dev. We therefore treat the remainder of the study as
exploratory and ask a narrower diagnostic question: when multi-turn
GRPO changes aggregate execution match, does it do so by improving
within-turn SQL correctness or by changing when the agent commits?
Using an exact turn-share/within-turn decomposition with paired
bootstrap confidence intervals, we find that the observed gains are
dominated by shifts toward one-turn completion; SKE-RL at G=8 shows
the largest such shift but also a negative within-turn term, a pattern
we report only as evidence consistent with over-commitment.
```

### Section 1 paragraph (current data, no B4)
```
This paper reports a failed preregistered hypothesis and an exploratory
diagnostic that followed from that failure. Our preregistration
targeted a "protocol beats structure" explanation: we hypothesized that
some of the apparent execution-match gains from agentic RL would come
from trajectories whose final SQL could be recovered by a leakage-safe
lenient extractor even when a strict parser failed. After implementing
that extractor and evaluating all four 7B runs on full Spider-dev, this
hypothesis was falsified: the strict and lenient extractors returned
the same final SQL string for essentially every trajectory. Rather than
retrofitting that negative result into a confirmatory narrative, we
report it explicitly and treat the analysis that follows as
exploratory. The exploratory question is narrower and operational: when
RL changes aggregate execution match, is the change more associated
with reallocating examples across turn counts or with changing
execution accuracy within a fixed turn bin? Across paired bootstrap
resamples, with interval bounds stable across bootstrap seeds, the
strongest pattern is that SKE-RL at G=8 shifts more examples into
one-turn completion but does not show corresponding within-turn gains;
we therefore interpret its negative within-turn term cautiously as a
pattern consistent with over-commitment, not as a preregistered causal
finding.
```

### Section 1 paragraph (post-B4, if predictions hold)
Replace the last sentence with:
```
Across paired bootstrap resamples, and in a preregistered Dense G=8
confound run, the strongest pattern is that larger group size increases
one-turn commitment, while the negative within-turn term observed for
SKE-RL G=8 is not reproduced by Dense; we therefore interpret the SKE-RL
result as evidence consistent with an SKE-specific over-commitment
failure mode.
```

## The single sentence (under 30 words)

> After a preregistered extractor hypothesis failed, we show that
> agentic GRPO's Spider gains arise mainly from earlier commitment,
> while SKE-RL amplifies this shift without improving conditional SQL
> accuracy.

## Top 3 actionable recommendations (ranked by impact-per-GPU-hour)

1. **Zero-GPU robustness fixes** (CPU only, hours of work):
   - Add Shapley-symmetric decomposition (sym primary, A & B in appendix).
   - Add difficulty-stratified-by-turn-count cells.
   - Add `R_same` paired same-turn subset analysis.
   - Add paired ΔEX + contribution columns to the 3×3 transition matrix.
   - Soften interpretation language ("39%/80%/100%" → "qualitative conclusion invariant to ordering").
2. **Run B4 with the prereg above** (~28 GPU-h, ~1.5 days):
   - Commit prereg text to git BEFORE launching.
   - Same launcher; only change G=4 → G=8.
   - ckpt75 primary endpoint; ckpt100 secondary.
3. **Multi-seed training replication** (1+ GPU-week, only if budget remains):
   - Replicate Dense G=4, Dense G=8, SKE G=8 across additional training seeds.
   - This is the single weakest part of the current evidence.
   - Skip BIRD, 14B, and more 3B archaeology — bigger but underpowered excursions.

## Forbidden language list

Strike from any draft:
- "v3 thesis" (implies it was preregistered).
- "SKE provides no signal that improves per-turn quality" → "under this decomposition, we observe no net per-turn improvement".
- "39% / 80% / 100% of the gain" → "qualitative conclusion invariant to decomposition ordering".
- Any wording that lets the v2 preregistration credibility leak into v3 claims.

## What to do with FINAL_PROPOSAL.md, EXPERIMENT_PLAN.md, PRE_REGISTRATION.md

Reviewer recommendation translated to repo actions:

| File | Action |
|---|---|
| `FINAL_PROPOSAL.md` (R4 9.00/10) | Mark archived; add header noting v2 was falsified; create `FINAL_PROPOSAL_v3.md` from current `FINAL_PROPOSAL_v3_DRAFT.md` with the Shapley-sym + honest-pivot fixes applied |
| `EXPERIMENT_PLAN.md` | Update Week 1 to reflect what was actually done (turn-progression decomposition + per-difficulty + per-record paired); add the zero-GPU robustness fixes as Week 1.5; lock B4 in Week 2 with the verbatim prereg above; Week 3 is writing |
| `PRE_REGISTRATION.md` | Append a new §3 "B4 Dense G=8" with the verbatim prereg block from this review BEFORE running B4 |

User instruction (from prior session): all three files remain untouched
until user authorizes. This review only proposes the actions; the user
makes the call.

## Recommended decision (reviewer + my gloss)

**Option 2: launch B4 while drafting**.

Rationale:
- The strongest weakness is single-seed + post-hoc-on-same-data. B4
  directly addresses the most obvious confound (G effect vs SKE effect).
- B4 is cheap (~28 GPU-h, ~1.5 days) for what it buys.
- Drafting the rest in parallel does not waste compute and exercises
  the framing language while the run cooks.
- The zero-GPU robustness fixes (Shapley-sym, R_same, difficulty cells)
  should be done **before** B4 results land so the analysis pipeline is
  ready and the prereg framework is locked.

If the user wants to delay B4 (e.g., write workshop draft first to find
out whether the story holds without it): acceptable, but the paper will
then read as cautious-descriptive, scoring 5/10 on workshop readiness
rather than 6-7/10 with B4 confirming or falsifying the SKE-specific
mechanism.

## Compute spent on this review

CPU only. Two codex xhigh rounds, full traces saved at:
- `.aris/traces/research-review/2026-04-29_run01/round1_full.json`
- `.aris/traces/research-review/2026-04-29_run01/round2_full.json`

Resumable thread: `019dd931-50bd-7511-94d8-6ce507348142`.
