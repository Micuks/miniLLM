---
title: Pre-registration — Protocol Beats Structure (Diagnostic Study)
date: 2026-04-29
parent: FINAL_PROPOSAL.md, EXPERIMENT_PLAN.md
status: LOCKED — written before any data inspection or A1+ runs
authors: micuks
---

# Pre-registration

This document locks every analysis decision **before** the diagnostic pipeline reads
any record from the four eval JSONs listed below. Anything not pre-registered here
must either follow the rules in `FINAL_PROPOSAL.md` deterministically or be reported
as exploratory, never as a confirmatory finding.

## 1. Input data — absolute paths + integrity hashes

All inputs are immutable. SHA256 verification is mandatory before each script reads
any file. Documents at `/mnt/data1/wuql/dev/miniLLM/refine-logs/`; data + code at
`/mnt/data2/wuql/miniLLM/`.

| Asset | Absolute path | size (bytes) | SHA256 | mtime |
|---|---|---:|---|---|
| 7B SFT eval (Spider-dev 1034) | `/mnt/data2/wuql/miniLLM/outputs/eval_full_dev_sft.json` | 1617445 | `291f54b88a937be2f0994f7d7dcf7f40c6a315e61de0563d882ef6c81bac5fa3` | 2026-04-27 12:23:57 +0800 |
| 7B Dense75 eval (Spider-dev 1034) | `/mnt/data2/wuql/miniLLM/outputs/eval_full_dev_dense75.json` | 1576943 | `e0b733ab181e980e0344e7b4d21723030280cae820b9930d5ca8438e06255888` | 2026-04-27 16:05:11 +0800 |
| 7B SKE-G4-75 eval (Spider-dev 1034) | `/mnt/data2/wuql/miniLLM/outputs/eval_full_dev_ske75.json` | 1586069 | `7d6cbd3e5f438731b4e1b1448d93845d2baf93683ea66d3aeae2c4328f83bc66` | 2026-04-27 19:54:20 +0800 |
| 7B SKE-G8-75 eval (Spider-dev 1034) | `/mnt/data2/wuql/miniLLM/outputs/eval_full_dev_ske_g8_ck75.json` | 1566308 | `390eb1bf86c9a934fa4e839854cc0a4e40a7f950c4c5ee0964351cba0567d85b` | 2026-04-29 01:02:09 +0800 |

Every script in `miniLLM/diag/` and `scripts/diag_*.py` must call a common
`verify_input_hashes()` helper that fails loudly if any hash differs from the table
above. If any input must change (e.g. a re-run replaces a JSON), this document is
**superseded** by a new pre-registration with a new date and re-locked hashes — never
silently updated.

## 2. Bootstrap protocol

- **Resampling unit**: Spider-dev record (n=1034). Records are paired across
  conditions (same record id, same `db_path`, same `gold_sql`).
- **Resamples**: 1000 per CI.
- **CI type**: 95% **percentile** interval (2.5th and 97.5th sample percentiles). No
  bias-corrected acceleration; report which percentiles in the methods section.
- **RNG**: `numpy.random.default_rng(seed=20260429)` for reproducibility. Seed is
  hard-coded in `miniLLM/diag/bootstrap.py`.
- **Sign convention**: `Δ = treatment − baseline` (so positive = improvement).
- **All headline numbers carry CIs**: `lenient_attribution_gap`, `attribution_ratio`,
  `transition_bucket_share[*]`, `global_bucket_rate[*]`, `lenient_EX`, `Cliff's δ`.

## 3. Bucket precedence + per-side flags

Per-side flags (computed identically for sft and rl, from each record's
trajectory + db_path):

```
strict_extract_ok      = current extract_final_sql() returned non-None AND sqlglot-parses
lenient_extract_ok     = lenient_extract() returned non-None  (parses by construction)
first_sql_exec_ok      = execution_errors[0] == "" if all_sql_actions else False
final_correct_strict   = exec_match(strict_pred_sql, gold_sql)  if strict_extract_ok  else False
final_correct_lenient  = exec_match(lenient_pred_sql, gold_sql) if lenient_extract_ok else False
observation_repair_flag= (not first_sql_exec_ok) and final_correct_strict
equiv_class_change_flag= (canonical_ast(SFT.strict_pred) != canonical_ast(RL.strict_pred))
```

Bucket rules — precedence: **protocol_gain > semantic_gain > no_change**. Both gain
buckets require `RL.final_correct_strict = True`, so `transition_bucket_share` has a
well-defined denominator equal to the strict-gain transition set.

```
protocol_gain = (not SFT.final_correct_strict
                 and SFT.final_correct_lenient
                 and RL.final_correct_strict)

semantic_gain = (not SFT.final_correct_lenient
                 and RL.final_correct_lenient
                 and RL.final_correct_strict)

no_change     = otherwise
```

**Subanalysis (NOT a bucket; appendix only):**
```
lenient_only_repair_flag = (not RL.final_correct_strict
                            and RL.final_correct_lenient)
```

## 4. A0 sanity gates (lenient extractor)

Two separate criteria; **both must pass** before A1 launches.

1. **Quantitative gate**:
   - Sample 50 records from `eval_full_dev_sft.json` where
     `strict_extract_ok = False`.
   - Run `lenient_extract` on each.
   - Pass if `parseable_lenient_rate >= 80%` (≥40 of 50 records produce a
     parseable SQL).
   - On fail: tighten extractor (e.g. drop last-action fallback) and re-run A0.

2. **Qualitative gate**:
   - From the 50-record sample, take the first 20 records where lenient extraction
     succeeded (chronological order in the JSON; if fewer than 20, use all
     successes).
   - For each, manually inspect `(trajectory, lenient_pred_sql, gold_sql)` and
     classify lenient_pred_sql as one of:
     - `intended_final` — the SQL the model meant as its final answer
     - `observation_echo` — text from an Observation block accidentally captured
     - `non_final_action` — an exploratory SQL action that wasn't the answer
   - Pass if `intended_final >= 18 / 20`.
   - On fail: tighten extractor (Answer-region anchor, `Observation:` blocking) and
     re-audit.

The old gate `lenient_EX(SFT) >= strict_EX(SFT)` is **removed**; on a sample where
strict was 0 by selection it would trivially pass.

Audit results are written to `refine-logs/A0_audit_<YYYY-MM-DD>.md` before A1 runs.

## 5. Pass/fail definitions

### C1 (Claim 1 — protocol attribution)
- **PASS**: `lenient_attribution_gap > 0` with 95% CI excluding 0,
  AND `transition_bucket_share[protocol_gain] > transition_bucket_share[semantic_gain]`
  with non-overlapping CIs.
- **INVERT**: `lenient_attribution_gap ≈ 0` with CI including 0
  → reframe to semantic-dominant per the FINAL_PROPOSAL "Branching Plan".
- **ANOMALY**: `lenient_attribution_gap < 0` with CI excluding 0
  → investigate extractor; treat as a finding rather than a bug only after
    re-validation.

### C2 (Claim 2 — SKE mechanism, mandatory E1 + E2; E3 is Tier-3 descriptive only)
- **E1 PASS**: `transition_bucket_share[semantic_gain]` for (SKE-G4 vs Dense) ≤ that
  for (Dense vs SFT), AND same for (SKE-G8 vs Dense), with 95% CIs.
- **E2 PASS**: `lenient_EX(SKE) ≤ lenient_EX(Dense)` within bootstrap CI for both
  G=4 and G=8 arms.
- **C2 PASS**: E1 AND E2 both hold.
- **C2 FAIL**: E1 OR E2 fails → SKE story reduced to a paragraph.
- E3 is reported as a Tier-3 descriptive analysis only — never as evidence for or
  against C2.

### R1 (lenient extractor sanity)
- **PASS**: `parseable_lenient_rate ≥ 80%` AND `intended_final ≥ 18/20`. See §4.

### R2 (semantic_gain non-empty)
- **PASS**: `|semantic_gain (Dense vs SFT)| ≥ 10` records.
- **FAIL**: < 10 → C2 contrast confidence reduced; report effect-size with full
  uncertainty; do not overclaim.

### R3 (Dense G=8 not generic-bad — only relevant if B4 launches)
- **PASS**: `EX_att(Dense G=8) ≥ EX_att(Dense G=4) − 1pp` within bootstrap CI.
- **FAIL**: SKE-specific story weakened to "rollout-budget tradeoff".

## 6. E3 status (R5 reviewer fix)

E3 is **NOT** a C2 pass condition. The per-`skeleton_class` advantage shifts E3
needs are **not persisted** by the existing SKE-G4 / SKE-G8 training runs.

Verified: `train_grpo.py` only writes aggregate counters to disk
(`ske_used_steps`, `ske_n_real_classes_sum`, `ske_fallback_reasons`). The per-class
shifts (`class_means`, `class_baseline`, `class_var`) live inside
`compute_class_aware_advantage` and are never serialized in the existing runs.

**Default path (chosen)**: E3 dropped from C2 pass condition; reported as Tier-3
aggregate descriptive comparison for transparency only.

**Optional re-run (NOT in this paper's scope)**: add per-`skeleton_class` adv-shift
logging to `train_grpo.py`, re-train SKE-G4 and SKE-G8 (~48 GPU-h), then run E3 with
Cliff's δ + pre-registered margin |δ| ≤ 0.147 (Vargha-Delaney). Exceeds the 3-week
budget. Explicitly deferred.

## 7. Dense G=8 (B4) launch gate

B4 is **GATED** on:

1. **C1 PASS** (`lenient_attribution_gap > 0`, 95% CI excluding 0), AND
2. **E1 PASS** (SKE-G4 `transition_bucket_share[semantic_gain]` ≤ Dense's, with CI).

If either fails, B4 is **SKIPPED** and the ~28 GPU-h is reallocated to writing or to
the optional appendix probes (3B Dense same-launcher, external-baseline). This
decision is recorded in `refine-logs/EXPERIMENT_TRACKER.md` at end-of-Week-1
inspection.

## 8. Equivalence margin (Tier-3 descriptive E3 only)

Even though E3 is descriptive, when reporting Cliff's δ on aggregate stratified
samples, use `|δ| ≤ 0.147` (Vargha-Delaney "small" effect threshold) as the
descriptive equivalence margin. State explicitly that this is a description, not
inference.

## 9. Order of operations (locked)

1. Module scaffold + tests (Day 1) — no data inspection.
2. Hash verification (every script).
3. A0 quantitative gate (50-record `parseable_lenient_rate`).
4. A0 qualitative gate (20-record manual audit).
5. A1 full lenient pass (1034 × {SFT, Dense75}).
6. A2 classifier + transition table.
7. A3 bootstrap CIs.
8. **End-of-Week-1 keystone gate**: C1 PASS / INVERT / ANOMALY.
9. B1 lenient pass (1034 × {SKE-G4, SKE-G8}) — runs unconditionally.
10. B2 classifier on SKE pairs + Tier-3 descriptive E3.
11. **End-of-Day-7 inspection**: E1 + E2 result; B4 gate decision.
12. (Conditional) B4 Dense G=8 train + eval + classifier.
13. Writing.

Any deviation from this order is recorded as a deviation in
`refine-logs/EXPERIMENT_TRACKER.md` with a paragraph explaining why.

## 10. What we will NOT do

- Multi-seed averaging.
- Re-tag bucket assignments using gold-conditioned features.
- Add new RL methods or reward variants.
- Reproduce HES-SQL / MTIR-SQL / MARS-SQL training (only consume public artifacts
  if available).
- Change bucket rules after seeing per-record outcomes.
- Drop A0 gates after seeing A1 results.

— locked 2026-04-29 by micuks

---

# v3 Addendum — falsified v2, exploratory turn-progression pivot, B4 confound prereg

**Date locked**: 2026-04-30
**Status**: ADDENDUM, supersedes §§3-7 above for v3 analyses; §§1-2 (data hashes,
bootstrap protocol) carry forward unchanged.
**Companion docs**: `MAIN_CONF_PLAN_2026-04-30.md` (3-GPU-week plan),
`RESEARCH_REVIEW_v3_2026-04-29.md` (round-1+2 reviewer findings),
`FINAL_PROPOSAL_v3_DRAFT.md` (proposal), `outputs/robustness_pack.json`
(Shapley-sym + R_same + 3×3 paired-ΔEX outputs).

## v2-1. v2 falsification (recorded)

The Phase-A0 lenient extractor implementation was completed and run on all four
locked eval JSONs. Outcome:

- `lenient_extract` returned the same SQL string as `extract_final_sql` on:
  - 1034/1034 SFT records
  - 1033/1034 Dense75 records (1 differs)
  - 1034/1034 SKE-G4-75 records
  - 1033/1034 SKE-G8-75 records
- `protocol_gain` bucket is empty by construction.
- A 50-record exec-aware lenient spike (Option 2) found 3/50 differing — gate was 10/50, killed.

The v2 "Protocol Beats Structure" thesis is therefore **falsified by data**. The
analyses that follow are **exploratory**, not confirmatory of any v2 claim.
Records of the falsification: `A0_PRE_AUDIT_FINDING_2026-04-29.md`,
`A0_OPTION_2_3_SPIKE_2026-04-29.md`.

## v3-1. v3 axis (turn-share / per-turn-quality decomposition)

Replace v2's lenient-vs-strict axis with the turn-progression axis:

```
total_ΔEX = share_shift_term + per_turn_gain_term       (per resample, exact)
```

Three orderings are produced by `miniLLM/diag/decomposition.py`. The
**Shapley-symmetric average ("sym")** is the **primary** reported quantity;
Ordering A and Ordering B are reported in the appendix as a sensitivity check.

Closed forms for binary turn binning `k ∈ {1, 2+}`:

```
sym:    share = (s_T - s_B) · ((b1 - b2) + (t1 - t2)) / 2
        per_turn = ((s_B + s_T)/2)·(t1 - b1) + (1 - (s_B + s_T)/2)·(t2 - b2)

A:      share = (s_T - s_B) · (b1 - b2)
        per_turn = s_T · (t1 - b1) + (1 - s_T) · (t2 - b2)

B:      share = (s_T - s_B) · (t1 - t2)
        per_turn = s_B · (t1 - b1) + (1 - s_B) · (t2 - b2)
```

The bootstrap protocol from §2 above carries unchanged: 1000 resamples, paired
by record id, percentile-95 CI, seed=20260429.

## v3-2. R_same paired ΔEX (composition decontamination)

For each pair (T vs B), report the paired EX delta restricted to
`R_same = {i : turn_T(i) = turn_B(i)}`. Within R_same, per-turn case mix is
held fixed by construction; a non-zero paired ΔEX cannot be a composition
artifact of share-shift.

`R_same` is reported with paired bootstrap CI in aggregate AND broken out by
SFT turn count.

## v3-3. v3 claims (operational definitions)

### V1 — share_shift dominates aggregate gain
- **PASS**: For at least one (Dense, SKE-G4, SKE-G8) vs SFT pair, the
  `share_shift_sym` 95% CI excludes 0 in the positive direction AND the
  qualitative ordering of `share_shift_sym` magnitudes is consistent across A
  / B / sym (Spearman rank correlation of point estimates ≥ 0.9).
- **FAIL**: All three pairs have `share_shift_sym` CIs overlapping 0.

### V2 — per-turn-quality story (cautious; descriptive language only)
- **REPORT**: `per_turn_sym` point + 95% CI for each pair; `R_same` paired
  ΔEX point + 95% CI for each pair.
- **CLAIM ALLOWED IF**: For the SKE-G8 pair, BOTH (a) `per_turn_sym` 95% CI
  excludes 0 in the negative direction AND (b) `R_same` paired ΔEX 95% CI
  overlaps 0 OR is in the negative direction.
- **CLAIM FORBIDDEN IF**: `R_same` paired ΔEX 95% CI strictly excludes 0 in
  the positive direction (this would mean the negative `per_turn_sym` is purely
  composition-driven and SKE-G8 actually does improve same-turn SQL quality).

### V3 — difficulty localization
- **REPORT**: Per-difficulty (easy/medium/hard) Shapley-sym decomposition with
  CIs. Strata sizes Easy=333, Medium=377, Hard=324.
- No pre-registered hypothesis on direction; descriptive only.

## v3-4. B4 (Dense G=8 confound run) — pre-registered, FALSIFIED 2026-04-30

**Status update (2026-04-30 evening)**: B4 was launched and completed
between the locking of this addendum and end-of-day. Result file:
`outputs/eval_full_dev_dense_g8_ck75.json`
(SHA256: `de48eb9dc0bbf3cb6d83771332fb458798393e6d35974f262f5db24b9faa402f`).
Outcome documented in `B4_RESULT_v3_PREDICTION_FALSIFIED_2026-04-30.md`.

**Pre-registered prediction (locked verbatim below)**: FALSIFIED.

- `share_shift_sym` ∈ [+1.8, +2.4] — observed +2.52 (overshoots band; CI > 0 ✓ but outside the predicted window).
- `per_turn_sym` ∈ [−0.5, +0.5] with CI overlapping 0 — observed −1.85 [−3.69, −0.13], CI strictly negative ✗.

**Decision rule applied**: per the prereg's "If Dense G=8 shows a negative
per_turn_sym with 95% CI excluding 0, we will drop the SKE-specific claim
and rewrite the result as evidence for a generic large-G over-commit/quality
tradeoff." Decision applied: SKE-specific framing **dropped**;
proposal **pivoted to v4** ("Group size G trades commitment for capability"
— see `FINAL_PROPOSAL_v4.md`).

The verbatim prediction text is preserved below as the audit record of
what was predicted vs observed:



This block is the verbatim B4 pre-registration from
`RESEARCH_REVIEW_v3_2026-04-29.md` round 2. Locked here BEFORE any B4 training
launch; subsequent commits to git serve as the timestamp of record.

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

**Training config** (locked):
- Same launcher / data / optimizer / LR schedule / reward / prompt format / rollout limits as the existing Dense G=4 run.
- Same training seed where deterministic.
- Single change: `G: 4 → 8`.
- Same 100-step schedule; **ckpt75 = primary endpoint** (matches comparison set).
- ckpt100 reported as secondary sensitivity.
- Do **not** scale steps to equalize total sampled rollouts; this is a
  mechanism-control run, not a compute-equalized scaling study.

## v3-5. Multi-seed replication — pre-registered

Replicate **Dense G=4, Dense G=8, SKE G=8** at training seeds **{20260429,
42, 12345}** (3 seeds each). Same launcher, only training seed changes. Eval
each seed at ckpt75 on full Spider-dev. Bootstrap CIs for `share_shift_sym`
and `per_turn_sym` are computed across seeds (record-resampled within seed,
seeds treated as fixed).

**Pass criterion** (kill switch for main-conf attempt): For SKE-G8 across the
3 seeds, `per_turn_sym` 95% CI in **at least 2 of 3 seeds** excludes 0 in
the negative direction. Failing this kills the SKE-specific claim and
demotes the paper to workshop note.

## v3-6. Stop-Calibrated reward intervention (Block B) — pre-registered

```
R_stop-cal = R_EX
           + λ · 1[turn=1 ∧ EX=1]
           - λ · 1[turn=1 ∧ EX=0]
```

**Pilot**: λ ∈ {0.05, 0.10}, single seed, 100 steps. Pick λ* with the higher
ckpt75 EX_att.

**Replication**: 3 Spider seeds at λ*.

**Inference-time gate** (zero training): if turn=1 wants `Answer:` but no
preceding `Action: execute_sql[...]` produced a non-error Observation,
force-continue one more turn. Apply to existing SKE-G8 ckpt75 inference.

**Pass criterion** (intervention claim allowed): on Spider Hard, the
intervention's `per_turn_sym` 95% CI strictly improves over SKE-G8's
`per_turn_sym` 95% CI (CI gap, not overlap). On Spider full-dev, intervention
preserves `share_shift_sym` (CI > 0) AND attains `per_turn_sym` 95% CI
overlapping or above 0.

## v3-7. BIRD cross-benchmark (Block C) — pre-registered

Train SFT + Dense G=8 + SKE G=8 + intervention each at 2 BIRD seeds. Eval on
BIRD-dev with the same Shapley-sym + R_same + difficulty-stratified
(`simple/moderate/challenging`) decomposition. **Pass**: at least the
share-shift-dominant pattern reproduces on at least 1 of {Dense G=8, SKE G=8};
intervention's improvement on Hard / challenging direction is preserved.

## v3-8. Spider-Realistic eval-only (Block D) — pre-registered

Eval-only on Spider-Realistic. Apply existing 4 checkpoints + B4 + multi-seed
runs once available. **Reported as descriptive robustness**; no pass/fail.

## v3-9. Kill switches for main-conference attempt

If any of these holds at the relevant decision point, the paper is demoted to
workshop note and Block B/C/D may be cut:

1. **Multi-seed unstable** (after Block A): SKE-G8 `per_turn_sym` is not
   stably negative across seeds (≤ 1 of 3 seeds shows CI excluding 0
   negative).
2. **B4 negative AND intervention fails**: Dense G=8 `per_turn_sym` 95% CI
   excludes 0 in the negative direction, AND stop-calibrated intervention does
   not preserve share-shift while reducing the per-turn regression.
3. **BIRD generality fails**: share-shift-dominant pattern absent on BIRD AND
   intervention has no effect on BIRD.

`B4 negative alone does NOT trigger demotion`; the trigger is the conjunction
of B4 negative + multi-seed unstable + intervention failure.

## v3-10. Forbidden language (audit list)

The following phrasings are forbidden in any v3 draft, abstract, or response
to reviewers:

- "v3 thesis" (implies preregistration before data).
- "39% / 80% / 100% of the gain" (reference-dependent ordering claim).
- "SKE provides no signal that improves per-turn quality" (over-strong;
  permitted: "under this decomposition, we observe no net per-turn improvement").
- Anything that lets the v2 preregistration credibility leak into v3 claims.

— locked 2026-04-30 by micuks
