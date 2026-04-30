# Reviewer Pass R5 — SKE-RL Proposal Patch Verification
**Date**: 2026-04-24
**Reviewer**: Independent pass verifying 5 R5 critiques applied by the author.
**Files inspected**:
- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`

All grep results below were obtained by direct file inspection. No author summaries were trusted.

---

## H-1: Phase 1.2 Diagnostic — Model-rollout Variance, Not Gold

**Status: YES — fixed.**

The file now contains an explicit header flagging the correction:

> `refine-logs/FINAL_PROPOSAL.md`, line 267:
> ```
> **Run 1.2 Within-class outcome variance — over MODEL ROLLOUTS, NOT gold** (R5 reviewer fix):
>
> The original formulation `σ²(gold_executes_correctly | z)` was meaningless — gold SQL by
> definition executes correctly, so variance is ~0 and the gate trivially passes. Replaced with:
> ```

The replacement procedure samples K=4 SFT rollouts per query at training temperature and computes `p_q` = fraction of rollouts that execute correctly. Within-class variance is then `Var_{q in z}(p_q)`. Gate 1.2 is: median within-class variance ≤ 0.15.

A grep for the old pattern (`gold_executes_correctly`, `gold SQL execution variance`, `gold always executes`, `executes_correctly`) confirmed the only hit is the line that documents the old bug as a historical note inside the fix description — it is not used as an operative gate criterion anywhere. No residual use of the old formulation exists.

**No new inconsistency introduced.**

---

## H-2: EXPERIMENT_TRACKER.md — R4 Staged Design (A1/A2/A3 + Phase 1.5 Gate)

**Status: YES — fixed.**

A grep for `C1`, `C2`, `C3`, `C4` in EXPERIMENT_TRACKER.md returns exactly one hit:

> `refine-logs/EXPERIMENT_TRACKER.md`, line 10:
> ```
> R3 4-way factorial (C1/C2/C3/C4) is **deprecated**. R4 staged design uses A1/A2/A3 in v1
> with optional v2/v3 conditional.
> ```

This is a deprecation notice, not an active use. No table or run entry uses C1/C2/C3/C4 as active experiment labels.

The tracker now contains:
- `## Phase 1.5: Skeleton Diversity in Actual Rollouts (Day 3, ~3 GPU-h) — NEW` (line 43)
- Gate rows: `% groups with ≥2 distinct classes (gate ≥40%)` and `Inter-class reward variance ≥0.1 (gate)` (lines 48–49)
- Run entries: `Run 3.1 — A1: raw GRPO baseline`, `Run 3.2 — A2: SKE-RL v1 σ_strict`, `Run 3.3 — A3: SKE-RL v1 σ_loose` (lines 71, 77, 86)
- Go/halt decision logic referencing A2/A3 (lines 97–98)

The R4 staged structure is fully present. **No new inconsistency introduced.**

---

## M-3: SKE_MIN_CLASSES Unified to 3

**Status: YES — fixed.**

All six occurrences of `SKE_MIN_CLASSES` across the three files:

| File | Line | Value |
|---|---|---|
| FINAL_PROPOSAL.md | 303 | `SKE_MIN_CLASSES=3` |
| EXPERIMENT_PLAN.md | 126 | `SKE_MIN_CLASSES=3` |
| EXPERIMENT_PLAN.md | 164 | `SKE_MIN_CLASSES=3` |
| EXPERIMENT_TRACKER.md | 61 | `SKE_MIN_CLASSES=3` |
| EXPERIMENT_TRACKER.md | 79 | `SKE_MIN_CLASSES=3` |
| EXPERIMENT_TRACKER.md | 88 | `SKE_MIN_CLASSES=3` |

Every occurrence is 3. No occurrence of `SKE_MIN_CLASSES=2` remains anywhere. The prose in FINAL_PROPOSAL.md line 34 also explicitly states "default **3**, raised from 2 per R4 review."

**No new inconsistency introduced.**

---

## M-4: "Gold-Shape Skeleton" Terminology + "Gold-with-Table-Set" Metric

**Status: YES — fixed.**

A grep for bare `gold skeleton` (without a qualifier like `-shape`) returned zero hits. All occurrences use the qualified form.

Confirmed occurrences of the corrected terminology:

> `refine-logs/FINAL_PROPOSAL.md`, line 289:
> ```
> fraction of groups where ANY rollout produced the **gold-shape skeleton** (matching σ(gold)
> — i.e., same arity, types, JOIN structure, aggregate signatures, BUT NOT same table-set
> or column-roles)
> ```

> `refine-logs/FINAL_PROPOSAL.md`, line 290:
> ```
> fraction with **gold-with-table-set skeleton** (stricter: same as above + table-set matches
> gold's via name normalization). This stricter metric is reported alongside but NOT the gate;
> it's a tighter sanity check on whether "matching gold-shape" actually corresponds to structural
> correctness.
> ```

The same language appears in EXPERIMENT_PLAN.md line 104 and EXPERIMENT_TRACKER.md lines 50 and 192. The distinction is correctly maintained: `gold-shape skeleton` is the gate condition; `gold-with-table-set` is the stricter reported-but-not-gated sanity check.

**No new inconsistency introduced.**

---

## M-5: Fallback "Reduces Exposure When Class Diversity Is Low" — Not "Bounds Harm"

**Status: PARTIAL.**

The primary fix is applied and correct. The risk table row for "v1 hurts performance" (FINAL_PROPOSAL.md, line 384) now reads:

> ```
> Standard-advantage fallback **only reduces exposure on low-diversity steps** (when class count
> < min_classes). On SKE-used steps, miscalibrated equivalence classes can still produce
> systematically wrong gradients. ... Fallback is exposure-reduction, not harm-bound.
> ```

This matches the demanded rephrasing exactly.

However, one row in the same risk table retains the stronger language:

> `refine-logs/FINAL_PROPOSAL.md`, line 382:
> ```
> Fallback **avoids harm on those specific low-diversity steps** by reverting to standard GRPO.
> Does NOT prevent miscalibrated equivalence on SKE-used steps.
> ```

"Avoids harm on those specific low-diversity steps" is a narrower claim than the original "bounds harm" (it is qualified to low-diversity steps only), and the caveat immediately follows. This is borderline — the wording is not technically wrong (on low-diversity steps, fallback does revert to standard GRPO and thus avoids SKE-induced gradient distortion), but it is inconsistent with the more careful phrasing used in the adjacent row. A reader scanning the risk table will see two rows with different phrasings for what is effectively the same mechanism.

**Recommended fix**: change line 382 to "Fallback **reduces exposure on those specific low-diversity steps**" to match line 384's language precisely.

This is a minor inconsistency, not a logical error. The substantive claim — fallback is not a global harm-bounding guarantee — is stated correctly in line 384 and confirmed in line 34.

---

## New Inconsistencies Introduced by Patches

One minor inconsistency was noted above (M-5, two rows in the same risk table use slightly different phrasings for the same mechanism). No other new inconsistencies were found.

No `SKE_MIN_CLASSES=2` values were reintroduced. No old `gold skeleton` bare references were reintroduced. No old tracker labels (C1–C4) were reintroduced as active entries. The Phase 1.2 gate logic is internally consistent throughout all three files.

---

## Final Verdict

Four of five critiques are fully resolved (H-1, H-2, M-3, M-4). M-5 is substantively resolved — the main risk table row uses the correct "exposure-reduction, not harm-bound" framing — but one adjacent row in the same table retains the slightly stronger "avoids harm on those specific low-diversity steps" phrasing. This is an intra-document wording inconsistency, not a logical flaw.

**If the single wording inconsistency on FINAL_PROPOSAL.md line 382 is corrected to "reduces exposure on those specific low-diversity steps", then:**

All 5 R5 critiques resolved. Plan is ready for Phase 0 implementation.
