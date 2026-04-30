# Research Proposal: Protocol Beats Structure — A Diagnostic Study of Agentic GRPO for Text-to-SQL

**Status**: READY (overall 9.00/10 after 4 review rounds, gpt-5.5 xhigh)
**Date**: 2026-04-29
**Supersedes**: `FINAL_PROPOSAL_SKE_RL_v1_archived_2026-04-29.md` (the SKE-RL-as-method proposal whose empirical result demanded a reframe).

---

## Problem Anchor (immutable)

- **Bottom-line problem**: Multi-turn ReAct GRPO on Text-to-SQL succeeds modestly at 7B (+2 pp Spider-dev EX_att/EX_all over 7B SFT) but fails at 3B; SKE-RL (a structural-advantage prior) fails to help and worsens with more rollouts (G=4 → -1.04 pp vs Dense; G=8 → -1.81 pp). We do not know what the +2 pp actually changed inside the agent, and we cannot explain why a natural structural prior fails.
- **Must-solve bottleneck**: Diagnostic gap. Recent agentic Text-to-SQL RL literature reports aggregate EX gains without decomposing protocol-compliance vs SQL-reasoning contributions, so neither positive (Dense GRPO) nor negative (SKE) results can be interpreted mechanistically.
- **Non-goals**: SOTA on Spider/BIRD; new RL method that wins; scaling-law claim across model sizes; toolkit-as-lead paper.
- **Constraints**: 1× Quadro RTX 6000 24 GB; ~3 weeks GPU + 1 week writing; SKE-RL stack + per-class diagnostics + full Spider-dev eval JSONs already on disk; workshop-tier deadline.
- **Success condition**: A reviewer would say the paper changes how they read agentic-SQL-RL leaderboards (because of the protocol decomposition) AND the SKE negative result has a single, convincingly measured mechanism (not "we tried it and it didn't work").

## Technical Gap

Recent agentic Text-to-SQL RL work — HES-SQL, MTIR-SQL, MARS-SQL, Graph-Reward-SQL, ReEx-SQL — reports +N pp EX_all gains and treats those gains as evidence of improved SQL reasoning under structural / execution / multi-agent supervision. **No paper in this line decomposes the gain into "the model produced more valid trajectories" vs "the model produced semantically better SQL".** The two are systematically conflated under EX_all because EX_all = (attempted_rate × EX_att), and aggregate metrics give attempted-rate gains the same weight as semantic gains.

This conflation has direct consequences:
- (a) Existing structural-reward methods (HES-SQL) cannot distinguish whether their gain comes from structural alignment or coincidental protocol cleanup.
- (b) Negative results like our SKE-RL (advantage-side cousin of HES-SQL skeleton reward) are uninterpretable beyond "it didn't work".
- (c) Naive scaling — more rollouts, more reward dimensions, multi-agent decomposition — keeps reporting the same conflated metric.

The smallest adequate intervention is a **measurement protocol**, not a new training method.

## Method Thesis

- **One-sentence thesis**: Agentic GRPO on multi-turn Text-to-SQL improves aggregate accuracy primarily by aligning tool-use protocols (valid action format, valid SQL syntax, valid Answer extraction, repair-after-observation), not by improving SQL semantic reasoning — which explains why a structural-advantage prior (SKE-RL) cannot help: it operates on the layer below the actual bottleneck.
- **Smallest adequate intervention**: A deterministic 3-bucket trajectory classifier + leakage-safe lenient re-execution + signed attribution gap with bootstrap CIs. **Zero new training for the dominant contribution.** Existing 4 checkpoints become the data; the contribution is the lens through which they are read.
- **Why timely**: Foundation-model-era agentic RL — web agents, code agents, tool-using assistants — all reports aggregate task-success gains. The protocol-vs-semantic conflation we identify on Text-to-SQL almost certainly applies to those settings. The paper offers a portable diagnostic, not a SQL-specific trick.

## Contribution Focus

- **Dominant contribution**: A deterministic, portable diagnostic decomposing any agentic Text-to-SQL RL gain into a **protocol-validity component** and an **SQL-semantic component**, with a single signed headline number (`lenient_attribution_gap`) carrying a 95% bootstrap CI. Applied to our 7B Dense GRPO, it shows the +2 pp gain is dominantly protocol.
- **Optional supporting contribution**: A mechanism-grounded negative result for SKE-RL — fails because the bottleneck is at the protocol layer it cannot affect; G=8 monotonic degradation is the predicted consequence. Confirmed via three pre-registered evidence requirements (E1 + E2 + E3).
- **Explicit non-contributions**: M1-M4 mechanism grid (appendix checklist only); new RL method; scaling law; SOTA; toolkit-as-lead.

## Proposed Method

### Complexity Budget
- **Frozen / reused**: All 4 existing 7B checkpoints (SFT, Dense75, SKE-G4-75, SKE-G8-75); existing eval JSONs with full per-record trajectories; existing skeleton extractor; existing query-class tagger; existing aggregator; existing SKE training-time per-class logs (n_real_classes, class_size_distribution, advantage variance per class — all captured by R3 patches).
- **New trainable components**: 0 for dominant contribution. 1 supporting (Dense G=8 confound check, ~1.2 run-eq).
- **Tempting additions intentionally not used**: Multi-seed averaging; BIRD GRPO; 3B Dense same-launcher rerun (appendix-only-if-time); 1.5B / 14B capacity points; v2 SKE; mechanism grid as main claim.

### System Overview

```
Existing eval JSON  ──►  Step 1: Strict + LEAKAGE-SAFE Lenient extraction
   (any agentic        │      strict_pred_sql = current extract_final_sql()
    SQL RL run)        │      lenient_pred_sql = first sqlglot-parseable SQL in
                       │                         final-Answer-region of trajectory;
                       │                         else last sqlglot-parseable SQL action;
                       │                         else None
                       │                         (selection NEVER consults gold_sql)
                       │      Re-execute both against db_path AFTER selection
                       ▼
                       Step 2: Per-record classifier (deterministic, mutually exclusive)
                       protocol_gain = (not SFT.final_correct_strict
                                        and SFT.final_correct_lenient
                                        and RL.final_correct_strict)
                       semantic_gain = (not SFT.final_correct_lenient
                                        and RL.final_correct_lenient
                                        and RL.final_correct_strict)
                       no_change     = otherwise
                       precedence: protocol_gain > semantic_gain > no_change
                       (R5 fix: both gain buckets are subsets of strict-gain transitions)
                       ▼
                       Step 3: Aggregate diagnostic (95% bootstrap CIs, 1000 resamples)
                       PRIMARY:   lenient_attribution_gap = ΔEX_strict − ΔEX_lenient (pp, signed)
                       SECONDARY: attribution_ratio = gap / max(|ΔEX_strict|, 0.5pp), signed
                                  (suppressed when |ΔEX_strict| < 0.5 pp)
                       transition_bucket_share[B] = |B| / |strict-gain transitions|
                       global_bucket_rate[B]      = |B| / 1034
                       ▼
                       Step 4: SKE confirmation (supporting)
                       Apply Step 2-3 to (SKE-G4 vs Dense), (SKE-G8 vs Dense)
                       Test E1 + E2 + E3 (with explicit skeleton_class join + Cliff's δ equivalence)
                       Confound: Dense G=8 row (single row, not centerpiece)
```

### Core Mechanism — leakage-safe extractor + direct-correctness bucket rules

```python
import re
import sqlglot

def lenient_extract(trajectory: str) -> str | None:
    """Pure trajectory → SQL. Never consults gold_sql."""
    ans = re.search(r"Answer:\s*(.*?)(?:\nObservation:|\Z)", trajectory, re.DOTALL)
    if ans:
        for cand in _sql_candidates_in_text(ans.group(1)):
            try:
                if sqlglot.parse_one(cand, dialect="sqlite") is not None:
                    return cand
            except Exception:
                continue
    for cand in reversed(_sql_action_strings(trajectory)):
        try:
            if sqlglot.parse_one(cand, dialect="sqlite") is not None:
                return cand
        except Exception:
            continue
    return None


def classify_pair(sft_record: dict, rl_record: dict) -> tuple[str, dict]:
    """
    Inputs (per-side, computed from each record):
      strict_pred_sql, lenient_pred_sql, gold_sql, all_sql_actions,
      execution_errors, final_answer_span, db_id

    Per-side flags (computed identically for sft and rl):
      strict_extract_ok       = strict_pred_sql is not None and sqlglot-parses
      lenient_extract_ok      = lenient_pred_sql is not None  (parses by construction)
      first_sql_exec_ok       = execution_errors[0] == "" if all_sql_actions else False
      final_correct_strict    = exec_match(strict_pred_sql, gold_sql)  if strict_extract_ok  else False
      final_correct_lenient   = exec_match(lenient_pred_sql, gold_sql) if lenient_extract_ok else False
      observation_repair_flag = (not first_sql_exec_ok) and final_correct_strict
      equiv_class_change_flag = (canonical_ast(SFT.strict_pred) != canonical_ast(RL.strict_pred))

    Bucket rules (precedence: protocol_gain > semantic_gain > no_change).
    Both gain buckets are subsets of the strict-gain transition set
    (RL.final_correct_strict = True), so transition_bucket_share has a
    well-defined denominator (R5 reviewer fix).

      protocol_gain = (not SFT.final_correct_strict
                       and SFT.final_correct_lenient
                       and RL.final_correct_strict)

      semantic_gain = (not SFT.final_correct_lenient
                       and RL.final_correct_lenient
                       and RL.final_correct_strict)   # R5 fix: also require strict

      no_change     = otherwise

      Subanalysis flag (NOT a bucket):
        lenient_only_repair_flag = (not RL.final_correct_strict
                                    and RL.final_correct_lenient)
        # captures "RL almost fixed it but final extraction was malformed"
        # Reported in appendix only; not in the headline bucket distribution.
    """
```

### Headline Numbers (with 95% bootstrap CIs, 1000 resamples)

| # | Metric | Formula | Primary use |
|---:|---|---|---|
| 1 | `lenient_attribution_gap` | ΔEX_strict − ΔEX_lenient (signed, in pp) | **PRIMARY** — large positive = gain mostly protocol; ~0 = mostly semantic; negative = anomaly flag |
| 2 | `attribution_ratio` | `gap / max(|ΔEX_strict|, 0.5pp)`; signed; no clipping | Secondary, suppressed when `|ΔEX_strict| < 0.5 pp` |
| 3 | `transition_bucket_share[B]` | \|B\| / \|{SFT-strict-wrong → RL-strict-right}\| | "of strict-gain transitions, fraction in bucket B" |
| 4 | `global_bucket_rate[B]` | \|B\| / 1034 | "fraction of dev set with outcome B" |
| 5 | `observation_repair_share` | \|protocol_gain ∩ obs_repair_flag\| / \|protocol_gain\| | Appendix subanalysis only |

**Why this is the main novelty**: No published agentic Text-to-SQL RL paper reports any of `lenient_attribution_gap`, `attribution_ratio`, `transition_bucket_share`, or `global_bucket_rate`. All four are interpretive primitives, not engineering tricks. They generalize trivially to other tool-using agentic RL settings by swapping `gold_sql` → `task_success_predicate` and `execute_sql` → arbitrary `tool_call`.

### Optional Supporting Component (= SKE single mechanism, with explicit E3 join + equivalence test)

Apply the same classifier to (SKE-G4 vs Dense) and (SKE-G8 vs Dense). Test the single prediction:

> **SKE-RL fails because the bottleneck is at the protocol layer it cannot affect.**

**C2 pass condition** (R5 reviewer fix): E1 + E2 must both hold. E3 is **OPTIONAL supplementary** — see asset-availability constraint below.

- **E1**: `bucket_share[semantic_gain]` for (SKE vs Dense) ≤ that for (Dense vs SFT), with 95% bootstrap CIs.
- **E2**: `lenient_EX(SKE) ≤ lenient_EX(Dense)` within bootstrap CI.
- **E3 (OPTIONAL — only if input data exists)**: SKE training-time per-`skeleton_class` advantage shifts are statistically equivalent across dev `skeleton_class`-buckets stratified by outcome density.
  - **Asset availability check (R5 reviewer fix)**: The current `train_grpo.py` only persists aggregate counters (`ske_used_steps`, `ske_n_real_classes_sum`, `ske_fallback_reasons`). Per-`skeleton_class` advantage-shift values are computed inside `compute_class_aware_advantage` (`class_means`, `class_baseline`, `class_var`) but **never written to disk in the existing SKE-G4 / SKE-G8 runs**. Therefore E3 cannot be executed on the existing checkpoints.
  - **Two paths forward**:
    - **Default (chosen)**: E3 is dropped from C2 pass condition; reported as a documented limitation. C2 stands on E1 + E2 alone.
    - **Optional re-run (NOT in this paper's scope)**: add per-`skeleton_class` advantage-shift logging to `train_grpo.py`, re-train SKE-G4 and SKE-G8 (~48 GPU-h, exceeds the 3-week budget), then run E3 with Cliff's δ + pre-registered margin |δ| ≤ 0.147 (Vargha-Delaney).
  - **Tier-3 fallback (always reported)**: aggregate descriptive comparison of training-time `ske_n_real_classes_sum` and `class_size_distribution` between high-density-`semantic_gain` and high-density-`protocol_gain` skeleton classes. Reports skeleton-class-frequency match between the two strata as a *necessary-but-insufficient* check that SKE training "saw" both kinds of classes equally.

**Confound check (Dense G=8)**: One Dense G=8 run (~1.2 run-eq). Gives the 2×2 grid {Dense, SKE} × {G=4, G=8}. If `EX_att(Dense G=8) ≥ EX_att(Dense G=4) − 1pp` (within bootstrap CI), SKE-G=8's collapse is structural to class-baseline math, not generic. **Single row in SKE table, not a centerpiece.**

**M1-M4 mechanism grid**: Appendix-only checklist for future negative-result analyses ("a diagnostic checklist for agentic-RL negative results"). Does NOT appear in proposal body or main paper sections.

### Modern Primitive Usage
- **Lenient re-execution** = real SQLite as inference-time verifier — recovers what the agent meant despite protocol noise. Selection is leakage-safe (no gold consultation).
- **AST-canonical equivalence (sqlglot)** = used for `equiv_class_change_flag` and as the `skeleton_class` join key in E3.
- **No new trained component** for the dominant contribution. Leverage is observational.

### Integration
The diagnostic attaches **after** an arbitrary agentic-RL pipeline. Reads eval JSONs only. No re-training, no model surgery. Reference implementation is one Python file consuming the listed input fields.

### Training Plan
- **Dominant contribution**: NONE — no training.
- **Supporting (SKE mechanism)**: One Dense G=8 confound run (~1.2 run-eq), 75 steps, identical recipe to Dense G=4 except group size.

### Failure Modes and Diagnostics

| Failure mode | How we detect | Fallback |
|---|---|---|
| `lenient_attribution_gap` ≈ 0 (gain fully semantic) | Lenient ΔEX ≈ Strict ΔEX with overlapping CIs | Story flips: gain IS semantic, SKE still fails (concentrated in queries SKE cannot reach). Diagnostic still novel; conclusion inverts; narrative reframed in writing phase. |
| `lenient_attribution_gap` negative outside CI | Lenient gain > strict gain | Anomaly: extractor over-recovering. Re-validate extractor on 50-record sample. If confirmed valid, decomposition is non-monotone: report and discuss as a finding (lenient view picks up semantic gain hidden by strict's protocol-failure miss). |
| Dense G=8 also degrades vs Dense G=4 by >1 pp | EX_att(Dense G=8) < EX_att(Dense G=4) − 1pp outside CI | Generic G effect; reframe as "rollout-budget vs quality trade-off in agentic GRPO". SKE single-mechanism story weakened. |
| Both gain buckets <5% of records | Most transitions in `no_change`; CIs include 0 | Honest CI reporting; possible "Dense gain is mostly noise" finding. Reduce confidence and flag as workshop-only. |
| Lenient extractor pathological | Lenient EX(SFT) < Strict EX(SFT) on 50-record sanity sample | Tighten extractor to "Answer-region only" (drop last-action fallback); re-validate. |
| `|ΔEX_strict| < 0.5 pp` (denominator unstable) | Bootstrap CI of ΔEX_strict crosses 0 | Suppress `attribution_ratio`; report only `lenient_attribution_gap` and bucket counts. |
| E3 inconclusive (CI of \|δ\| straddles 0.147) | Bootstrap CI extends past margin | Honest report: "we cannot rule out a small but non-trivial relationship between SKE class-baseline shifts and semantic outcomes". SKE single-mechanism claim weakened to "consistent with but not proving" the protocol-bottleneck thesis. |
| E3 join coverage <50% | <50% of dev `skeleton_class` values appear in training-time SKE logs | Tier-3 aggregate-only fallback for E3; note as limitation. |
| External-baseline artifact unavailable | 1-day search of HES-SQL/MTIR-SQL/MARS-SQL public logs returns no per-record trajectories | Drop external probe; note as limitation. Diagnostic remains internally validated. |

### Novelty and Elegance Argument

**Closest published work**:
- **HES-SQL** (skeleton REWARD): targets structural alignment via reward; reports +N pp; does not decompose attribution.
- **Graph-Reward-SQL** (graph-matching reward): same conflation issue.
- **MTIR-SQL** (multi-turn tool-integrated GRPO): aggregate EX only.
- **MARS-SQL** (multi-agent ReAct + validation): aggregate EX only.
- **LearNAT** (AST decomposition + DPO): structural reward-side; no diagnostic.
- **ReEx-SQL** (execution-aware RL with intermediate DB feedback): aggregate EX only.

**Exact difference**: Every above paper sits inside the EX_all conflation — they propose interventions and show aggregate gains. We do **not** propose a winning intervention. We propose a diagnostic that turns those existing aggregate gains into mechanistic claims, and use SKE-RL (a natural advantage-side cousin of HES-SQL) as a designed-to-fail control that *makes the diagnostic visible in both directions* (positive on Dense, null on SKE).

**Why focused, not module pile-up**: One classifier + one ratio + one bucket distribution + one supporting failure analysis (predicted by the main thesis). Zero new training for the dominant contribution. The whole paper rotates around exactly one question — "what does agentic SQL RL actually change?" — with exactly one new way to answer it.

**Why elegant**: The diagnostic is one leakage-safe re-execution pass + one deterministic classifier + one signed gap with CI. Anyone with eval JSONs from any agentic Text-to-SQL RL paper can apply it tomorrow. That portability is the elegance.

## Claim-Driven Validation Sketch

### Claim 1 (dominant): Dense agentic GRPO gain on Text-to-SQL is dominantly protocol/tool-use alignment

- **Minimal experiment**: Lenient-extraction re-execution of SFT vs Dense75 on full Spider-dev (1034). Compute `lenient_attribution_gap`, `attribution_ratio`, and bucket distributions with 95% bootstrap CIs.
- **Cost**: ~6 GPU-h (SQLite-bound re-execution); 0 for classifier + bootstrap.
- **Sanity check**: lenient_EX(SFT) ≥ strict_EX(SFT) on 50-record pre-pipeline sample.
- **Metric**: signed `lenient_attribution_gap` in pp; `attribution_ratio` if denominator stable; `transition_bucket_share` and `global_bucket_rate` for protocol_gain vs semantic_gain.
- **Expected directional outcome**: `lenient_attribution_gap` > 0 with CI excluding 0 (≥ half the gain disappears under lenient eval); `transition_bucket_share[protocol_gain] > transition_bucket_share[semantic_gain]` with non-overlapping CIs.

### Claim 2 (supporting): SKE-RL fails because the bottleneck is at the protocol layer it cannot affect

- **Minimal experiment**: Apply the same diagnostic to (SKE-G4 vs Dense) and (SKE-G8 vs Dense). Test E1 + E2 with bootstrap CIs. **Dense G=8 confound run is conditional** (only if C1 supports protocol-dominant thesis). E3 is dropped from pass condition (per-class adv shifts not persisted; see Method §Optional Supporting Component); reported as Tier-3 aggregate descriptive comparison for transparency.
- **Baseline / ablation**: 2×1 grid {Dense, SKE} × {G=4} mandatory; G=8 column added only if Dense G=8 confound is launched.
- **Metric**: `bucket_share[semantic_gain]` comparison; `lenient_EX` comparison; (Tier-3) descriptive `skeleton_class`-frequency match between strata.
- **Expected directional outcome (C2 PASS)**: E1 AND E2 hold. Dense G=8 (if run) ≥ Dense G=4 − 1pp.

## Experiment Handoff Inputs

- **Must-prove claims**: C1 (protocol attribution); C2 (SKE single mechanism via E1 + E2 + E3).
- **Must-run ablations**: Dense G=8 confound; lenient-extraction sanity on SFT (50-record); bootstrap CIs on all headline numbers.
- **Critical datasets / metrics**: Spider-dev full 1034; `lenient_attribution_gap`; `attribution_ratio`; `bucket_share` (both views); `lenient_EX`; E3 Cliff's δ.
- **Highest-risk assumptions** (R5 reviewer fix to R1, R4):
  - **(R1)** Lenient extractor produces a parseable SQL on ≥80% of randomly sampled format-failed SFT trajectories (50-record sample). PLUS: manual audit of 20 records confirms the extractor returns the *intended* final SQL (not Observation text or non-final action SQL).
  - **(R2)** `semantic_gain` ≥ 10 transitions in (Dense vs SFT) — else C2 has no contrast to test against.
  - **(R3)** Dense G=8 ≥ Dense G=4 within 1 pp — else G effect confounds SKE story. **(Conditional: only run Dense G=8 if C1 supports the protocol-dominant thesis; see EXPERIMENT_PLAN.md decision tree.)**
  - **(R4)** *(Removed — E3 dropped from C2 pass condition because per-`skeleton_class` advantage shifts were never persisted by the existing SKE training runs.)* Tier-3 aggregate-only descriptive comparison is reported for transparency.
- **Reviewer-required execution discipline**:
  1. Implement the extractor exactly as specified — no gold-conditioned candidate selection.
  2. Pre-register the E3 equivalence margin (|δ| ≤ 0.147) and bootstrap procedure **before** running final diagnostics on the SKE arms.
  3. Make the main paper's first figure/table the strict-vs-lenient decomposition; SKE is supporting interpretation after.

## Compute & Timeline Estimate

| Block | GPU-h | Wall (1×24GB) | Note |
|---|---:|---|---|
| Lenient re-execution: SFT + Dense75 on full Spider-dev | ~6 (SQLite-bound) | <1 day | Existing trajectory logs |
| Classifier + transition table (SFT vs Dense) | ~0 | <1 day | Pure CPU |
| Bootstrap CIs (1000 resamples × all headline numbers) | ~0 | <0.5 day | Pure CPU |
| SKE diagnostic (apply to G4 + G8 vs Dense, E1 + E2 + E3) | ~6 | ~1 day | Existing eval JSONs + lenient pass + skeleton_class join |
| Dense G=8 confound run | ~28 (1.2 run-eq + eval) | ~1.5 days | Single confound check |
| Writing + figures | n/a | ~1 week | After diagnostics |
| (Appendix-only-if-found) External-baseline probe | ~2 if available | <1 day | 1-day artifact-search gate; drop if no public logs |
| (Appendix-only-if-time) 3B Dense same-launcher | ~12 | ~1 day | Anchor 3B cliff context only |

**Mandatory total: ~40 GPU-h (1.7 run-eq) + 3 days CPU + 1 week writing.**
**With both optional appendix items: ~54 GPU-h (2.3 run-eq).**

Both fit comfortably in the 3-week budget.
