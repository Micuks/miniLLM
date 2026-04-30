# Round 2 Refinement

## Problem Anchor (verbatim)

- **Bottom-line problem**: Multi-turn ReAct GRPO on Text-to-SQL succeeds modestly at 7B (+2 pp Spider-dev EX_att/EX_all) but fails at 3B; SKE-RL fails to help and worsens with more rollouts. We do not know what the +2 pp actually changed inside the agent, and we cannot explain why a natural structural prior fails.
- **Must-solve bottleneck**: Diagnostic gap. Literature reports aggregate EX gains without decomposing protocol-compliance vs SQL-reasoning contributions.
- **Non-goals**: SOTA; new method that wins; scaling-law claim; toolkit-as-lead.
- **Constraints**: 1× Quadro RTX 6000 24 GB, ~3 weeks GPU + 1 week writing.
- **Success condition**: Reviewer says paper changes how they read agentic-SQL-RL leaderboards AND SKE negative result has a single, measured, convincing mechanism.

## Anchor Check
- Original bottleneck still preserved.
- All R2 fixes (leakage-safe extractor, direct-correctness rules, E3 join, CIs) sharpen the diagnostic — no drift risk.
- Drop-3B / drop-external-probe / drop-M1-M4-from-body: all narrow scope toward anchor, none jeopardize the success condition.

## Simplicity Check
- Dominant contribution unchanged: 1 deterministic diagnostic + 1 attribution ratio + 1 bucket distribution.
- Removed from main proposal body: 3B Dense same-launcher run, external-baseline probe (now appendix-only-if-available footnote), M1-M4 mechanism grid (gone from body entirely).
- Remaining mechanism is still smallest adequate: classifier + lenient pass + 1 confound run for SKE.

## Changes Made

### 1. Leakage-safe lenient extractor (CRITICAL)
- **Reviewer said**: multi-candidate "best executes" → pseudo-oracle eval.
- **Action**: lock `lenient_pred_sql` to a single deterministic rule, **no candidate selection by execution match against gold**:
  ```
  lenient_pred_sql = (
      first sqlglot-parseable SQL in the final-Answer-region of the trajectory
      else last sqlglot-parseable SQL action over the entire trajectory
      else None
  )
  ```
  Re-execution against `gold_sql` happens AFTER selection, not as a tiebreaker.
- **Impact**: Lenient eval is now a deterministic recovery procedure, not an oracle.

### 2. Direct-correctness bucket rules (CRITICAL)
- **Reviewer said**: `RL.lenient_correct == SFT.lenient_correct` is awkward.
- **Action**: rewrite buckets using direct strict/lenient correctness:
  ```python
  protocol_gain = (not SFT.final_correct_strict
                   and SFT.final_correct_lenient
                   and RL.final_correct_strict)
  semantic_gain = (not SFT.final_correct_lenient
                   and RL.final_correct_lenient)
  no_change = otherwise
  # precedence: protocol_gain > semantic_gain > no_change
  ```
  Translation: `protocol_gain` = "the SFT model already had the correct SQL inside its trajectory but failed protocol; RL fixed protocol". `semantic_gain` = "the SFT model never produced the correct SQL anywhere; RL produced it under tolerant extraction".
- **Impact**: Rules are now exclusive, ordered, and trivially auditable by reading the per-record table.

### 3. E3 join key (IMPORTANT)
- **Reviewer said**: "class-baseline shifts uncorrelated with semantic_gain records" needs explicit join.
- **Action**: define the join hierarchy explicitly:
  - **Tier 1 (preferred)**: SKE training rollouts and Spider-dev eval are not the same examples (train/dev split), so exact-id join is impossible. **Fallback to Tier 2.**
  - **Tier 2 (used)**: join by `(skeleton_class, db_id)`. Each Spider-dev record gets `dev_skeleton_class = extract_skeleton_strict(gold_sql)`. SKE training logs already record `(prompt_id, skeleton_class, advantage_per_class, n_real_classes)`. We aggregate training-time class-baseline shift magnitude per `skeleton_class`, then join to dev records by `skeleton_class`. The E3 statement becomes: *"in dev records where (Dense vs SFT) classifies as `semantic_gain`, the average training-time class-baseline shift for that record's `skeleton_class` is statistically indistinguishable from records where (Dense vs SFT) classifies as `protocol_gain`"*.
  - **Tier 3 (degraded fallback)**: if Tier 2 join is too noisy (e.g., many dev `skeleton_class` values absent from training), report E3 as an aggregate-only statement: *"SKE training-time per-class advantage shift distribution does not differ between high-`semantic_gain`-density and high-`protocol_gain`-density skeleton classes"*.
- **Impact**: E3 is now a measurable correlation, not hand-wave.

### 4. Bootstrap confidence intervals (IMPORTANT)
- **Reviewer said**: small bucket counts → brittle shares without CIs.
- **Action**: report all four headline numbers with 95% bootstrap CIs (1000 resamples over Spider-dev records):
  - `attribution_ratio`
  - `bucket_share[protocol_gain]`
  - `bucket_share[semantic_gain]`
  - `lenient_attribution_gap = ΔEX_strict − ΔEX_lenient`
- **Impact**: A single line in every figure caption — required uncertainty accounting.

### 5. Drop 3B Dense from main (SIMPLIFICATION 1)
- **Action**: removed from §Compute & Timeline. Anchor mentions 3B cliff, but Claim 1 + Claim 2 do not need it. If time remains after mandatory work, may run as appendix-only "additional context", not as a main-text claim.

### 6. External-baseline probe → appendix-only footnote (SIMPLIFICATION 2)
- **Action**: removed §6.4 from main; now a single appendix paragraph: *"We searched for per-record trajectory artifacts from HES-SQL / MTIR-SQL / MARS-SQL within 1 day. If found, we apply the diagnostic to one such artifact and report in Appendix B. If not found, the diagnostic is internally validated only."*

### 7. M1-M4 grid: removed from body (SIMPLIFICATION 3)
- **Action**: M1-M4 grid is gone from the proposal body. Appendix C is a 1-page checklist titled *"A diagnostic checklist for negative results in agentic RL"* listing the four mechanism templates with one-line operational definitions. Body never mentions them.

## Revised Proposal

### Problem Anchor
(verbatim above; not repeated)

### Technical Gap

Recent agentic Text-to-SQL RL work (HES-SQL, MTIR-SQL, MARS-SQL, Graph-Reward-SQL) reports +N pp EX_all gains and treats those gains as evidence of improved SQL reasoning. **No paper in this line decomposes the gain into protocol-validity vs SQL-semantics.** EX_all conflates the two. Consequences: structural-reward methods cannot distinguish their stated mechanism from coincidental protocol cleanup; negative results like SKE-RL are uninterpretable; naive scaling reports the same conflated metric. The smallest adequate intervention is a measurement protocol, not a new training method.

### Method Thesis
- **One-sentence thesis**: Agentic GRPO on multi-turn Text-to-SQL improves aggregate accuracy primarily by aligning tool-use protocols, not by improving SQL semantic reasoning — which explains why a structural-advantage prior (SKE-RL) cannot help: it operates on the layer below the actual bottleneck.
- **Smallest adequate intervention**: Deterministic 3-bucket classifier + leakage-safe lenient re-execution + single attribution ratio with bootstrap CIs. Zero new training for dominant contribution.
- **Why timely**: Foundation-model-era agentic RL all reports aggregate task-success gains. Protocol-vs-semantic conflation is portable beyond SQL.

### Contribution Focus
- **Dominant**: Deterministic, portable diagnostic decomposing any agentic Text-to-SQL RL gain into protocol-validity vs SQL-semantic components, with a single headline `attribution_ratio` (95% bootstrap CI). Applied to our 7B Dense GRPO, it shows the +2 pp gain is dominantly protocol.
- **Optional supporting**: SKE-RL fails because bottleneck is protocol-layer; same diagnostic confirms via E1+E2+E3 (with explicit join key for E3).
- **Explicit non-contributions**: M1-M4 mechanism grid (appendix checklist only); new RL method; scaling law; SOTA; toolkit-as-lead.

### Proposed Method

#### Complexity Budget
- **Frozen / reused**: 4 existing 7B checkpoints; existing eval JSONs with full trajectories; sqlglot AST; SQLite execution env; existing SKE training-time per-class logs.
- **New trainable**: 0 for dominant. 1 supporting (Dense G=8 confound, ~1.2 run-eq).
- **Tempting additions intentionally not used**: Multi-seed; BIRD GRPO; 3B Dense rerun (moved to potential appendix only); 1.5B/14B sizes; v2 SKE; mechanism grid as main claim.

#### System Overview

```
Existing eval JSON  ──►  Step 1: Strict + LEAKAGE-SAFE Lenient extraction
   (any agentic        │      strict_pred = current extract_final_sql()
    SQL RL run)        │      lenient_pred = first sqlglot-parseable SQL in
                       │                     final-Answer-region of trajectory;
                       │                     else last sqlglot-parseable SQL action;
                       │                     else None
                       │                     (selection NEVER consults gold_sql)
                       │      Re-execute both against db_path AFTER selection
                       ▼
                       Step 2: Per-record classifier (deterministic, mutually exclusive)
                       protocol_gain = (not SFT.final_correct_strict
                                        and SFT.final_correct_lenient
                                        and RL.final_correct_strict)
                       semantic_gain = (not SFT.final_correct_lenient
                                        and RL.final_correct_lenient)
                       no_change     = otherwise
                       precedence: protocol_gain > semantic_gain > no_change
                       ▼
                       Step 3: Aggregate diagnostic table (with 95% bootstrap CIs)
                       For (RL vs SFT) pair:
                         attribution_ratio = (ΔEX_strict − ΔEX_lenient) / max(ΔEX_strict, ε)
                         bucket_share[protocol_gain], bucket_share[semantic_gain]
                         lenient_attribution_gap = ΔEX_strict − ΔEX_lenient
                       ▼
                       Step 4: SKE confirmation (supporting)
                       Apply Step 2-3 to (SKE-G4 vs Dense), (SKE-G8 vs Dense)
                       Test E1 + E2 + E3
                       Confound: Dense G=8 row (1 confound check, not centerpiece)
```

#### Core Mechanism

**Classifier interface** (deterministic, leakage-safe):

```python
def lenient_extract(trajectory: str) -> str | None:
    """No gold_sql access. Pure trajectory → SQL."""
    ans = re.search(r"Answer:\s*(.*?)(?:\nObservation:|\Z)", trajectory, re.DOTALL)
    if ans:
        for cand in _sql_candidates_in_text(ans.group(1)):
            if sqlglot.parse_one(cand, dialect="sqlite") is not None:
                return cand
    for cand in reversed(_sql_action_strings(trajectory)):
        if sqlglot.parse_one(cand, dialect="sqlite") is not None:
            return cand
    return None


def classify_pair(sft_record: dict, rl_record: dict) -> tuple[Bucket, Flags]:
    """
    Per-pair inputs:
      strict_pred_sql, lenient_pred_sql, gold_sql, all_sql_actions,
      execution_errors, final_answer_span, db_id

    Per-pair flags computed for both SFT and RL:
      strict_extract_ok      = strict_pred_sql is not None and sqlglot-parses
      lenient_extract_ok     = lenient_pred_sql is not None  (already parses by construction)
      first_sql_exec_ok      = execution_errors[0] == "" if all_sql_actions else False
      final_correct_strict   = exec_match(strict_pred_sql, gold_sql)  if strict_extract_ok else False
      final_correct_lenient  = exec_match(lenient_pred_sql, gold_sql) if lenient_extract_ok else False
      observation_repair_flag= (not first_sql_exec_ok) and final_correct_strict
      equiv_class_change_flag= (canonical_ast(SFT.strict_pred) != canonical_ast(RL.strict_pred))

    Bucket rules (precedence: protocol_gain > semantic_gain > no_change):
      protocol_gain = (not SFT.final_correct_strict
                       and SFT.final_correct_lenient
                       and RL.final_correct_strict)
      semantic_gain = (not SFT.final_correct_lenient
                       and RL.final_correct_lenient)
      no_change     = otherwise
    """
```

**Headline numbers (per RL-vs-SFT pair, all with 95% bootstrap CIs over 1000 resamples)**:

- `attribution_ratio = (ΔEX_strict − ΔEX_lenient) / max(ΔEX_strict, ε)` ∈ [0, 1]. 1.0 = entire gain disappears under lenient eval (protocol-only). 0.0 = lenient and strict gains identical (semantic).
- `bucket_share[protocol_gain]`, `bucket_share[semantic_gain]` over all dev records.
- `lenient_attribution_gap = ΔEX_strict − ΔEX_lenient` (in absolute pp).
- `observation_repair_share`: of `protocol_gain` records, fraction with `observation_repair_flag = True` (subanalysis).

**Why main novelty**: No published agentic Text-to-SQL RL paper reports any of these. The classifier is a portable interpretive primitive — generalizes by swapping `gold_sql` for `task_success_predicate` and `execute_sql` actions for arbitrary `tool_call` actions.

#### Optional Supporting Component (= SKE single mechanism, with E3 join key)

Apply the same classifier to (SKE-G4 vs Dense) and (SKE-G8 vs Dense). Test the single prediction:

> SKE-RL fails because the bottleneck is at the protocol layer it cannot affect.

Required evidence (all three must hold):
- **E1**: `bucket_share[semantic_gain]` for (SKE vs Dense) ≤ that for (Dense vs SFT). I.e., SKE produces no more semantic improvements than Dense does.
- **E2**: `lenient_EX(SKE) ≤ lenient_EX(Dense)` (within bootstrap CI). I.e., even granting SKE the protocol-noise tolerance, it does not exceed Dense.
- **E3**: SKE training-time per-class advantage shift distribution does not differ between dev `skeleton_class`-buckets that are high-density-`semantic_gain` vs high-density-`protocol_gain` (Mann-Whitney U test, p > 0.05). **Join key**: dev record `skeleton_class = extract_skeleton_strict(gold_sql)` joined to SKE training-time per-`skeleton_class` advantage-shift aggregates. **Degraded fallback** if join coverage <50% of dev records: aggregate-only statement that SKE class-baseline shifts are uniformly distributed across dev outcome buckets.

**Confound check (Dense G=8)**: One Dense G=8 run (~1.2 run-eq) gives the 2×2 grid {Dense, SKE} × {G=4, G=8}. If Dense G=8 EX_att ≥ Dense G=4 EX_att within 1 pp, SKE-G=8 collapse is structural to class-baseline math, not generic. **Single row in the SKE table.**

#### Modern Primitive Usage
- Lenient re-execution = real SQLite as inference-time verifier — recovers what the agent meant despite protocol noise. Selection is leakage-safe.
- AST-canonical equivalence (sqlglot) for `equiv_class_change_flag` and the E3 join key.
- No new trained component for dominant contribution.

#### Integration
Diagnostic attaches AFTER any agentic-RL pipeline. Reads eval JSONs only. Reference impl is one Python file consuming the listed input fields.

#### Training Plan
Dominant: NONE. Supporting: Dense G=8 confound (~1.2 run-eq).

#### Failure Modes

| Failure | Detect | Fallback |
|---|---|---|
| `attribution_ratio` ≈ 0 (gain fully semantic) | Lenient ΔEX ≈ Strict ΔEX, both with overlapping CIs | Story flips: gain IS semantic, SKE still fails (concentrated in queries SKE cannot reach). Diagnostic still novel; conclusion inverts. |
| Dense G=8 also degrades | EX_att(Dense G=8) < EX_att(Dense G=4) by >1 pp outside CI | Generic G effect; reframe as rollout-budget tradeoff. SKE story weakened. |
| Both gain buckets <5% of records | Most transitions in `no_change` (CIs include 0) | Effect-size CI honest reporting; possible "Dense gain is mostly noise" finding. |
| Lenient extractor pathological | lenient EX(SFT) < strict EX(SFT) on 50-record sanity sample | Tighten extractor to "Answer-region only" (drop last-action fallback); re-validate. |
| E3 join coverage <50% | <50% of dev `skeleton_class` values appear in training-time SKE logs | Use Tier-3 aggregate-only fallback for E3; note as limitation. |

#### Novelty / Elegance

Closest published: HES-SQL, Graph-Reward-SQL, MTIR-SQL, MARS-SQL, LearNAT — all sit inside EX_all conflation. Our differentiation: we do NOT propose a winning intervention. We propose a diagnostic that turns existing aggregate gains into mechanistic claims, with SKE-RL as a designed-to-fail control that makes the diagnostic visible in both directions (positive on Dense, null on SKE).

Focused: One classifier + one ratio + one bucket distribution + one supporting failure analysis. No new training for dominant contribution. Whole paper rotates around one question.

Elegant: One leakage-safe re-execution pass + one deterministic classifier + one ratio with CI. Anyone with eval JSONs can apply tomorrow.

### Claim-Driven Validation Sketch

#### Claim 1 (dominant): Dense agentic GRPO gain on Text-to-SQL is dominantly protocol/tool-use alignment

- **Min experiment**: Lenient-extraction re-execution of SFT vs Dense75 on full Spider-dev (1034). Compute `attribution_ratio` and `bucket_share` with 95% bootstrap CIs.
- **Cost**: ~6 GPU-h (SQLite re-execution); 0 for classifier.
- **Baseline / sanity**: lenient_EX(SFT) ≥ strict_EX(SFT) on a 50-record pre-pipeline sample (extractor sanity).
- **Metric**: `attribution_ratio` ∈ [0,1] with CI; `bucket_share` over all dev records with CIs.
- **Expected**: `attribution_ratio` ≥ 0.5 with CI excluding 0; `bucket_share[protocol_gain] > bucket_share[semantic_gain]` with non-overlapping CIs.

#### Claim 2 (supporting): SKE-RL fails because the bottleneck is at the protocol layer it cannot affect

- **Min experiment**: Apply diagnostic to (SKE-G4 vs Dense) and (SKE-G8 vs Dense). Test E1+E2+E3 with bootstrap CIs and Mann-Whitney U for E3. Plus Dense G=8 confound (~1.2 run-eq).
- **Baseline / ablation**: 2×2 grid {Dense, SKE} × {G=4, G=8}.
- **Metric**: `bucket_share[semantic_gain]_(SKE vs Dense)` vs `_(Dense vs SFT)` with CIs; `lenient_EX(SKE)` vs `lenient_EX(Dense)`; E3 Mann-Whitney p-value over join-keyed buckets.
- **Expected**: E1, E2, E3 all hold; Dense G=8 ≥ Dense G=4 within 1 pp.

### Experiment Handoff
- **Must-prove**: C1 (protocol attribution); C2 (SKE single mechanism via E1+E2+E3)
- **Must-run**: Dense G=8 confound; lenient sanity on SFT (50 records); bootstrap CIs on all headline numbers
- **Critical metrics**: `attribution_ratio`, `bucket_share`, `lenient_EX`, E3 Mann-Whitney p
- **Highest-risk assumptions**:
  - (R1) Lenient extractor recovers SQL on ≥80% of format-failed SFT trajectories (50-record sanity)
  - (R2) `semantic_gain` ≥ 10 transitions in (Dense vs SFT) — else C2 has no contrast
  - (R3) Dense G=8 ≥ Dense G=4 within 1 pp — else G effect confounds SKE story
  - (R4) E3 join coverage ≥ 50% — else fallback to aggregate-only E3

### Compute & Timeline

| Block | GPU-h | Wall | Note |
|---|---:|---|---|
| Lenient re-execution: SFT + Dense75 | ~6 | <1d | Existing trajectory logs |
| Classifier + transition table | ~0 | <1d | Pure CPU |
| Bootstrap CIs (1000 resamples × 4 numbers) | ~0 | <0.5d | Pure CPU |
| SKE diagnostic (apply to G4+G8 vs Dense, E1+E2+E3) | ~6 | ~1d | Existing eval JSONs + lenient pass + join with training logs |
| Dense G=8 confound run | ~28 (1.2 run-eq + eval) | ~1.5d | Single confound check |
| Writing + figures | n/a | ~1 week | After diagnostics |
| (Appendix-only-if-found) External probe | ~2 if avail | <1d | Skip if no public artifacts within 1d search |

**Mandatory**: ~40 GPU-h (1.7 run-eq) + 3 days CPU + 1 week writing. Easily fits 3-week budget.
