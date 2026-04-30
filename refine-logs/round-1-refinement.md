# Round 1 Refinement

## Problem Anchor (verbatim)

- **Bottom-line problem**: Multi-turn ReAct GRPO on Text-to-SQL succeeds modestly at 7B (+2 pp Spider-dev EX_att/EX_all) but fails at 3B; a structural-advantage prior (SKE-RL) fails to help and worsens with more rollouts. We do not know what the +2 pp actually changed inside the agent, and we cannot explain why a natural structural prior fails.
- **Must-solve bottleneck**: Diagnostic gap. The literature reports aggregate EX gains without decomposing protocol-compliance vs SQL-reasoning contributions, so neither positive (Dense GRPO) nor negative (SKE) results can be interpreted mechanistically.
- **Non-goals**: SOTA on Spider/BIRD; new method that fixes everything; scaling-law claim across model sizes; a toolkit-as-lead paper.
- **Constraints**: 1× Quadro RTX 6000 24 GB, ~3 weeks GPU + 1 week writing.
- **Success condition**: A reviewer would say the paper changes how they read agentic-SQL-RL leaderboards AND the SKE negative result has a single, convincing, measured mechanism.

## Anchor Check

- **Original bottleneck**: diagnostic gap — current literature cannot decompose agentic-RL gains into protocol-compliance vs SQL-reasoning contributions.
- **Why the revised method still addresses it**: The revision *narrows* the diagnostic to exactly one binary split (protocol vs semantic) with a deterministic classifier, then uses SKE strictly as one falsifying control of "is the bottleneck protocol or semantic?". This sharpens the original anchor; it does not drift.
- **Reviewer suggestions rejected as drift**: NONE. All three reviewer fixes (deterministic classifier, single mechanism, simplification) move toward the anchor, not away.

## Simplicity Check

- **Dominant contribution after revision**: A deterministic, portable diagnostic that decomposes any agentic Text-to-SQL RL gain into a **protocol-validity component** and a **SQL-semantic component**, plus the empirical finding that for our 7B Dense GRPO the gain is dominantly protocol.
- **Components removed or merged**:
  - DELETED from main paper: M1-M4 mechanism grid for SKE → moved to appendix as a diagnostic checklist that future researchers can apply.
  - MERGED: `format_fix` bucket + lenient-extraction attribution → single **protocol-validity** section with one headline number (the lenient-attribution ratio).
  - DEMOTED: `observation_repair` from top-level bucket to a per-record flag inside the protocol-validity section.
  - SIMPLIFIED: bucket taxonomy from 4 → 2 main buckets (`protocol_gain`, `semantic_gain`) with `observation_repair`, `protocol_only`, `unchanged` as flags inside the joint table.
- **Reviewer suggestions rejected as unnecessary complexity**: The "external sanity demonstration on a published baseline" is **conditionally accepted**: only if a public log of a recent agentic SQL RL paper (HES-SQL, MTIR-SQL, MARS-SQL) is available with per-record trajectories. We will not reproduce a baseline ourselves — that would be a parallel contribution. Will check public artifact availability and report in §Failure Modes.
- **Why the remaining mechanism is still smallest adequate**: The diagnostic is now exactly one re-execution pass + one classifier with deterministic precedence + one headline number. There is nothing smaller that still answers "what did RL change inside the agent".

## Changes Made

### 1. Deterministic classifier interface (CRITICAL fix)
- **Reviewer said**: bucket boundaries leave too much script discretion.
- **Action**: Spec'd the classifier as a pure function `classify(record) → (bucket, flags)` with explicit inputs, explicit flag set, explicit precedence. Each flag has a one-line operational definition. Each bucket is a logical predicate over flags only.
- **Reasoning**: A deterministic interface is required for both reproducibility and for the dominant contribution to be a *portable* primitive. Without it, the diagnostic becomes "our scripted opinions about trajectory categories", which is exactly the kind of forensics the reviewer warned against.
- **Impact on core method**: Method specificity goes from "conceptually clear" to "engineer-can-implement-this-tonight".

### 2. SKE collapsed to single falsifiable mechanism (IMPORTANT fix)
- **Reviewer said**: M1-M4 grid risks becoming a second paper / post-hoc forensics.
- **Action**: SKE failure analysis collapsed to **one** falsifiable claim: *"SKE fails because the bottleneck is protocol, not SQL structure."* The required evidence is the same diagnostic applied to SKE-vs-Dense — three predictions (no `semantic_gain` lift, no lenient-EX lift, only class-baseline shifts that don't move protocol errors). M1-M4 framework moved to appendix as a checklist for future agentic-RL negative-result analyses.
- **Reasoning**: If the dominant thesis is "the bottleneck is protocol", then SKE's failure follows trivially — it cannot help because it operates on the layer below the bottleneck. Confirming this strengthens the dominant thesis and dispenses with the parallel mechanism-grid claim.
- **Impact on core method**: Validation Focus from 6 → expected 8+. Paper rotates around exactly one new measurement.

### 3. Bucket simplification (IMPORTANT fix)
- **Reviewer said**: Merge `format_fix` into protocol-validity; demote `observation_repair` to flag.
- **Action**: Final bucket set is now {`protocol_gain`, `semantic_gain`, `no_change`} with a per-record flag set including `observation_repair_flag`, `format_failure_flag`, `equivalence_class_change_flag`. Per-record table reports flags; aggregate table reports buckets.
- **Reasoning**: The 2-bucket split mirrors the lenient-vs-strict attribution number, so the empirical decomposition is consistent across all reported figures.
- **Impact on core method**: Reads as one decomposition, not three competing taxonomies.

### 4. External-baseline sanity (conditional accept)
- **Reviewer said**: One small external sanity demo would help portability.
- **Action**: Added a §6.4 **External Probe** subsection — *if* HES-SQL, MTIR-SQL, MARS-SQL, or another recent agentic SQL RL paper publishes per-record trajectory logs with comparable schema, we apply the diagnostic to one (1) such artifact and report. *If not* available within 1 day of search, we drop and note "external probe blocked by missing public artifacts" as a limitation. No reproduction, no re-training, no leaderboard comparison.
- **Reasoning**: The reviewer explicitly said "do not add a benchmark suite". This is exactly that — a 0-GPU sanity check on whatever exists, gated by 1-day availability check.
- **Impact on core method**: Venue Readiness 6 → expected 7-8 if probe lands; unchanged if artifacts unavailable.

## Revised Proposal

### Problem Anchor

(verbatim from above; not repeated)

### Technical Gap

Recent agentic Text-to-SQL RL work (HES-SQL, MTIR-SQL, MARS-SQL, Graph-Reward-SQL) reports +N pp EX_all gains and treats those gains as evidence of improved SQL reasoning. **No paper in this line decomposes the gain into protocol-validity vs SQL-semantics.** The two are systematically conflated under EX_all. As a consequence:

- (a) Existing structural-reward methods (HES-SQL) cannot say whether their gain comes from structural alignment or coincidental protocol cleanup.
- (b) Negative results like SKE-RL are uninterpretable beyond "it didn't work".
- (c) Naive scaling — more rollouts (G=8), more reward dimensions, multi-agent decomposition — keeps reporting the same conflated metric.

The smallest adequate intervention is a measurement protocol, not a new training method.

### Method Thesis

- **One-sentence thesis**: Agentic GRPO on multi-turn Text-to-SQL improves aggregate accuracy primarily by aligning tool-use protocols (valid action format, valid SQL, valid Answer extraction), not by improving SQL semantic reasoning — which explains why a structural-advantage prior (SKE-RL) cannot help: it operates on the layer below the actual bottleneck.
- **Smallest adequate intervention**: A deterministic 2-bucket trajectory classifier + lenient-vs-strict re-execution + a single attribution ratio. No new training for the dominant contribution.
- **Why timely**: Foundation-model-era agentic RL all reports aggregate task-success gains. The protocol-vs-semantic conflation we identify is portable to other tool-using agentic settings.

### Contribution Focus

- **Dominant**: A deterministic, portable diagnostic that decomposes any agentic Text-to-SQL RL gain into protocol-validity vs SQL-semantic components, with a single headline attribution ratio. Applied to our 7B Dense GRPO, it shows the +2 pp gain is dominantly protocol.
- **Optional supporting**: SKE-RL fails because the bottleneck is protocol-layer; same diagnostic confirms — SKE shows no `semantic_gain` lift, no lenient-EX lift, only class-baseline shifts that do not move protocol errors. G=8 degradation is the predicted consequence.
- **Explicit non-contributions**: Mechanism grid for SKE (appendix only); new RL method; scaling law; SOTA; toolkit-as-lead.

### Proposed Method

#### Complexity Budget
- **Frozen / reused**: 4 existing 7B checkpoints (SFT, Dense75, SKE-G4-75, SKE-G8-75); existing eval JSONs with full trajectories; sqlglot AST extractor; SQLite execution env.
- **New trainable components**: 0 for dominant contribution; 1 supporting (Dense G=8 confound check, ~1.2 run-eq) — kept as a confound check, not centerpiece.
- **Tempting additions intentionally not used**: Multi-seed, BIRD GRPO, 1.5B/14B sizes, v2 SKE, mechanism grid as main claim.

#### System Overview

```
Existing eval JSON  ──►  Step 1: Strict + Lenient extraction
   (any agentic        │      strict_pred = current extract_final_sql()
    SQL RL run)        │      lenient_pred = relaxed regex over Answer-region
                       │                    + last-action-fallback if no Answer
                       │                    must parse with sqlglot
                       │      Re-execute both against db_path
                       │      Output per-record: strict_extract_ok,
                       │                          lenient_extract_ok,
                       │                          strict_correct, lenient_correct
                       │
                       ▼
                       Step 2: Per-record classifier (DETERMINISTIC)
                       Inputs: see below
                       Outputs: bucket ∈ {protocol_gain, semantic_gain, no_change}
                                + flags (boolean set)
                       │
                       ▼
                       Step 3: Aggregate diagnostic table
                       For (RL vs SFT) pair:
                         attribution = (ΔEX_strict − ΔEX_lenient) / max(ΔEX_strict, ε)
                         bucket distribution over {SFT-wrong → RL-right} transitions
                       │
                       ▼
                       Step 4: SKE confirmation (supporting)
                       Apply Step 2-3 to (SKE-G4 vs Dense), (SKE-G8 vs Dense)
                       Predict: no semantic_gain; no lenient-EX lift; only adv-stats shift
                       Confound: Dense-G8 control to rule out generic G effect
```

#### Core Mechanism (= deterministic diagnostic)

**Classifier interface**:
```python
def classify(record) -> tuple[Bucket, Flags]:
    """
    Inputs (per-record dict):
      strict_pred_sql: str | None    # output of current extract_final_sql()
      lenient_pred_sql: str | None   # output of relaxed extractor (Answer-region
                                     #   regex, then last-action fallback,
                                     #   must sqlglot-parse)
      gold_sql: str
      all_sql_actions: list[str]     # all execute_sql[...] action strings in trajectory
      execution_errors: list[str]    # error messages for each action, "" if success
      final_answer_span: str | None  # text after last "Answer:" tag
      db_id: str                     # for re-execution

    Computed flags (booleans, per-record):
      strict_extract_ok    = strict_pred_sql is not None and parses
      lenient_extract_ok   = lenient_pred_sql is not None and parses
      first_sql_exec_ok    = execution_errors[0] == "" if all_sql_actions else False
      final_sql_exec_ok    = re-execute(strict_pred_sql or lenient_pred_sql) succeeds
      final_correct_strict = exec_match(strict_pred_sql, gold_sql)
      final_correct_lenient= exec_match(lenient_pred_sql, gold_sql)
      sql_ast_changed      = canonical_ast(strict_pred_sql) != canonical_ast(first_sql_action)
      table_set_changed    = table_set(strict_pred_sql) != table_set(first_sql_action)
      condition_set_changed= where_clause_set(strict_pred_sql) != where_clause_set(first_sql_action)
      observation_repair_flag = (not first_sql_exec_ok) and final_sql_exec_ok and final_correct_strict
      equivalence_class_change_flag = (canonical_ast(SFT_strict) != canonical_ast(RL_strict))

    Bucket precedence (for {SFT vs RL} pair classification):
      protocol_gain = (not SFT.strict_extract_ok or not SFT.final_sql_exec_ok)
                       AND RL.final_correct_strict
                       AND (RL.lenient_correct_lenient == SFT.lenient_correct_lenient)
                       # i.e., RL fixed protocol; lenient parity says no semantic change
      semantic_gain = SFT.lenient_correct_lenient == False
                       AND RL.lenient_correct_lenient == True
                       # lenient-eval also moved → not just protocol cleanup
      no_change     = otherwise (covers: both right, both wrong, equivalence-class-only edits)
    """
```

**Headline numbers (per RL-vs-SFT pair)**:
- `attribution_ratio = (ΔEX_strict − ΔEX_lenient) / max(ΔEX_strict, ε)` ∈ [0, 1].
  - 1.0 = entire gain disappears under lenient eval → all protocol.
  - 0.0 = lenient and strict gains identical → all semantic.
- `bucket_share`: % of {SFT-wrong → RL-right} transitions that fall in `protocol_gain` vs `semantic_gain`.
- `observation_repair_share`: of `protocol_gain` transitions, what fraction had `observation_repair_flag = True` (subanalysis).

**Why this is the main novelty**: No published agentic Text-to-SQL RL paper reports either `attribution_ratio` or `bucket_share`. Both are interpretive primitives; they generalize trivially (browser agent: replace `gold_sql` with `task_success_predicate`, replace `execute_sql` actions with `tool_call` actions).

#### Optional Supporting Component (= SKE single falsifiable mechanism)

Apply the same classifier to (SKE-G4 vs Dense) and (SKE-G8 vs Dense). Test the single prediction:

> SKE-RL fails because the bottleneck is at the protocol layer it cannot affect.

Required evidence (all three must hold for the prediction to be confirmed):
- **E1**: `bucket_share[semantic_gain]` for (SKE vs Dense) is ≤ that for (Dense vs SFT). I.e., SKE does not produce more semantic improvements than Dense.
- **E2**: `lenient_EX(SKE) ≤ lenient_EX(Dense)`. I.e., even granting SKE the protocol-noise tolerance, it does not exceed Dense.
- **E3**: SKE's per-rollout class statistics (already logged via R3 patches: `n_real_classes`, `class_size_distribution`, advantage variance per class) show that the class baseline IS doing what it's supposed to do (i.e., shifting advantage estimates) but those shifts are *uncorrelated* with `bucket = semantic_gain` records.

**Confound check (Dense G=8)**: Run one Dense G=8 (~1.2 run-eq) so we have the 2×2 grid {Dense, SKE} × {G=4, G=8}. If Dense G=8 ≥ Dense G=4, then SKE-G=8's degradation is structural to the class-baseline math, not generic to G. Reported as a single confound row, not a centerpiece.

**M1-M4 mechanism grid → appendix only.** Future researchers facing similar negative results can apply the checklist; for THIS paper, the single mechanism above is the supporting claim.

#### Modern Primitive Usage
- Lenient re-execution leverages real SQLite as an inference-time verifier — recovers what the agent meant despite protocol noise.
- AST-canonical equivalence (sqlglot) verifies that the `equivalence_class_change_flag` is structural, not literal-only.
- No new trained component for dominant contribution.

#### Integration
Diagnostic attaches AFTER an arbitrary agentic-RL pipeline. Reads eval JSONs only. Reference implementation works on any JSON with the listed input fields.

#### Training Plan
Dominant contribution: NONE. Supporting: one Dense G=8 confound run (~1.2 run-eq).

#### Failure Modes and Diagnostics

| Failure mode | Detect | Fallback |
|---|---|---|
| `attribution_ratio` ≈ 0 (gain fully semantic) | `lenient_EX(Dense) − lenient_EX(SFT)` ≈ `strict` delta | Story flips: "Dense gain IS semantic, SKE still fails because semantic improvements are concentrated in queries SKE cannot reach". Diagnostic still novel — same protocol applies, conclusion inverts. |
| Dense G=8 also degrades vs Dense G=4 | EX_att(Dense G=8) < EX_att(Dense G=4) by >1 pp | The G effect is generic, SKE not specially blamed. Frame becomes "rollout-budget vs quality trade-off in agentic GRPO" — still publishable, requires reframe of E3 evidence. |
| Both `protocol_gain` and `semantic_gain` cells are <5% of transitions | Most transitions land in `no_change` | Δ effect is small/noisy; report effect-size CI honestly. Either accept as a finding ("Dense gain is mostly noise") or reduce confidence and flag as workshop-only. |
| Lenient extractor pathological (lenient < strict on SFT) | Sanity check on SFT alone: lenient EX must be ≥ strict EX | Tighten extractor: "any final-Answer-region SQL that sqlglot-parses; else last action that sqlglot-parses; else None". Re-validate on 50 SFT records before pipeline runs. |
| External probe artifact unavailable | 1-day search of HES-SQL/MTIR-SQL/MARS-SQL public logs returns no per-record trajectories | Drop §6.4 External Probe; note as limitation. Diagnostic remains internally validated. |

#### Novelty and Elegance Argument

Closest published work: HES-SQL (skeleton REWARD), Graph-Reward-SQL (graph reward), MTIR-SQL (multi-turn GRPO), MARS-SQL (multi-agent ReAct), LearNAT (AST + DPO). All sit inside the EX_all conflation. **Our differentiation**: we propose a diagnostic that turns those existing aggregate gains into mechanistic claims, and use SKE-RL as a designed-to-fail control that makes the diagnostic visible in both directions (positive on Dense, null on SKE).

Focused: One classifier. One attribution ratio. One bucket distribution. One supporting failure analysis (predicted by the main thesis). No new training for dominant contribution. Whole paper rotates around one question — "what does agentic SQL RL actually change?" — with exactly one new way to answer it.

Elegant: The diagnostic is one re-execution pass + one deterministic classifier + one ratio. Anyone with eval JSONs from any agentic Text-to-SQL RL paper can apply it tomorrow.

### Claim-Driven Validation Sketch

#### Claim 1 (dominant): Dense agentic GRPO gain on Text-to-SQL is dominantly protocol/tool-use alignment

- **Min experiment**: Lenient-extraction re-execution of SFT vs Dense75 on full Spider-dev (1034). Compute `attribution_ratio` and bucket distribution over SFT-wrong→Dense-right transitions.
- **Cost**: ~6 GPU-h (SQLite-bound re-execution); 0 for classifier.
- **Baselines / ablations**: SFT vs Dense75; sanity: lenient-EX(SFT) ≥ strict-EX(SFT).
- **Metric**: `attribution_ratio` ∈ [0,1]; bucket share over transitions.
- **Expected directional outcome**: `attribution_ratio` ≥ 0.5 (≥ half the gain is protocol); `bucket_share[protocol_gain] > bucket_share[semantic_gain]`.

#### Claim 2 (supporting): SKE-RL fails because the bottleneck is at the protocol layer it cannot affect

- **Min experiment**: Apply the same diagnostic to (SKE-G4 vs Dense) and (SKE-G8 vs Dense). Test E1 + E2 + E3. Plus Dense G=8 confound run (~1.2 run-eq).
- **Baselines / ablations**: 2×2 grid {Dense, SKE} × {G=4, G=8}.
- **Metric**: bucket_share[semantic_gain]_(SKE vs Dense) vs (Dense vs SFT); lenient_EX(SKE) vs lenient_EX(Dense); correlation between SKE class-baseline shifts and `bucket=semantic_gain` records.
- **Expected directional outcome**: E1, E2, E3 all hold. SKE does not move semantic gains; SKE class-baseline math operates correctly but on the wrong layer.

### Experiment Handoff Inputs
- **Must-prove**: C1 (protocol attribution); C2 (SKE single mechanism via E1+E2+E3).
- **Must-run ablations**: Dense G=8 confound; lenient-extraction sanity on SFT.
- **Critical metrics**: `attribution_ratio`, bucket_share, lenient_EX, per-class advantage shift correlation.
- **Highest-risk assumptions**:
  - (R1) Lenient extractor recovers SQL from format-failed SFT trajectories on a 50-record sample (sanity check before pipeline).
  - (R2) `semantic_gain` bucket has ≥10 transitions in (Dense vs SFT) (otherwise C2 has no contrast to test against).
  - (R3) Dense G=8 ≥ Dense G=4 within 1 pp (else G effect confounds SKE story).

### Compute & Timeline

| Block | GPU-h | Wall (1×24GB) | Note |
|---|---:|---|---|
| Lenient re-execution: SFT + Dense75 | ~6 | <1 day | Existing logs |
| Classifier + transition table | ~0 | <1 day | Pure CPU on existing JSONs |
| SKE diagnostic (apply to G4+G8 vs Dense) | ~6 | <1 day | Existing eval JSONs + lenient re-exec |
| Dense G=8 confound run | ~28 (1.2 run-eq + eval) | ~1.5 days | Single confound check |
| (Optional, 1-day gated) External probe artifact | ~2 if available | <1 day | HES-SQL/MTIR-SQL public logs |
| (Optional) 3B Dense same-launcher | ~12 | ~1 day | Anchor 3B cliff |
| Writing + figures | n/a | ~1 week | After diagnostics |

**Mandatory**: ~40 GPU-h (1.7 run-eq) + 3 days CPU + 1 week writing. Easily fits 3-week budget.
