# Research Proposal: Protocol Beats Structure — A Diagnostic Study of Agentic GRPO for Text-to-SQL

## Problem Anchor

- **Bottom-line problem**: Multi-turn ReAct GRPO on Text-to-SQL succeeds modestly at 7B (+2 pp Spider-dev EX_att/EX_all) but fails at 3B; a structural-advantage prior (SKE-RL) fails to help and worsens with more rollouts. We do not know what the +2 pp actually changed inside the agent, and we cannot explain why a natural structural prior fails.
- **Must-solve bottleneck**: Diagnostic gap. The literature reports aggregate EX gains without decomposing protocol-compliance vs SQL-reasoning contributions, so neither positive (Dense GRPO) nor negative (SKE) results can be interpreted mechanistically.
- **Non-goals**: SOTA on Spider/BIRD; new method that fixes everything; scaling-law claim across model sizes; a toolkit-as-lead paper.
- **Constraints**: 1× Quadro RTX 6000 24 GB, ~3 weeks GPU + 1 week writing; SKE-RL stack + per-class diagnostics + full Spider-dev eval JSONs already on disk; workshop-tier deadline driven.
- **Success condition**: A reviewer would say the paper changes how they read agentic-SQL-RL leaderboards (because of the protocol decomposition) AND the SKE negative result has a single, convincing, measured mechanism (not "we tried it and it didn't work").

## Technical Gap

Recent agentic Text-to-SQL RL work (HES-SQL, MTIR-SQL, MARS-SQL, Graph-Reward-SQL) reports +N pp EX_all gains and treats those gains as evidence of improved SQL reasoning under structural / execution / multi-agent supervision. **No paper in this line decomposes the gain into "the model produced more valid trajectories" vs "the model produced semantically better SQL".** The two are systematically conflated under EX_all because EX_all = (attempted-rate × EX_att), and aggregate metrics give attempted-rate gains the same weight as semantic gains.

This conflation has direct consequences:

- **Interpretation failure**: Existing skeleton-reward methods (HES-SQL) assume gain comes from structural alignment. If gain is largely format-compliance, the structural reward is doing something other than what the paper claims.
- **Negative-result opacity**: Our SKE-RL — the natural "advantage-side cousin" of HES-SQL skeleton reward — fails. Without a protocol/semantic decomposition we cannot explain whether SKE failed because (a) skeleton is wrong abstraction, (b) the bottleneck is upstream of structure, (c) class baseline math degenerates, or (d) optimization side-effects.
- **Naive bigger systems do not fix this**: Throwing more rollouts (G=8), more rewards (Graph-Reward), or multi-agent decomposition (MARS) does not surface what is actually being learned. They report the same conflated metric.

The smallest adequate intervention is a **measurement protocol**, not a new training method. We instrument existing eval JSONs and one new diagnostic eval to split EX_all into a 4-cell taxonomy {format-fix / executable-but-wrong / executable-and-better-SQL / unchanged}, then map SKE's negative result onto that taxonomy to identify the mechanism.

## Method Thesis

- **One-sentence thesis**: Agentic GRPO on multi-turn Text-to-SQL improves aggregate accuracy primarily by aligning tool-use protocols (valid action format, valid SQL syntax, valid Answer extraction, repair-after-observation), not by improving SQL semantic reasoning — and this protocol-vs-semantic split is what makes the SKE-RL structural-advantage prior fail (it targets the layer below the actual bottleneck, and degrades with more rollouts because larger groups give the class-baseline more opportunity to wash out the few semantically-corrective trajectories).
- **Smallest adequate intervention**: A diagnostic measurement layer (lenient-extraction re-eval + win/loss transition table + repair-attribution counter). No new training. Existing 4 checkpoints (SFT, Dense75, SKE-G4-75, SKE-G8-75) become the data; the contribution is the lens through which they are read.
- **Why timely**: Foundation-model-era agentic RL (web agents, code agents, tool-using assistants) all report aggregate task-success gains. The protocol-vs-semantic conflation we identify on Text-to-SQL almost certainly applies to those settings. The paper offers a portable diagnostic, not a SQL-specific trick.

## Contribution Focus

- **Dominant contribution**: An empirical diagnostic showing that Dense agentic GRPO gain on Text-to-SQL is primarily protocol/tool-use alignment (lenient-extraction Δ ≪ strict-extraction Δ; win/loss transitions concentrated in {format_fix, observation_repair} not {semantic_repair}), and a portable protocol that surfaces this split in any agentic-RL evaluation.
- **Optional supporting contribution**: A mechanism-grounded negative result for SKE-RL — skeleton-equivalence-class advantage estimation fails *because the bottleneck is at the protocol layer it cannot affect*, with G=8 monotonic degradation explained by the measured fact that {larger G inflates the within-protocol-failure cluster, washing out the rare semantic-repair signal in the class baseline}.
- **Explicit non-contributions**: No new RL method that wins; no scaling law; no SOTA Spider/BIRD numbers; no claim about model "reasoning capacity"; no toolkit paper.

## Proposed Method

### Complexity Budget

- **Frozen / reused**: All 4 existing 7B checkpoints (SFT, Dense75, SKE-G4-75, SKE-G8-75); existing eval JSONs with full per-record trajectories; existing skeleton extractor; existing query-class tagger; existing aggregator. No new model training for the dominant contribution.
- **New trainable components**: **0 for the dominant contribution.** For the supporting contribution (SKE mechanism), one new low-cost training run: Dense-G=8 control (~1.2 run-eq) to disambiguate "G effect" from "SKE effect". (≤1 new component, well under the MAX_NEW_TRAINABLE_COMPONENTS=2 cap.)
- **Tempting additions intentionally not used**: Multi-seed averaging (>3 run-eq); BIRD GRPO (~2-4 run-eq); 1.5B/14B capacity points; v2 SKE (skeleton-first generation); reward-side skeleton bonus large-scale comparison (HES-SQL replication is out of scope).

### System Overview

```
Existing eval JSONs (SFT, Dense75, SKE-G4-75, SKE-G8-75; full Spider-dev)
   │
   ├─ Diagnostic layer 1: STRICT vs LENIENT extraction
   │     re-extract any plausible final SQL from each trajectory transcript
   │     re-execute, recompute EX_all_lenient
   │     ΔEX_lenient < ΔEX_strict  →  RL gain = protocol/format
   │     ΔEX_lenient ≈ ΔEX_strict  →  RL gain = semantic
   │
   ├─ Diagnostic layer 2: WIN/LOSS TRANSITION TAXONOMY (per-record)
   │     classify each query by SFT-vs-Dense outcome:
   │       format_fix          : SFT format/tool failed,   Dense executes
   │       observation_repair  : both attempt,  Dense recovers via observation
   │       semantic_repair     : both produce executable SQL, Dense's is correct (skeleton/tables/conditions changed)
   │       protocol_only       : SFT executable, Dense executable, no semantic change, both correct or both wrong
   │     report % of {SFT-wrong → Dense-right} delta in each bucket
   │
   ├─ Diagnostic layer 3: TURN-LEVEL REPAIR COUNTERS
   │     per-trajectory: # of execute_sql actions, % first-action-failed, % final-correct-after-error
   │     does Dense use observations more effectively (semantic) or just produce fewer errors (protocol)?
   │
   └─ SKE mechanism mapping
         apply same 3 layers to SKE-G4-75 vs Dense75 and SKE-G8-75 vs Dense75
         pick the mechanism candidate whose signature matches the data:
           M1 variance-reduction failure → measure advantage variance per group; SKE doesn't reduce vs Dense
           M2 class fragmentation        → measure |distinct skeleton classes| / G; many singleton classes
           M3 wrong abstraction          → semantic_repair queries have HIGH within-skeleton-class outcome variance (so class baseline gives 0 information about who in the class will repair)
           M4 optimization side-effect   → measure advantage sign distribution per class; SKE shifts toward zero/negative for exploratory rollouts
```

### Core Mechanism (dominant contribution = the diagnostic protocol)

- **Input**: an eval JSON with per-record trajectories from an agentic RL pipeline (we provide reference implementation for ReAct + execute_sql). Required fields: `question`, `gold_sql`, `pred_sql`, `trajectory` (full ReAct text), `execution_match` (or recoverable from re-execution), `db_id`/`db_path`.
- **Output**: A 4-row table per (RL-method vs SFT) pair:

| Bucket | Definition | What it tells you |
|---|---|---|
| format_fix | SFT trajectory has no extractable SQL OR malformed Action; RL trajectory executes | RL fixed protocol/format — *not reasoning* |
| observation_repair | Both attempt SQL; first action failed; RL final SQL correct via observation use | RL improved error-recovery — *partial reasoning* |
| semantic_repair | Both produce executable SQL of different skeleton/table-set; RL is correct | RL improved SQL semantics — *true reasoning gain* |
| protocol_only | Trajectory shape identical (or only SQL-equivalent edits); both executable | RL did not change the answer; either both right or both wrong |

  Plus a single headline number: `lenient_extraction_attribution = (ΔEX_strict - ΔEX_lenient) / ΔEX_strict ∈ [0, 1]`. If 1.0, all gain is protocol; if 0.0, all gain is semantic.

- **Why this is the main novelty**: No published agentic Text-to-SQL RL paper reports either of these two measurements. Both are interpretive primitives, not engineering tricks. They generalize trivially to other tool-using agentic RL settings (browser agents, code agents) by swapping {SQL extractor, gold-equivalence test} for the equivalent task primitives.

### Optional Supporting Component (= mechanism for SKE failure)

- Apply the same 3 diagnostic layers to SKE-G4-75 and SKE-G8-75.
- Match the SKE signature against the 4 mechanism candidates (M1-M4 above). Each candidate has a distinctive observable signature; the data picks one.
- The SKE negative result then becomes "SKE fails because mechanism M_*: [evidence row]", not "SKE fails for unknown reasons". Crucially, M3 (wrong abstraction) is the prediction made by the dominant contribution: if the bottleneck is protocol, then a structural advantage prior cannot help, and one would expect class-baseline subtraction to *wash out* the rare semantic-repair signal that *does* exist within high-protocol-failure groups → larger G makes this worse.
- Why no contribution sprawl: M_* is *predicted by* the dominant thesis. Confirming it strengthens both, not adds a parallel claim.

### Modern Primitive Usage

- **LLM** = the agent under study (Qwen2.5-7B). Not novel as primitive but central as object.
- **GRPO + tool-call rollout** = the RL setting being diagnosed. We do not propose changes to GRPO.
- **AST-canonical equivalence (sqlglot)** = used in (1) the existing SKE-RL extractor and (2) the new `semantic_repair` bucket detector to verify that two SQLs differ in skeleton/tables, not just literals.
- **Lenient-extraction re-execution** = leverages real SQLite execution as a verifier — a foundation-model-era inference-time-verifier pattern, applied here to *recover what the agent meant to say despite protocol noise*. This is the natural primitive: rather than train the model to be more compliant, we instrument the eval to be tolerant, and the gap between tolerant-eval and strict-eval IS the protocol-attribution measurement.
- **No new trained component is required** for the dominant contribution. This is intentional: the paper's leverage is observational, not parametric.

### Integration into Base Generator / Downstream Pipeline

- The diagnostic layer attaches **after** an arbitrary agentic-RL pipeline. It reads eval JSONs only. No re-training, no model surgery.
- Reference implementation works on our existing JSON schema (`{summary, records: [{question, gold_sql, pred_sql, trajectory, execution_match, db_id, ...}]}`).
- For the SKE mechanism diagnostic, a minor extension records per-rollout `n_real_classes`, `class_size_distribution`, `advantage_per_class` during training; our `train_grpo.py` already logs these (R3-R4 patches).

### Training Plan

For the dominant contribution: **none — no training**. The 4 existing 7B checkpoints + their full Spider-dev eval JSONs are the data.

For the supporting SKE-mechanism contribution:
- **One** new GRPO training run: Dense G=8 control, 75 steps, identical recipe to Dense G=4 except group size. Cost ~1.2 run-eq. Gives the missing comparison cell {Dense G=4, Dense G=8, SKE G=4, SKE G=8} so G-effect and SKE-effect are not entangled.
- (Optional, if time) 3B Dense GRPO same launcher, ~0.5 run-eq, anchors the "3B cliff" claim with the current launcher generation.

### Failure Modes and Diagnostics

| Failure mode | How we detect | Fallback |
|---|---|---|
| `lenient_extraction_attribution` ≈ 0 (Dense gain is fully semantic) | Lenient ΔEX ≈ Strict ΔEX | Pivot story to "RL gives genuine reasoning lift; SKE still fails; mechanism = M3 wrong abstraction or M2 fragmentation". Diagnostic toolkit still novel. |
| Dense G=8 also degrades as much as SKE G=8 | Dense G=8 EX_att ≪ Dense G=4 EX_att | The G-effect is generic, not SKE-specific. Story shifts to "rollout-budget-vs-quality trade-off in agentic GRPO" — still publishable diagnostic. |
| Win/loss transition table is dominated by `protocol_only` (no transitions in either direction) | Both pipelines produce the same outcomes on most queries | Δ effect is small/noise; report effect-size confidence interval honestly; this is itself a finding (Dense gain is noise, not real). Might force pivot to v2. |
| Lenient extraction is unreliable (no extractable SQL even by greedy regex) | Lenient EX_all < Strict EX_all on SFT (sanity check) | Tighten extractor to "any final-Answer-region SQL that parses with sqlglot"; re-run sanity. |
| Mechanism candidates ambiguous (signature matches 2 of M1-M4) | Two M_i scores within 1 SD of each other on the SKE-vs-Dense diagnostics | Report both as candidate mechanisms with the data; do not over-claim a single pick. |

### Novelty and Elegance Argument

- **Closest published work**:
  - HES-SQL (2510.08896): GRPO + skeleton-completeness reward. Uses skeleton as REWARD signal; reports +N pp; does not decompose attribution.
  - Graph-Reward-SQL (Findings EMNLP 2025): graph-matching stepwise reward. Same conflation issue.
  - MTIR-SQL (2510.25510): multi-turn tool-integrated GRPO. Reports aggregate EX; no protocol/semantic split.
  - MARS-SQL (2511.01008): multi-agent ReAct + validation. Same.
  - LearNAT (2504.02327): AST-guided decomposition + DPO. Reward-side structure; not advantage-side; no diagnostic.
- **Exact difference**: Every above paper sits inside the conflation — they propose interventions and show aggregate EX gains. We do not propose a winning intervention. We propose **a diagnostic that turns those existing aggregate gains into mechanistic claims**, and use SKE-RL (a natural advantage-side cousin of HES-SQL) as a designed-to-fail control that *makes the diagnostic visible*.
- **Why this is focused, not module-pile-up**: One measurement layer. One supporting failure analysis that is *predicted by* the main thesis. No new training for the dominant contribution. The whole paper rotates around a single question — "what does agentic SQL RL actually change?" — with exactly one new way to answer it.
- **Why elegant**: The diagnostic is one re-execution pass + one trajectory-classification table. Anyone with eval JSONs from any agentic Text-to-SQL RL paper can apply it tomorrow. That portability is the elegance.

## Claim-Driven Validation Sketch

### Claim 1 (dominant): Dense agentic GRPO gain on Text-to-SQL is dominantly protocol/tool-use alignment, not SQL semantic reasoning

- **Minimal experiment**: Lenient-extraction re-eval of SFT vs Dense75 on full Spider-dev (1034); compute `lenient_extraction_attribution`. Plus win/loss transition table on the 200-query subset for which we already have rich trajectory logs. **Cost: ~0.4 run-eq for the lenient re-execution; 0 for the table.**
- **Baselines / ablations**: SFT (no RL) vs Dense75 (RL); within Dense, examine the ΔEX_lenient vs ΔEX_strict gap.
- **Metric**: `lenient_extraction_attribution` ∈ [0, 1] for Dense-vs-SFT. % of {SFT-wrong → Dense-right} transitions in each of {format_fix, observation_repair, semantic_repair, protocol_only}.
- **Expected directional outcome**: Attribution ≥ 0.5 (most gain is protocol). format_fix + observation_repair > 60% of positive transitions; semantic_repair < 25%. If we see attribution < 0.3 (mostly semantic), the story flips — but the diagnostic itself is still novel and the paper just reframes.

### Claim 2 (supporting): SKE-RL fails because it targets the SQL-structure layer while the bottleneck is at the protocol layer; G=8 degradation is the predicted consequence

- **Minimal experiment**: Apply the same diagnostic layers to SKE-G4-75 vs Dense75 and SKE-G8-75 vs Dense75. Score the 4 mechanism candidates M1-M4 from per-rollout SKE-training logs (already captured by R3 patches: `n_real_classes`, `class_size_distribution`, advantage variance per class) + the new diagnostic. **Plus** one new training run: Dense G=8 control (~1.2 run-eq) to confirm G-effect is not generic.
- **Baselines / ablations**: Dense G=4, Dense G=8, SKE G=4, SKE G=8 — full 2×2.
- **Metric**: Per-mechanism signature score (a tuple per candidate); the winning candidate has the largest difference between its predicted-signature value and the alternatives. Specifically the M3 prediction "SKE wins on protocol_only / SKE loses on semantic_repair, and the loss grows with G" is the keystone.
- **Expected directional outcome**: M3 (wrong abstraction) wins. Dense G=8 ≥ Dense G=4 (so the G effect is not generic — SKE's degradation is structural to the class baseline, not to G alone).

## Experiment Handoff Inputs

- **Must-prove claims**: C1 (protocol attribution); C2 (SKE mechanism = M3 or whatever the data picks).
- **Must-run ablations**: Dense G=8 control; lenient-extraction sanity on SFT (lenient ≥ strict by construction).
- **Critical datasets / metrics**: Spider-dev full 1034, lenient_extraction_attribution, 4-bucket transition table, per-mechanism-candidate signatures.
- **Highest-risk assumptions**:
  - (R1) Lenient-extraction can recover SQL from most "format-failed" SFT trajectories — needs sanity check on a 50-record sample.
  - (R2) `semantic_repair` bucket is non-empty (≥10 records); if it isn't, the protocol-only conclusion is too dominant and the SKE-mechanism story has too little signal to interpret.
  - (R3) Dense G=8 ≥ Dense G=4 (not generically degrading); without this we cannot attribute SKE-G=8's collapse to SKE-specific machinery.

## Compute & Timeline Estimate

| Block | GPU-h | Wall (1×24GB) | Note |
|---|---:|---|---|
| Lenient re-execution: SFT + Dense75 (1034) | ~6 (SQLite-bound, no GPU rollout) | <1 day | Reuses existing trajectory logs; only re-extracts and re-executes SQL. |
| Trajectory classifier + transition table | ~0 | <1 day | Pure CPU on existing JSONs. |
| Per-mechanism SKE diagnostic | ~0 | <1 day | Reuses training-time SKE logs + the new diagnostic. |
| Dense G=8 control run | ~28 (1.2 run-eq + eval) | ~1.5 days | Disambiguates G effect. |
| (Optional) 3B Dense same-launcher | ~12 | ~1 day | Anchors 3B cliff; not strictly required for either claim. |
| (Optional) BIRD-mini Dense + diagnostic | ~30-50 | ~2 days | Generality probe; not required for workshop. |
| Writing + figures | n/a | ~1 week | After diagnostics land. |

**Total mandatory: ≤ ~36 GPU-h (1.5 run-eq) + ~3 days CPU/eval work + 1 week writing.**
**With both optionals: ~80 GPU-h (3.3 run-eq).**

Both fit comfortably in the 3-week budget.
