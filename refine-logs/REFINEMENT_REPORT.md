# Refinement Report

**Problem**: Multi-turn ReAct GRPO on Text-to-SQL — Dense 7B +2 pp; SKE-RL fails and worsens with G; mechanistic explanation missing
**Initial Approach**: A+B story (protocol-vs-reasoning decomposition + SKE-as-mechanism-evidence)
**Date**: 2026-04-29
**Rounds**: 4 / 5
**Final Score**: 9.00 / 10
**Final Verdict**: READY

## Problem Anchor (verbatim)

See `round-0-anchor.md` for the full anchor. Summary:
- Bottom-line problem: We don't know what +2 pp Dense GRPO actually changed; we can't explain SKE failure.
- Must-solve bottleneck: Diagnostic gap in agentic Text-to-SQL RL literature (no protocol-vs-semantic decomposition).
- Non-goals: SOTA; new winning method; scaling law; toolkit-as-lead.
- Constraints: 1×24GB; 3 weeks GPU + 1 week writing; existing 4 checkpoints + diagnostic stack on disk.
- Success: Paper changes how reviewer reads agentic-SQL-RL leaderboards AND SKE negative result has a single measured mechanism.

## Output Files
- Initial proposal: `refine-logs/round-0-initial-proposal.md`
- Round 1 review + refinement: `round-1-review.md`, `round-1-refinement.md`
- Round 2 review + refinement: `round-2-review.md`, `round-2-refinement.md`
- Round 3 review + refinement: `round-3-review.md`, `round-3-refinement.md`
- Round 4 review (READY): `round-4-review.md`
- Review summary: `REVIEW_SUMMARY.md`
- Final proposal: `FINAL_PROPOSAL.md`
- Score evolution: `score-history.md`
- Raw codex traces: `.aris/traces/research-refine/2026-04-29_run01/round{1..4}.txt`
- Codex thread for resumption: `019dd71e-70d4-7541-9a31-edbe9bfe6ccb`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1 | 9 | 6 | 8 | 8 | 8 | 6 | 6 | 7.35 | REVISE |
| 2 | 9 | 8 | 9 | 8 | 8 | 8 | 7 | 8.35 | REVISE |
| 3 | 9 | 9 | 9 | 9 | 9 | 8 | 8 | 8.90 | REVISE |
| 4 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9.00** | **READY** |

## Round-by-Round Review Record

| Round | Main reviewer concerns | What was changed | Result |
|-------|-------------------------|------------------|--------|
| 1 | Buckets need deterministic rules (CRITICAL); M1-M4 grid risks 2nd-paper sprawl (IMPORTANT); generality (IMPORTANT) | Spec'd classifier; collapsed mechanism grid to single E1+E2+E3; demoted Dense G=8 to confound; bucket count 4→3 | resolved |
| 2 | Lenient leakage (CRITICAL); bucket rules awkward (CRITICAL); E3 join undefined (IMPORTANT); no CIs (IMPORTANT) | Locked deterministic gold-free extractor; rewrote bucket rules with direct strict/lenient predicates; defined skeleton_class join + Tier-3 fallback; bootstrap CIs everywhere | resolved |
| 3 | E3 used `p > 0.05` (non-equivalence, IMPORTANT); bucket denominator ambiguous (MINOR); ratio not [0,1] (MINOR) | Replaced p-value with Cliff's δ + pre-registered margin 0.147; two-view bucket share; signed gap as primary metric | resolved |
| 4 | None blocking | Verified all R3 fixes intact | READY |

## Final Proposal Snapshot
- Canonical clean version lives in `refine-logs/FINAL_PROPOSAL.md`.
- 5-bullet summary:
  1. **Thesis**: Agentic GRPO on multi-turn Text-to-SQL improves aggregate accuracy primarily by aligning tool-use protocols, not by improving SQL semantic reasoning.
  2. **Dominant contribution**: A deterministic, leakage-safe diagnostic protocol — strict-vs-lenient re-execution + 3-bucket classifier + signed `lenient_attribution_gap` with bootstrap CIs.
  3. **Supporting contribution**: SKE-RL fails because the bottleneck is at the protocol layer it cannot affect; confirmed by E1 + E2 + E3 (Cliff's δ equivalence test, pre-registered margin |δ| ≤ 0.147).
  4. **Method posture**: Zero new training for the dominant contribution; one Dense G=8 confound run (~1.2 run-eq) for SKE.
  5. **Generality**: Diagnostic generalizes to any tool-using agentic RL setting by swapping `gold_sql` → `task_success_predicate` and `execute_sql` → arbitrary `tool_call`.

## Method Evolution Highlights

1. **Most important simplification**: M1-M4 mechanism grid removed entirely from proposal body in R3 (kept only as appendix checklist for future researchers). This eliminated the second-paper sprawl risk that the R1 reviewer flagged as the only drift threat.

2. **Most important mechanism upgrade**: Lenient extractor locked in R2 to a deterministic gold-free rule (`first parseable SQL in Answer-region else last parseable SQL action else None`). This eliminated the "pseudo-oracle eval" leakage risk and made the diagnostic legitimate as a measurement.

3. **Most important rigor upgrade**: E3 evidence test rewritten in R3 from `Mann-Whitney p > 0.05` (non-significance — invalid for equivalence claims) to Cliff's δ with 95% bootstrap CI vs pre-registered margin |δ| ≤ 0.147. This gives the SKE single-mechanism claim a properly testable falsification criterion.

## Pushback / Drift Log

| Round | Reviewer said | Author response | Outcome |
|-------|---------------|-----------------|---------|
| 1 | "Add external baseline (HES-SQL/MTIR-SQL) for portability" | Conditionally accepted: ONLY if public per-record artifacts available within 1-day search; otherwise dropped as limitation. No reproduction. | accepted with gate; later (R3) demoted to artifact-dependent appendix-only |
| 2 | "Define E3 join key explicitly" | Defined Tier 1/2/3 join hierarchy with `skeleton_class` as Tier 2 (used) and aggregate-only as Tier 3 fallback (used if join coverage <50%) | accepted |
| 3 | "Replace E3 p-value with equivalence test" | Replaced with Cliff's δ + pre-registered margin |δ| ≤ 0.147 (Vargha-Delaney small-effect threshold). Failure mode = "consistent with but not proving" rather than collapse. | accepted |
| 3 | "Drop optional 3B Dense from main" | Accepted: removed from §Compute & Timeline; relegated to "appendix-only-if-time" with no main-text claim dependency | accepted |

No reviewer suggestion was rejected as drift in any round.

## Remaining Weaknesses

**Proposal-side**: NONE blocking. All R1-R3 critiques resolved.

**Execution-side risks** (legitimate empirical outcomes, all with documented fallbacks):
- (R1) Lenient extractor must recover SQL on ≥80% of format-failed SFT trajectories — sanity check on 50-record sample before pipeline.
- (R2) `semantic_gain` bucket may be sparse (<10 transitions in Dense vs SFT) — would weaken C2 contrast; honest CI reporting required.
- (R3) Dense G=8 may degrade ≥1pp vs Dense G=4 — would mean generic G effect, weakening SKE single-mechanism story; documented as "rollout-budget vs quality trade-off" reframe.
- (R4) E3 join coverage may be <50% — Tier-3 aggregate-only fallback documented.
- (R5) `lenient_attribution_gap` could be ≈ 0 (story flips to semantic gain) — diagnostic still novel, conclusion inverts; narrative reframed in writing phase.

## Raw Reviewer Responses

<details>
<summary>Round 1 Review (verbatim trace path)</summary>

`.aris/traces/research-refine/2026-04-29_run01/round1.txt` — full Codex output.
Score 7.35 / 10, REVISE.

</details>

<details>
<summary>Round 2 Review</summary>

`.aris/traces/research-refine/2026-04-29_run01/round2.txt`.
Score 8.35 / 10, REVISE.

</details>

<details>
<summary>Round 3 Review</summary>

`.aris/traces/research-refine/2026-04-29_run01/round3.txt`.
Score 8.90 / 10, REVISE.

</details>

<details>
<summary>Round 4 Review (READY)</summary>

`.aris/traces/research-refine/2026-04-29_run01/round4.txt`.
Score 9.00 / 10, READY.

</details>

## Next Steps
- READY → proceed to `EXPERIMENT_PLAN.md` for full execution-ready experiment roadmap (week-by-week schedule, GPU costs, claim-experiment matrix).
- Then `/run-experiment` to execute Week 1 zero-cost diagnostics, then Dense G=8 confound run, then writing.
- Codex thread `019dd71e-70d4-7541-9a31-edbe9bfe6ccb` remains available for v2 / response-to-review iterations after submission.
