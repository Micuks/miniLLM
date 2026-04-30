# Review Summary

**Problem**: Multi-turn ReAct GRPO on Text-to-SQL — Dense 7B +2 pp; SKE-RL fails and worsens with G; we lack a mechanistic explanation
**Initial Approach**: A+B story (protocol-vs-reasoning decomposition + SKE-as-mechanism-evidence)
**Date**: 2026-04-29
**Rounds**: 4 / 5
**Final Score**: 9.00 / 10
**Final Verdict**: READY

## Problem Anchor (verbatim)

- **Bottom-line problem**: Multi-turn ReAct GRPO on Text-to-SQL succeeds modestly at 7B (+2 pp Spider-dev EX_att/EX_all) but fails at 3B; SKE-RL fails to help and worsens with more rollouts. We do not know what the +2 pp actually changed inside the agent, and we cannot explain why a natural structural prior fails.
- **Must-solve bottleneck**: Diagnostic gap. Literature reports aggregate EX gains without decomposing protocol-compliance vs SQL-reasoning contributions.
- **Non-goals**: SOTA; new method that wins; scaling-law claim; toolkit-as-lead.
- **Constraints**: 1× Quadro RTX 6000 24 GB, ~3 weeks GPU + 1 week writing.
- **Success condition**: Reviewer says paper changes how they read agentic-SQL-RL leaderboards AND the SKE negative result has a single, measured, convincing mechanism.

## Round-by-Round Resolution Log

| Round | Main reviewer concerns | What this round simplified / modernized | Solved? | Remaining risk |
|-------|-------------------------|------------------------------------------|---------|----------------|
| 1 | Buckets non-deterministic (CRITICAL); M1-M4 grid sprawl (IMPORTANT); generality concern (IMPORTANT) | Spec'd classifier interface; collapsed M1-M4 to 1 mechanism with E1-E3; demoted Dense G=8 to confound; merged buckets to 3 with `observation_repair` as flag | yes | Statistical rigor of E3 |
| 2 | Lenient leakage risk (CRITICAL); bucket rules underspecified (CRITICAL); E3 join key (IMPORTANT); CIs missing (IMPORTANT) | Locked lenient_extract to deterministic gold-free rule; rewrote bucket rules using direct strict/lenient correctness; defined skeleton_class join + Tier-3 fallback; added bootstrap CIs everywhere | yes | E3 used `p > 0.05` which is non-equivalence |
| 3 | E3 statistical criterion weak (IMPORTANT); bucket denominator ambiguous (MINOR); ratio bounds wrong (MINOR) | Replaced p-value with Cliff's δ + pre-registered margin |δ| ≤ 0.147; added two views (transition_bucket_share, global_bucket_rate); switched primary metric to signed `lenient_attribution_gap` | yes | None blocking |
| 4 | None blocking | Verified all R3 fixes; READY at 9.00 | yes | Execution-side risks (small effect size, sparse semantic_gain, possible E3 fail) — handled in failure-modes table |

## Overall Evolution

- **Method became more concrete**: From conceptual buckets ("recovers via observation", "trajectory shape identical") in R0 → to a fully deterministic Python interface with explicit flag set, mutually-exclusive ordered bucket rules, and leakage-safe extractor in R3.
- **Dominant contribution became more focused**: From "protocol-vs-reasoning decomposition + 4-mechanism SKE forensics" in R0 → to "one signed attribution gap + 3-bucket classifier; SKE collapsed to 1 falsifiable mechanism predicted by the main thesis".
- **Unnecessary complexity removed**: M1-M4 grid moved from main body to appendix checklist; 3B Dense rerun moved from "must-run" to appendix-only-if-time; external-baseline probe moved from §6.4 to artifact-dependent footnote; bucket count reduced from 4 → 3.
- **Modern technical leverage stayed appropriately minimal**: Verifier-style re-execution + AST canonicalization. NONE in modernization opportunities every round — the proposal is appropriately measurement-first, not LLM-judge-driven.
- **Drift avoided**: NONE in every drift warning. The reviewer specifically called out the M1-M4 grid as the main drift risk in R1 and we eliminated it. Anchor preserved across all rounds.

## Final Status
- **Anchor status**: PRESERVED (all 4 rounds: drift = NONE).
- **Focus status**: TIGHT (one classifier + one signed gap + one supporting falsifiable mechanism).
- **Modernity status**: APPROPRIATELY FRONTIER-AWARE (verifier-style re-execution; no forced LLM judge).
- **Strongest parts of final method**: (i) leakage-safe lenient extractor with explicit deterministic rule; (ii) direct-correctness bucket rules trivially auditable from per-record table; (iii) E3 as a pre-registered equivalence test, not a non-significance argument; (iv) zero new training for the dominant contribution; (v) the diagnostic generalizes by parameter swap to any tool-using agentic RL setting.
- **Remaining weaknesses (execution-side, not design-side)**: (R1) lenient extractor must recover SQL on ≥80% of format-failed SFT trajectories (sanity check first); (R2) `semantic_gain` bucket may be sparse (<10 transitions); (R3) Dense G=8 may degrade and confound the SKE story; (R4) E3 join coverage may be <50%. All have documented fallbacks in the proposal.
