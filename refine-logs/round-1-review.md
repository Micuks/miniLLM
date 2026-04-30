# Round 1 Review

**Reviewer**: gpt-5.5 via codex exec, xhigh reasoning
**Session**: 019dd71e-70d4-7541-9a31-edbe9bfe6ccb
**Date**: 2026-04-29

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9 |
| Method Specificity | 6 |
| Contribution Quality | 8 |
| Frontier Leverage | 8 |
| Feasibility | 8 |
| Validation Focus | 6 |
| Venue Readiness | 6 |
| **Overall** | **7.35** |

## Verdict
**REVISE**

## Action items (priority-ordered)

### CRITICAL — Method Specificity (6)
The diagnostic buckets (`format_fix`, `observation_repair`, `semantic_repair`, `protocol_only`) are conceptually clear but not operational. Phrases like "recovers via observation", "trajectory shape identical", "SQL-equivalent edits" leave too much script discretion.

**Fix**: Define a strict classifier interface:
- Inputs: `strict_pred_sql`, `lenient_pred_sql`, `gold_sql`, `all_sql_actions`, `execution_errors`, `final_answer_span`, `db_id`.
- Outputs: one mutually exclusive bucket plus per-record flags `strict_extract_ok`, `lenient_extract_ok`, `first_sql_exec_ok`, `final_sql_exec_ok`, `final_correct`, `sql_ast_changed`, `table_set_changed`, `condition_set_changed`.
- Explicit bucket precedence: `format_fix > observation_repair > semantic_repair > protocol_only`.

### IMPORTANT — Validation Focus (6)
Claim 2 risks becoming a second paper. M1-M4 mechanism grid + Dense-G8 control reads as forensics, dilutes the diagnostic contribution.

**Fix**: Collapse SKE analysis to **one** primary falsifiable mechanism: *"SKE fails because semantic structure is not the bottleneck."* Required evidence:
- SKE does not increase `semantic_repair`.
- SKE does not improve lenient EX over Dense.
- SKE mainly changes class-baseline/advantage statistics without moving protocol errors.

Keep Dense-G=8 only as a **confound check**, not a centerpiece.

### IMPORTANT — Venue Readiness (6)
Strong workshop material as written; main-track readiness depends on whether the diagnostic is *general* vs a local autopsy of one failed SKE variant.

**Fix**: Add one small external sanity demo if available — apply strict-vs-lenient decomposition to one published or reproduced agentic SQL baseline. Even a partial trajectory-compatible baseline would help show portability.

## Simplification Opportunities
1. Delete M1-M4 framing from main method. Keep as appendix diagnostic checklist. Main paper says: protocol bottleneck explains why structural prior has little room to help.
2. Merge `format_fix` and strict-vs-lenient attribution into one **protocol-validity** section. They measure adjacent phenomena.
3. Make `observation_repair` a **flag**, not a top-level bucket. Clean split = protocol gain vs semantic gain; repair-after-observation as subanalysis.

## Modernization Opportunities
NONE. Already appropriately modern (verifier-style re-execution as the natural fit).

## Drift Warning
NONE. Anchored on diagnostic gap. Only drift risk: M1-M4 grid becoming parallel contribution.

<details>
<summary>Raw response</summary>

See `.aris/traces/research-refine/2026-04-29_run01/round1.txt` for full verbatim trace.

</details>
