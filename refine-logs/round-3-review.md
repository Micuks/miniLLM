# Round 3 Review

**Reviewer**: gpt-5.5 via codex-reply (thread 019dd71e)
**Date**: 2026-04-29

## Scores

| Dimension | Score | Δ vs R2 |
|---|---:|---:|
| Problem Fidelity | 9 | 0 |
| Method Specificity | 9 | +1 |
| Contribution Quality | 9 | 0 |
| Frontier Leverage | 9 | +1 |
| Feasibility | 9 | +1 |
| Validation Focus | 8 | 0 |
| Venue Readiness | 8 | +1 |
| **Overall** | **8.90** | **+0.55** |

## Verdict
**REVISE** (very close; not READY only because E3 uses weak statistical criterion `p > 0.05` = absence of significance, not evidence of equivalence)

- **Drift**: NONE. Anchor preserved.
- **Dominant contribution**: SHARP. Clean center: strict EX vs lenient EX + deterministic classifier = protocol-vs-semantic decomposition. SKE correctly positioned as supporting.
- **Simplicity**: Right size. Dense G=8 appropriately demoted. External probe + 3B no longer distracting.
- **Frontier leverage**: APPROPRIATE. SQLite verifier + lenient extraction + AST/skeleton canonicalization are natural modern primitives. No forced LLM judge.

## Action items

### IMPORTANT — E3 statistical criterion
`p > 0.05` is absence of significance, not evidence of equivalence. Replace with effect-size + equivalence margin:
- Report **Cliff's delta** or **rank-biserial correlation** with 95% CI
- Pre-define an equivalence margin (e.g., |Cliff's δ| ≤ 0.15) before seeing results
- Accept E3 only if the CI lies inside the margin

### MINOR — Bucket-share denominator explicit
Define two views:
- `transition_bucket_share`: denominator = strict-gain transitions {SFT strict wrong → RL strict right}
- `global_bucket_rate`: denominator = all Spider-dev records

### MINOR — Attribution ratio bounds
The ratio can go negative or exceed 1 (if lenient gain exceeds strict gain, or strict gain is tiny). Report the raw **signed** ratio + `lenient_attribution_gap` as the primary number. Do not clip; clipping is presentation-only if needed.

## Simplification Opportunities
1. Fold `lenient_attribution_gap` into main table. Make `attribution_ratio` secondary if denominator small.
2. `observation_repair_share` → subanalysis figure or appendix row only.
3. Remove external probe from main timeline; mention only as "artifact-dependent appendix" (already done).

## Modernization Opportunities
NONE.

<details>
<summary>Raw response</summary>
See `.aris/traces/research-refine/2026-04-29_run01/round3.txt`.
</details>
