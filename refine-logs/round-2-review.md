# Round 2 Review

**Reviewer**: gpt-5.5 via codex-reply (same thread 019dd71e)
**Date**: 2026-04-29

## Scores

| Dimension | Score | Δ vs R1 |
|---|---:|---:|
| Problem Fidelity | 9 | 0 |
| Method Specificity | 8 | +2 |
| Contribution Quality | 9 | +1 |
| Frontier Leverage | 8 | 0 |
| Feasibility | 8 | 0 |
| Validation Focus | 8 | +2 |
| Venue Readiness | 7 | +1 |
| **Overall** | **8.35** | **+1.0** |

## Verdict
**REVISE** (not READY: <9; one blocking methodological detail — leakage-safe lenient extractor + pairwise-consistent bucket rules)

## Drift / focus / simplicity / modernity statements
- **Drift**: NONE. Anchor preserved.
- **Dominant contribution**: SHARPER. Buckets to 3, observation_repair as flag. SKE reads as supporting.
- **Method simplicity**: Simpler but not fully clean — classifier has one fragile edge.
- **Frontier leverage**: APPROPRIATE. Verifier-style re-execution + AST canonicalization, no LLM judge needed.

## Action items

### CRITICAL — Lenient extraction leakage
Multi-candidate "best executes" → pseudo-oracle eval. Fix: single deterministic rule, no gold-based candidate selection:

```
lenient_pred_sql = (
    final-Answer-region SQL if parseable
    else last parseable SQL action
    else None
)
```

### CRITICAL — Classifier underspecified
Current `protocol_gain = ... AND (RL.lenient_correct == SFT.lenient_correct)` is awkward. Rewrite as direct strict/lenient correctness rules:

```python
protocol_gain = (not SFT.final_correct_strict
                 and SFT.final_correct_lenient
                 and RL.final_correct_strict)
semantic_gain = (not SFT.final_correct_lenient
                 and RL.final_correct_lenient)
no_change = otherwise
# precedence: protocol_gain > semantic_gain > no_change
```

### IMPORTANT — E3 join key for SKE
"Class-baseline shifts uncorrelated with semantic_gain records" needs explicit join between training rollout stats and eval records. Define now: exact id (if same examples) > skeleton class / db_id / difficulty > aggregate-only fallback.

### IMPORTANT — Uncertainty intervals
Report binomial / bootstrap CIs for `attribution_ratio` and bucket shares. Required uncertainty accounting, not new experiments.

## Simplification Opportunities
1. Drop optional 3B Dense run from main proposal. Anchor mentions 3B cliff but contribution does not require it.
2. External-baseline probe → footnote/appendix-only "if artifacts exist". Do NOT include in main validation.
3. Remove M1-M4 mention from proposal body entirely. Appendix-only is fine, but body mention creates latent sprawl.

## Modernization Opportunities
NONE. Right level of FM-era machinery. LLM judge would add noise.

<details>
<summary>Raw response</summary>
See `.aris/traces/research-refine/2026-04-29_run01/round2.txt`.
</details>
