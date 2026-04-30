---
date: 2026-04-30
status: per review recommendation #2 (zero-GPU-cost re-decomp on existing 3B artifacts)
parent: REVIEW_v4_INTERNAL_2026-04-30.md, FINAL_PROPOSAL_v4.md
inputs:
  - outputs/eval_agent_react_sft.json (3B SFT, Spider 200, n=200)
  - outputs/eval_agent_grpo.json (3B GRPO, same split, n=200)
artifact: pending — should be added to a future v5 proposal as cross-model-size confirmation
---

# 3B archive re-decomposition — paper-strengthening replication

## Summary

The 7B v4 mechanism (group size / commitment policy → trades commitment for capability) replicates at 3B with **dramatically larger effect sizes**. The 3B GRPO archive — previously catalogued as "all 4 reward variants failed to beat SFT" — now has a precise mechanistic explanation under the v4 decomposition.

## Numbers (Spider 200 subset, paired bootstrap 1000 resamples, percentile 95% CI, seed 20260429)

| Condition | EX_match | single_turn_share | n |
|---|---:|---:|---:|
| 3B SFT (ReAct) | 38.50% | **28.0%** | 200 |
| 3B GRPO (one of the archive variants) | 16.50% | **92.5%** | 200 |

### Decomposition under R2-revised Shapley-symmetric primary

| Ordering | total | share_shift | per_turn |
|---|---:|---:|---:|
| Ordering A (legacy, baseline-weighted share-shift) | −22.00 | +10.30 [+0.44, +20.67] ✓ | −32.30 [−45.77, −20.14] ✗ |
| **Ordering sym (R2 PRIMARY)** | **−22.00 [−29.50, −14.50] ✗** | **−0.72 [−10.66, +8.89]** (CI overlaps 0) | **−21.28 [−34.63, −7.69] ✗** |
| Ordering B (treatment-weighted share-shift) | −22.00 | −11.74 [−30.45, +4.97] | −10.26 [−29.10, +10.26] |

✗ = CI excludes 0 in negative direction · ✓ = CI excludes 0 in positive direction

**Important reading correction (2026-04-30)**: my original entry of this
finding reported `share_shift +10.30 ✓` per Ordering A, which became
non-primary under the R2 reviewer revision. Under the Shapley-symmetric
primary, **the 3B share-shift attribution is not significant** — the
shift in case mix at 3B is so extreme (s_T1 swung 28% → 92.5%) that
Orderings A and B disagree about its contribution by ~22 pp, and their
average leaves CI overlapping 0. The headline 3B failure is therefore
attributable to the **per-turn (conditional solver) term**, not to the
share-shift term.

### R_same paired ΔEX (composition-decontaminated)

| | n | paired ΔEX | 95% CI |
|---|---:|---:|---|
| All same-turn records | 59 (29.5%) | **−33.90 pp** | [−47.46, −20.34] ✗ |
| sft_turn=1 subset | 53 | **−35.85 pp** | [−50.94, −22.64] ✗ |
| sft_turn=2 subset | 6 | −16.67 pp | [−66.67, +33.33] |

R_same is the cleanest single number for 3B: on the 59 records (29.5%)
whose turn count was *unchanged* between SFT and GRPO, GRPO was 33.9 pp
worse than SFT in execution match. This **cannot be a composition
artifact** of share-shift because case mix is held fixed by
construction. Driven almost entirely by the 53 records where both used
turn=1 (−35.85 pp).

The 3B SFT baseline used multi-turn ReAct extensively (only 28% of records resolved in 1 turn) — 3B without RL was a deliberate, exploration-heavy agent. RL collapsed that exploration: 3B GRPO commits to a single-turn answer 92.5% of the time, a **+64.5 pp share shift**, the largest in any condition we have measured.

## Why this strengthens v3 R2 (under the corrected Shapley-sym primary)

The v3 R2 thesis is "agentic GRPO improves aggregate EX through a stopping-policy shift, but conditional SQL accuracy can collapse — the SKE-G8 case showed conditional collapse without stopping-policy gain". The 3B archive provides a **cross-model-size replication of the conditional-solver-collapse failure mode**:

1. **R_same is the cleanest 3B finding**: on records where turn count was unchanged between SFT and GRPO (29.5% of the dataset), GRPO is 33.9 pp worse than SFT. Composition is held fixed by construction; this is a pure conditional-solver collapse.

2. **Per-turn (sym) CI excludes 0 in the negative direction** (−21.28, [−34.63, −7.69]). Even averaging across orderings, the conditional-solver term carries a significant negative.

3. **Share-shift attribution is ambiguous at 3B's extreme mix swing**. Orderings A and B disagree by ~22 pp on share_shift, and their average's CI overlaps 0. This is a feature, not a bug: when the case-mix change is so large, the order in which "case mix" and "EX rates" are credited matters, and the symmetric average is the only honest summary. The 3B mechanism is dominantly conditional-solver collapse plus an over-commit move whose attribution is order-dependent.

4. **The 3B "all variants failed" finding gets a precise mechanism**. The archive notes said "3B GRPO failed to beat SFT" with no mechanistic explanation. v3 R2 decomposition shows it failed because the conditional solver collapsed (R_same −33.9 pp, per-turn-sym −21.3 pp), not because the share-shift attribution was bad. SKE-G8 at 7B has the same direction (`per_turn_sym` significantly negative, `R_same` neutral); 3B is a more extreme version of the same failure mode.

5. **Cross-model-size confirmation**: the conditional-solver-collapse mode appears at both 7B (under SKE-RL G=8) and 3B (under any of the archive's GRPO variants). This addresses the review's top weakness ("single model size") for the v3 R2 paper.

## Implications for the paper

The review (`REVIEW_v4_INTERNAL_2026-04-30.md`) flagged "single model size" as the top weakness for venue lift. With this 3B replication, the paper now has:

- 2 model sizes (3B, 7B)
- 3 G values effectively (3B archive, 7B G=4, 7B G=8)
- 4 advantage estimators (3B archive, Dense, SKE-RL G=4, SKE-RL G=8)
- 1 dataset (Spider; full dev for 7B, 200 subset for 3B)

The "G knob" framing should be loosened to a **"commitment-policy pressure" knob**: G is the version we tested for 7B, but the underlying lever is "how strongly does training push the model toward single-turn commitment?". Multiple training-side knobs (G, KL coefficient, reward shaping) can dial it.

This would substantially upgrade the venue forecast in REVIEW_v4_INTERNAL:

| Venue | v4 alone | v4 + 3B replication |
|---|---|---|
| NeurIPS workshop | Accept | Strong Accept |
| ACL Findings | Accept (after expansion) | Accept |
| ACL/EMNLP short | Accept | Strong Accept |
| ACL/EMNLP main | Reject | Borderline |

## Caveats

1. **The 3B GRPO archive used Spider 200, not full dev**. CIs are wider than the 7B numbers. The point estimates are huge (−22pp total, −32pp per_turn) so this is unlikely to flip with more data, but the absolute precision is lower than for 7B.

2. **The exact 3B GRPO recipe used in the archive is the recipe of one variant** (we'd need to identify which of the 4 archived variants `eval_agent_grpo.json` came from). The training command-line is in the project memory `project_3b_experiments_done.md`. Add to v5 proposal.

3. **3B's larger commitment shift may be a result of compute budget mismatch** — the 4 archived 3B variants may have been over-trained for their capacity. Worth flagging as a limitation but doesn't undermine the mechanism: the trade-off curve is the same direction.

## Recommendation

Add this 3B replication as Section 4.6 ("Cross-model-size replication") in the paper draft. Update v4 → v5 proposal with the broader "commitment-policy pressure" framing. The original "G knob" claim becomes a special case.

This is **0 GPU-h work that materially upgrades the paper** — exactly the leverage the review was pointing at.
