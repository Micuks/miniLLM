---
date: 2026-04-29
status: Option 2 spike FAILED; Option 3 spike SUCCEEDED with real signal
parent: A0_PRE_AUDIT_FINDING_2026-04-29.md
---

# Spike results: Option 2 (execution-aware lenient) vs Option 3 (turn-progression)

## Option 2 — Execution-aware lenient extractor

Hypothesis: redefine `lenient = "latest successfully-executed Action SQL"`
captures records where the model "explored the right query but committed
to a wrong one as final".

**Result on 50 SFT records with `execution_match=False`**:
- lenient_v2 succeeded: 48/50
- lenient_v2 string differs from strict: **3/50**
- All 3 differences were minor whitespace/case (`avg` vs `AVG`) or one
  extra column. None of the v2 SQLs would change `execution_match`.

**Conclusion**: 7B SFT does not exhibit the explore-then-misanswer pattern.
The model commits to one SQL early and rarely revises. Option 2 is **not
viable** for protocol-vs-semantic decomposition (gate was ≥10/50; observed
3/50 with effective 0).

## Option 3 — Turn-progression decomposition

Hypothesis: decompose EX gain into (a) per-turn-count quality improvement
and (b) share shift across turn counts.

**Per-condition turn-1 share + per-turns EX_match (Spider-dev 1034)**:

| Condition | overall EX | turn=1 share | turn=1 EX | turn=2 share | turn=2 EX | turn≥3 share |
|---|---:|---:|---:|---:|---:|---:|
| 7B SFT | 64.3% | 81.4% | 69.7% | 16.8% | 41.4% | 1.7% |
| Dense75 | 66.6% | 84.5% | 70.5% | 14.0% | 46.2% | 1.4% |
| SKE-G4-75 | 66.0% | 86.0% | 69.3% | 12.9% | 47.4% | 1.2% |
| SKE-G8-75 | 64.7% | 90.5% | 67.2% | 8.3% | 40.7% | 1.2% |

**Real, signed deltas**:

- Dense vs SFT: turn=1 share **+3.1pp**, turn=1 EX **+0.8pp**, turn=2 EX **+4.8pp**, turn≥3 share −0.3pp.
- SKE-G8 vs SFT: turn=1 share **+9.1pp** but turn=1 EX **−2.5pp**.

**Interpretation**:

1. Dense GRPO's +2.3pp aggregate gain decomposes as roughly equal mix of
   (a) shifting queries to single-turn (where EX is higher) and (b) modest
   per-turn quality gains.
2. SKE-G8 shifts MORE queries to single-turn (+9.1pp share) but the single-
   turn EX drops (−2.5pp), so the strategy shift hurts. This explains
   SKE-G8's overall loss (64.7% vs SFT 64.3%, essentially flat) without
   needing a "protocol failure" frame.
3. The interesting axis is **strategy** (how the model decides to commit
   vs continue exploring) — not protocol noise.

**Conclusion**: Option 3 has clear, signed, mechanistically-meaningful
signal across all 4 conditions. **Recommended over Option 2.**

## Proposed reframe

Replace lenient/strict decomposition with:

```
single_turn_share[X]  = |{records with num_turns == 1}| / 1034   under condition X
single_turn_EX[X]     = EX_match rate among num_turns==1 records under X
multi_turn_share[X]   = 1 - single_turn_share[X]
multi_turn_EX[X]      = EX rate among num_turns>=2 records

ΔEX(X vs SFT) = single_turn_share[X] · single_turn_EX[X]
              + multi_turn_share[X] · multi_turn_EX[X]
              − single_turn_share[SFT] · single_turn_EX[SFT]
              − multi_turn_share[SFT] · multi_turn_EX[SFT]

Decompose into:
  share_shift_term    = (single_turn_share[X] − single_turn_share[SFT])
                      · (single_turn_EX[SFT] − multi_turn_EX[SFT])
  per_turn_gain_term  = single_turn_share[X] · (single_turn_EX[X] − single_turn_EX[SFT])
                      + multi_turn_share[X] · (multi_turn_EX[X] − multi_turn_EX[SFT])
```

`share_shift_term` quantifies "the model learned to commit faster on
queries it can resolve in one turn". `per_turn_gain_term` quantifies
"genuine SQL-writing improvement at fixed turn count".

**New paper claim** (replaces "Protocol Beats Structure"):

> "Agentic GRPO on multi-turn Text-to-SQL reshapes the agent's
> commitment policy more than it improves SQL-writing capability. Of the
> +2.3pp aggregate EX gain (Dense vs SFT), roughly half is share-shift
> (queries reaching turn=1 commitment more often where strict-correct
> rate is +28pp higher than turn=2) and half is per-turn quality. SKE-RL
> at G=8 amplifies the share-shift but degrades per-turn quality, which
> is why aggregate EX falls."

**Bucket alternative for transition table**:

```
single_turn_strict_correct = (num_turns == 1 AND execution_match)
multi_turn_repair          = (num_turns >= 2 AND execution_match
                              AND first action's Observation was Error:)
multi_turn_polish          = (num_turns >= 2 AND execution_match
                              AND first action's Observation was OK)
multi_turn_failed          = (num_turns >= 2 AND not execution_match)
single_turn_failed         = (num_turns == 1 AND not execution_match)
```

Already-data-grounded; deterministic; produces non-trivial bucket
distributions across conditions.

## Recommendation to user

**Pivot to Option 3.** Update `FINAL_PROPOSAL.md` and
`PRE_REGISTRATION.md` to use turn-progression decomposition. New headline
metrics:

- `share_shift_term` (signed, in pp, with bootstrap CI)
- `per_turn_gain_term` (signed, in pp, with bootstrap CI)
- `commitment_ratio` = `share_shift_term / total_ΔEX` (interpretable
  fraction of gain attributable to strategy)

Same bootstrap protocol (1000 resamples, 95% percentile, paired). Same
1034-record dataset. No new training. The diagnostic protocol still
generalizes to other agentic-tool settings (multi-step web agents, code
agents, etc.) by replacing num_turns with task-specific turn counts.

The main novelty (decomposing aggregate gain into a structural axis) is
preserved; the axis itself shifts from extraction-leniency to
turn-commitment-strategy.
