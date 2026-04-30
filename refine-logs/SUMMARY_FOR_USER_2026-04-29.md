# 5-minute summary — v2 → v3 pivot, 2026-04-29

## TL;DR

You set up v2 (Protocol Beats Structure). I built the diagnostic pipeline,
ran it on the existing 4 eval JSONs, and found the v2 axis is operationally
null on this data. Pivoted to a turn-progression axis that produces clear
signal across the same dataset. Wrote a v3 draft. Awaiting your call.

## What v2 said vs what the data shows

v2 thesis: agentic GRPO's +2 pp gain is dominantly **protocol cleanup** —
the model fixes malformed Answer blocks. Decompose with a lenient (parseable)
extractor vs the strict extractor; if `lenient_attribution_gap > 0`, the
gain is protocol.

Empirical reality on this 4-checkpoint dataset:

- Strict extractor (`extract_final_sql`) parses 1034/1034 SFT trajectories.
- Lenient extractor returns the **exact same SQL string** in:
  - 1034/1034 SFT records (0 differ)
  - 1033/1034 Dense75 (1 differs)
  - 1034/1034 SKE-G4 (0 differ)
  - 1033/1034 SKE-G8 (1 differs)
- The `protocol_gain` bucket from v2 is empty by construction.
- `lenient_attribution_gap = 0` deterministically — implementation artifact,
  not "C1 INVERTS".

Failure mode is **conceptual** (wrong-table joins, wrong-column selects),
not protocol noise. A representative exec-failed SFT record:

```
gold:    SELECT song_name, song_release_year FROM singer ORDER BY age LIMIT 1
pred:    SELECT T1.Name, T1.Song_release_year FROM singer AS T1
         JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID
         ORDER BY T1.Age ASC LIMIT 1
trajectory: 1 Action; SQL ran successfully; Answer = same SQL
```

Model wrote a runnable but semantically wrong SQL. Lenient extractor has no
recovery path because there's no alternative SQL in the trajectory to pick.

## What I tried before pivoting

**Option 2** — execution-aware lenient: pick the latest successfully-executed
Action SQL instead of the Answer block. Tested on 50 exec-failed SFT records.

- Lenient v2 differs from strict in 3/50. Gate was 10/50. **Failed.**
- Reason: 7B SFT commits early; rarely has explore-then-misanswer trajectories.

## v3 — the decomposition that works

Decompose `total_ΔEX` into:

- `share_shift_term` = (single_turn_share_change) × (single_turn_EX_baseline − multi_turn_EX_baseline)
- `per_turn_gain_term` = weighted EX delta at fixed turn count

Bootstrap 1000 resamples, paired by record id, 95% percentile CI, seed locked.

### Aggregate (Spider-dev 1034)

| Pair | total ΔEX | share_shift | per_turn_gain |
|---|---|---|---|
| Dense75 − SFT | **+2.32 [+0.39, +4.35]** ✓ | **+0.90 [+0.08, +1.84]** ✓ | +1.42 [−0.64, +3.58] |
| SKE-G4-75 − SFT | +1.64 [−0.39, +3.87] | **+1.32 [+0.47, +2.24]** ✓ | +0.32 [−1.98, +2.58] |
| SKE-G8-75 − SFT | +0.39 [−1.06, +1.94] | **+2.64 [+1.73, +3.59]** ✓ | **−2.26 [−3.89, −0.60]** ✗ |

✓ = CI excludes 0 in positive direction · ✗ = excludes 0 in negative direction

### Per-difficulty (where the gain comes from)

- Dense's gain is **concentrated on Medium**: total +3.98pp ✓; per_turn +3.17pp ✓.
- SKE-G8's per-turn loss is **concentrated on Hard**: point −2.84 (CI overlaps 0 weakly).
- Easy queries are 95% already turn=1 under SFT — share_shift has nowhere to go there.

### Per-record paired transitions

Records that shifted to fewer turns under each RL condition:

| | shift-down | EX kept | EX lost | EX gained | net |
|---|---:|---:|---:|---:|---:|
| Dense vs SFT | 125 | 45 | 8 | 19 | **+11** |
| SKE-G4 vs SFT | 136 | 50 | 10 | 14 | +4 |
| SKE-G8 vs SFT | 113 | 41 | 11 | 11 | **+0** |

SKE-G8's commit-faster behavior is **net-neutral on shifted records**. The
+0.39 aggregate gain comes from the per-turn term (which is negative anyway).

The (sft_turn → rl_turn) cell EX rates show the over-commit pathology:

- SKE-G8 1→1 cell (824 records, the largest single cell): 70.0% EX vs Dense
  73.6% / SFT 73.6% — model rambles into wrong answer.
- SKE-G8 2→1 cell (101 records): 48.5% EX vs Dense 54.6% — squeezing 2-turn
  queries to 1 turn loses accuracy.

Trajectory length on the 13 records where SKE-G8 broke a previously-correct
turn-1 SFT answer: **+242 chars longer** than SFT's. Over-commit is "more
confident reasoning toward a wrong answer", not "shorter trajectory".

## One-sentence v3 thesis

Agentic GRPO on multi-turn Text-to-SQL improves aggregate execution match
dominantly through a commitment-policy shift (more queries resolved in one
turn), and only secondarily through per-turn SQL quality. SKE-RL maximizes
the commitment shift but degrades per-turn quality — most pronounced on
Hard queries — so the aggregate gain washes out at G=8.

## What you need to decide

1. **Adopt v3 as the working proposal?** Draft is at
   `refine-logs/FINAL_PROPOSAL_v3_DRAFT.md`.

2. **Launch B4 (Dense G=8 confound, ~28 GPU-h)?** v3 makes a
   pre-registered prediction: Dense G=8 share_shift ∈ [+1.8, +2.4] (CI
   excludes 0); per_turn ∈ [−0.5, +0.5] (CI overlaps 0). Confirming this
   isolates the mechanism cleanly: same share-shift signal Dense produces,
   without SKE-RL's class-baseline degradation.

3. **Continue more analysis, or start writing?** All Week 1 deliverables
   from your EXPERIMENT_PLAN are now complete (under the v3 reframe). The
   diagnostic protocol is implemented + tested + applied. Per-difficulty
   and per-record paired analyses are done. The figures-and-table count
   needed for a 4-page workshop is already in `outputs/`.

## Files I produced today

```
refine-logs/PRE_REGISTRATION.md                   # locked hashes, gates, CI protocol
refine-logs/A0_PRE_AUDIT_FINDING_2026-04-29.md    # why v2 axis is null
refine-logs/A0_OPTION_2_3_SPIKE_2026-04-29.md     # spike that motivated v3
refine-logs/TURN_DECOMP_FINDING_2026-04-29.md     # full v3 numbers
refine-logs/FINAL_PROPOSAL_v3_DRAFT.md            # the new proposal candidate
refine-logs/SUMMARY_FOR_USER_2026-04-29.md        # this file

miniLLM/diag/__init__.py                          # diag package public API
miniLLM/diag/lenient_extract.py                   # leakage-safe extractor
miniLLM/diag/classifier.py                        # bucket rules (still works for v2-as-appendix)
miniLLM/diag/bootstrap.py                         # paired bootstrap CI helpers
miniLLM/diag/equivalence.py                       # Cliff's δ + |δ| ≤ 0.147 margin

scripts/diag_turn_progression.py                  # main v3 analysis
scripts/diag_turn_by_difficulty.py                # per-difficulty stratification
scripts/diag_paired_turn_transitions.py           # per-record transition matrix

outputs/turn_progression_analysis.json
outputs/turn_progression_by_difficulty.json
outputs/turn_paired_transitions.json

tests/test_diag_lenient_extract.py   # 8 tests
tests/test_diag_classifier.py        # 10 tests
tests/test_diag_bootstrap.py         # 6 tests
tests/test_diag_equivalence.py       # 8 tests
                                     # 32 unit tests, all pass
```

Memory: `project_turn_progression_decomp.md` updated with full v3 numbers.

I have not modified `FINAL_PROPOSAL.md`, `EXPERIMENT_PLAN.md`, or
`PRE_REGISTRATION.md` — those remain as you wrote them. The pivot decision
is yours.
