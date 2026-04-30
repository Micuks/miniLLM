---
date: 2026-04-29
status: blocking finding before A0 audit
parent: PRE_REGISTRATION.md, FINAL_PROPOSAL.md
---

# Pre-A0 finding: lenient extractor ≡ strict on this dataset

Before running the manual A0 audit, I ran the lenient extractor against all 4
input eval JSONs and recorded how often the lenient SQL string differs from
the strict SQL string already stored in `pred_sql`.

## Measurements

| Eval JSON | Records | lenient succeeded | lenient differs from strict |
|---|---:|---:|---:|
| eval_full_dev_sft.json (SFT) | 1034 | 1034 (100.0%) | **0** (0.0%) |
| eval_full_dev_dense75.json | 1034 | 1034 (100.0%) | **1** (0.1%) |
| eval_full_dev_ske75.json | 1034 | 1034 (100.0%) | **0** (0.0%) |
| eval_full_dev_ske_g8_ck75.json | 1034 | 1034 (100.0%) | **1** (0.1%) |

Strict-extract failure (parse-fail or empty) on SFT is **0/1034** — the
existing `extract_final_sql()` is already perfect on every record.

## Diagnostic protocol implication

The decomposition rests on `final_correct_strict ≠ final_correct_lenient`.
Across these JSONs, strict and lenient produce the **same SQL string** in
≥99.9% of records, so:

- `final_correct_strict ≡ final_correct_lenient` for every paired comparison
- `protocol_gain` bucket = `(¬strict_correct ∧ lenient_correct ∧ rl.strict_correct)` ≈ ∅
- `semantic_gain` bucket = full strict-gain transition set
- `lenient_attribution_gap = ΔEX_strict − ΔEX_lenient ≈ 0` deterministically

This does not represent C1 INVERT in the meaningful "semantic-dominant"
sense the proposal contemplates. It is an **implementation artifact**: the
lenient extractor as specified does the same thing strict already does on
well-formed trajectories, so the protocol-vs-semantic axis collapses by
construction, not by data.

## Mechanism

`extract_final_sql()` (the existing strict path, `agent/react.py:224`):

```python
if parsed.final_answer:
    return _clean_sql_candidate(parsed.final_answer)
for step in reversed(parsed.steps):
    if step.action_sql:
        return _clean_sql_candidate(step.action_sql)
return None
```

`lenient_extract` (new, `diag/lenient_extract.py`):
1. First parseable SQL in the `Answer:` region.
2. Else, last parseable SQL among `Action: execute_sql[...]` payloads.

Both paths land on the same Answer-block content because the 7B SFT model
always emits a well-formed Answer block. `_clean_sql_candidate` does
aggressive post-processing (strip wrappers, drop trailing markers) but on
the inputs at hand it returns exactly the same string sqlglot would parse.

## Failure-mode evidence (single representative record)

Sample exec-failed record (concert_singer / medium):

```
gold_sql:    SELECT song_name, song_release_year FROM singer ORDER BY age LIMIT 1
pred_sql:    SELECT T1.Name, T1.Song_release_year FROM singer AS T1 JOIN
             singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID
             ORDER BY T1.Age ASC LIMIT 1
trajectory:  one Action whose SQL == pred_sql, executed successfully
             (returned "Adele | 2011"), then Answer == same SQL
```

The model wrote a runnable SQL that is **semantically wrong** (joined
`singer_in_concert` instead of selecting the right column from `singer`).
There is no earlier-action / later-action divergence for lenient extraction
to recover. The failure is purely conceptual.

## Implication for the paper as proposed

The "Protocol Beats Structure" thesis posits that aggregate EX gains are
dominantly protocol-cleanup. On a well-trained 7B SFT this is **empirically
false** — but not for the reason the FINAL_PROPOSAL anticipated. Failures
are dominantly conceptual; there is no protocol axis to decompose against
on this dataset.

A2's `lenient_attribution_gap` headline number cannot carry the paper.

## Branching options (await user decision)

### Option 1 — Reframe to "no protocol axis at 7B SFT"
Treat the empirical absence of a protocol-vs-semantic axis as the central
finding. Claim: agentic SFT at 7B saturates the protocol layer; subsequent
RL gains must be semantic by elimination. SKE-RL's negative result is then
secondary because it operates on the (saturated) protocol layer — not what
the original proposal said but consistent with its data.
- **Pros**: requires no extractor redesign; uses what we already see.
- **Cons**: weakens C1 from a measured decomposition into a definitional
  argument; less reviewer-novel.

### Option 2 — Redefine lenient extractor to be execution-aware
Lenient = "the latest SQL action whose first execution succeeded (no SQL
error), regardless of whether the model later replaced it with a different
Answer". Captures the pattern "model executed the right query but locked in
a different one as its answer". Test: on the 369 SFT records with
`execution_match=False`, does this lenient definition find a SQL that
*does* match gold? If yes → real protocol-gain bucket exists, just at a
different definition than the proposal originally specified. If no → 7B
trajectories still don't have explore-then-misanswer pattern.
- **Pros**: keeps the decomposition story; gives a real chance at signal.
- **Cons**: requires re-running A0 with the new extractor; the bucket
  semantics shifts ("protocol" now means "model knew the answer in an
  earlier action but committed to a wrong one").

### Option 3 — Different decomposition axis entirely
Replace lenient/strict with a turn-progression decomposition:
`turn1_correct` vs `final_correct`. Did the model fix-after-observation? Is
the gain in single-turn accuracy or multi-turn repair behavior? This is a
genuine compositional axis that *will* differ between SFT and Dense75
because the trajectories use different numbers of turns under different
conditions.
- **Pros**: real signal almost guaranteed; the existing `num_turns` field
  is right there.
- **Cons**: bigger reframe; requires updating FINAL_PROPOSAL §Core
  Mechanism; might or might not align with the SKE-RL secondary claim.

## Recommendation

**Option 2 with a 30-minute spike to confirm**: redefine lenient as
"latest successfully-executed SQL action OR Answer SQL", run on 50 SFT
records with `execution_match=False`, count records where lenient finds a
gold-matching SQL that strict doesn't. If ≥10 such records, proceed with
the redefinition. If 0–5, switch to Option 3.

Awaiting decision before proceeding to the manual audit (A0).
