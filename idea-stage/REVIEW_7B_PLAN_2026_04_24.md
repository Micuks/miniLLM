---
date: 2026-04-24
status: SUPERSEDED — initial metric comparison was wrong; see correction below
---

# Review of 7B Plan — with Metric Correction

## Metric correction (IMPORTANT)

The initial reviewer critique (v1 of this doc, now archived) compared 7B SFT `EX_all` (52.5%) to what I incorrectly labeled 3B SFT `57%`. In fact:

- 3B SFT v1 reports `execution_match = 0.5703 = 77/135` — that is the **EX_attempted** ratio (num correct / num attempted), because 3B only attempted SQL on 135/200 samples (67.5% attempt rate).
- 3B SFT `EX_all` (num correct / 200) is actually **77/200 = 38.5%**.
- 7B SFT v1 reports `EX_attempted = 0.6213 = 105/169` and `EX_all = 105/200 = 52.5%`; attempt rate 84.5%.

Apples-to-apples (both on EX_attempted):

| Difficulty | 3B SFT | 7B SFT | Δ |
|---|---|---|---|
| Easy | 73.7% | 88.5% | **+14.8** |
| Medium | 51.4% | 56.6% | **+5.2** |
| Hard | 39.5% | 38.2% | -1.3 |
| **Overall** | **57.0%** | **62.1%** | **+5.1** |
| **Attempt rate** | 67.5% | 84.5% | **+17.0** |

On EX_all the delta is even larger: 7B SFT 52.5% vs 3B SFT 38.5% → **+14.0pp**.

## Revised verdict

Capacity hypothesis is **partially supported** — scale lifts Easy (+14.8) and Medium (+5.2) queries robustly; Hard is tied. Attempt rate jumps 17pp, meaning 7B is also more reliable at producing SQL at all (a separate capability axis).

The key remaining research question is:

> **Does 7B GRPO lift Medium above 7B SFT 56.6%**, where 3B GRPO regressed Medium by 18-28pp across all 4 reward shapes?

If yes → capacity + RL story (strong narrative, Findings-plausible, possibly main if paired with good diagnostic).

If no → reward-and-scale-invariant Medium regression (diagnostic paper, Findings-viable).

## Revised experiment priorities (highest lift-per-GPU-h)

| # | Experiment | GPU-h | Rationale |
|---|---|---|---|
| D | **7B Dense GRPO** (150 steps, bnb4 QLoRA, from current SFT init) | ~10 | Core test of capacity+RL |
| D' | **7B CGFR+RVDS GRPO** (best 3B recipe replicated at 7B) | ~10 | Second reward point for H2 |
| E | Per-query-class diagnostic (tag by JOIN/GROUP/nested) | ~4 | Narrative lift per hour — highest |
| F | Trajectory error taxonomy on 100 failures | ~6 | Section 5 mechanistic evidence |
| C | Full Spider-dev (1034) eval on 3B SFT, 7B SFT, 7B GRPO | ~6 | Kills "only 200 samples" critique |
| B | 3-seed variance on 3B SFT + 3B sparse + 7B SFT + 7B GRPO | ~40 | Required for reviewers |
| H | External baseline (DIN-SQL or CoT Qwen2.5-7B single-pass) | ~8 | Required for reviewers |
| A (deferred) | 7B SFT rank=64 ablation | ~12 | Useful ablation, not blocking |

Total critical path: ~44 GPU-h (D + D' + E + F + C) = 1 week on our compute.

## Results-to-claims matrix (corrected)

| 7B GRPO Medium vs 7B SFT | Best available claim | Venue |
|---|---|---|
| +5pp or more | Scale + RL unlocks Medium; capacity hypothesis confirmed | Findings strong / main-track borderline |
| ±2pp (tied) | Reward is second-order at scale; 7B SFT is strong baseline; scale fixes Medium via SFT alone | Findings |
| -5pp or worse | Scale-invariant Medium regression in agentic GRPO (diagnostic) | Findings (diagnostic) |

With the `Overall +5.1pp` SFT lift already locked, **every row of the matrix is Findings-viable** — strictly better than the pre-correction outlook.

## Single most important NEXT action

**Launch 7B Dense GRPO pilot** (task #58, unblocked). Config: QLoRA bnb4, 100-150 steps, NUM_GEN=4 or 8, MAX_COMPLETION=384, from `outputs/react-sft-7b-bf16` adapter. Target: check Medium at ck50/ck100 vs 7B SFT 56.6%.

Second track: per-query-class diagnostic on the existing 3B archive + 7B SFT (offline, no GPU). Highest narrative lift per hour.

## Items that remain correct from v1 review

- ACL main-track unlikely given single-GPU scale and budget; Findings is realistic target
- Need external baseline (DIN-SQL or CoT Qwen2.5-7B)
- Need per-query-class diagnostic (JOIN count, GROUP BY, etc.) to make "Medium" mechanistic
- Need seed variance (3 seeds) before final submission
- Need full Spider-dev eval (1034) before final submission
