---
date: 2026-04-30
reviewer: gpt-5.4 (xhigh)
thread_id: 019dd931-50bd-7511-94d8-6ce507348142 (round 3)
verdict: NOT main-conf-ready as-is; concrete path to 7/10 plausibility exists
---

# Main-Conference Plausibility — concrete path

## Bottom line

- **As-is**: not enough for main conference (4/10). 1 model × 1 dataset × 1 train-seed = workshop note.
- **Minimum credible main-conf scope**: `2 datasets × 2 model families × (1 SFT + 3 RL × 3 seeds) = 40 rows` (with intervention) or **28 rows** (pure diagnostic, only if scope is wider on 2nd axis).
- **Optimal 3-GPU-week plan**: multi-seed (Spider) + intervention + BIRD + zero-GPU robustness ≈ 432–462 GPU-h. Lifts to **7/10 main-conf plausibility**.

## Reframe required

The "commitment policy" thesis is local. Main-conf-grade reframe:

> **In tool-augmented Text-to-SQL, RL first learns a `stopping policy`, not a `conditional solver`; this bias is diagnosable and partially fixable by a stop-aware intervention.**

Rename axis: `commitment-policy shift` → `stopping policy vs conditional solver` (or `answer-timing shift vs conditional SQL correctness`). More general, more "main-conf shaped".

## Three deltas separating workshop note from main-conf

1. **Internal validity** — multi-seed (3+) replication of every headline number.
2. **External validity** — at least 2 datasets (Spider + BIRD) and 2 model families (Qwen + one of Llama/Mistral/DeepSeek 7B-8B).
3. **Actionability** — at least one intervention derived from the diagnostic that *does something* (preserves share-shift while reducing per-turn regression on Hard).

A diagnostic-only paper without an intervention is workshop-bound unless empirical scope is dramatically wider than current budget allows.

## The recommended intervention (cheapest viable)

**Stop-Calibrated Reward** (a.k.a. Commitment-Calibrated Reward):

```
R_stop-cal = R_EX + λ · 1[turn=1 ∧ EX=1] − λ · 1[turn=1 ∧ EX=0]
```

- 1 pilot (λ ∈ {0.05, 0.1}) to pick best λ
- Then 3 Spider seeds at chosen λ
- Claim form: "preserves positive share_shift, reduces negative per_turn_sym on Hard"

**Zero-training-cost supporting ablation**: inference-time gate — if turn=1 wants to Answer but no successfully-executed Action SQL exists earlier, force one more turn. If this gate alone improves SKE-G8, mechanism story is reinforced.

## The recommended formalization (one page only)

Treat agent as episodic policy with stopping action `Answer` vs `Continue`. Define stopping time `τ`. Decompose:

```
V(π) = Σ_t P_π(τ=t) · q_π(t)
```

This upgrades the decomposition from "ad-hoc accounting" to **stopping-marginal × conditional-value factorization**. Useful, but not the main sell — keep to one page in §3.

**Don't** invest in a theorem paper; main-conf reviewers won't reward it.

## Concrete 3-GPU-week plan

| Block | Content | GPU-h | Unlocks |
|---|---|---:|---|
| A | B4 + Spider Dense G=8 / SKE G=8 each to 3 seeds | 140 | not-single-seed-fluke; HARK pressure relieved |
| B | Stop-calibrated intervention: 1 pilot + 3 Spider seeds | 112 | diagnostic → actionable fix |
| C | BIRD: SFT + Dense G=8 + SKE G=8 + intervention, 2 seeds each | 180–200 | not-Spider-only; cross-benchmark |
| D | Spider-Realistic eval + R_same + difficulty×turn + paired-ΔEX + ckpt sensitivity | 0–10 | robustness completeness |
| **Total** | | **432–462** | **lifts to 7/10 main-conf** |

(On 8×3090 with parallelism ≈ 2.5–3 wall-days for B4-style 28 GPU-h runs; total ~2.5–3 weeks compute calendar.)

## Highest-EV experiments, ranked

| Rank | Experiment | GPU-h | Risk reduced | Claim unlocked |
|---:|---|---:|---|---|
| 1 | Spider Dense G=8 + SKE G=8 to 3 seeds (incl. B4) | 140 | single-seed / HARK / variance | share_shift and negative per_turn_sym are stable phenomena |
| 2 | Stop-calibrated reward on Spider | 112 | "analysis but no method" | mechanism is interventionable; Hard improves |
| 3 | Port to BIRD: same conditions, 2 seeds | 180–200 | Spider-specificity | phenomenon and fix transfer cross-benchmark |
| 4 | Spider-Realistic + zero-GPU robustness pack | 0–10 | artifact / decomposition-order / composition | decomposition story is not statistical artifact |
| 5 | Second 7B/8B family Spider pilot | 100–200 | Qwen-only | not-single-backbone phenomenon |

**Explicitly NOT worth doing**:
- 14B GRPO (3090s can't carry it cleanly)
- Re-running 3B reward zoo
- Wide checkpoint grids without seed coverage
- EHRSQL before BIRD

## Venue routing

| Venue | Status | Verdict |
|---|---|---|
| NeurIPS 2026 main | abs/full deadline May 4–6, 2026 | too soon, skip |
| ACL 2026 / COLM 2026 | deadlines passed | skip |
| EMNLP 2026 | ARR May 25, 2026 | technically open but bad fit + too rushed |
| **ICLR 2027** | dates not posted; 2026 was Sep 19/24, expect ~Sep 2026 | **PRIMARY TARGET** |
| **NeurIPS 2026 workshop** (TRL / RL-for-LLMs) | deadlines ~Aug 2026 | **FALLBACK** |
| ICML 2027 | Jan 2027 deadline | secondary fallback |

Reasoning: ICLR 2027 ~Sep 2026 deadline gives clean window to execute A/B/C/D over summer. NeurIPS workshop is the soft-landing if seed/intervention results disappoint.

## Kill switches — abandon main-conf attempt and write workshop note if any of:

1. After Spider multi-seed: SKE G=8 `per_turn_sym` no longer stably negative (CI crosses 0). → core phenomenon was variance-driven.
2. B4 shows Dense G=8 has significant negative `per_turn_sym` AND stop-calibrated intervention does not fix it. → degenerates to "generic large-G tradeoff", losing SKE-specific edge.
3. On BIRD: share_shift-dominant pattern disappears, OR intervention has no effect. → no cross-dataset generality, no actionability.

**B4 alone does NOT kill the paper.** Main-conf killer is: B4 negative + multi-seed unstable + intervention fails.

## Competitive gap (the exact hole this paper plugs)

Closest related work is NOT CodeS/DTS-SQL (those are pre-RL/SFT-era). It is the **agentic-RL Text-to-SQL line**:

| Paper | What they do | What they don't do | Your gap |
|---|---|---|---|
| SQL-R1 ([arXiv:2504.08600](https://arxiv.org/abs/2504.08600)) | RL reward, cold start, Spider/BIRD aggregate gain | no interaction-depth / turn-distribution analysis | you separate "answer-earlier" from "answer-better" |
| ReEx-SQL ([arXiv:2505.12768](https://arxiv.org/abs/2505.12768)) | exec-aware decoding, tree decoding, Spider/BIRD/Spider-Realistic | doesn't verify gains aren't from aggressive stopping | you test whether self-correction actually improves conditional SQL quality |
| MTIR-SQL ([arXiv:2510.25510](https://arxiv.org/abs/2510.25510)) | multi-turn tool-integrated RL, trajectory filtering, KL-free | no paired per-record stop/quality decomposition | you give turn-conditioned mechanism diagnosis |
| MARS-SQL ([arXiv:2511.01008](https://arxiv.org/abs/2511.01008)) | multi-agent + validator agent + SOTA aggregate EX | doesn't distinguish solver-improvement from trajectory-selection/timing | you decompose monolithic agentic gain |
| HES-SQL ([arXiv:2510.08896](https://arxiv.org/abs/2510.08896)) | skeleton + latency reward, error-type breakdown | no stop-time / answer-timing policy analysis | you fill the missing axis of RL-agent behavior |

**One-sentence positioning**: 他们都在优化"最终答对率"，你在证明"RL 到底先学会了什么"。

In English for the paper: "Existing agentic-RL Text-to-SQL papers report end-to-end EX improvements and frame interactive self-correction as a monolithic capability; we are the first to decompose RL-induced gains into stopping-policy shift and per-turn conditional quality, and to show that the dominant axis is the former."

## Rating after the recommended plan executes

| Dimension | Current (4/10 panel) | After plan |
|---|---:|---:|
| Logical soundness of decomposition | 6 | 8 |
| Empirical strength | 5 | 7 |
| Workshop readiness | 6 | 9 |
| **Main-conf plausibility** | **4** | **7** |
| Overall | 6 | 7 |

7/10 is "borderline accept territory at ICLR-tier", not guaranteed. To push 7→8, would need a 3rd model family (e.g., DeepSeek-7B in addition to Qwen+Llama) or scaling-curve evidence — but those add another 1+ GPU-week and are bad ROI on 3090s.

## Decision tree

```
Today
  └── Apply zero-GPU robustness fixes (Shapley-sym, R_same, paired-ΔEX) → ~1 day
  └── Commit B4 prereg → < 1 hour
  └── Launch B4 + Spider Dense G=8 seed-2 (parallelize) → ~2 days

Week 1-2 (≈ 6 GPU-days)
  └── Spider multi-seed (Block A): Dense G=8 ×3 + SKE G=8 ×3 → ~140 GPU-h
  └── Pilot stop-calibrated reward (λ sweep) → 1-2 days

Week 2-3 (≈ 6 GPU-days)
  └── BIRD ports (Block C) + intervention seeds (Block B) → 280-310 GPU-h

Week 3-4 (writing)
  └── Spider-Realistic eval (Block D, near-zero GPU)
  └── Draft toward ICLR 2027 (Sep 2026 deadline)

Decision points
  ├── End of Week 1: if Spider multi-seed instability → kill main-conf attempt
  ├── End of Week 2: if intervention fails on Spider → demote to workshop
  └── End of Week 3: if BIRD shows no transfer → demote to workshop
```

## What I (Claude) recommend you do TODAY

1. **DO** the zero-GPU robustness fixes (Shapley-sym + R_same + paired-ΔEX + difficulty×turn). Hours of CPU work.
2. **DO** commit B4 prereg verbatim into git BEFORE any training.
3. **DO** launch B4 (28 GPU-h, ≈ 1.5 days).
4. **DON'T** start writing the paper toward main-conf yet — premature. Wait for B4 + multi-seed signal.
5. **DON'T** touch FINAL_PROPOSAL.md / EXPERIMENT_PLAN.md / PRE_REGISTRATION.md without explicit user authorization (per existing user instruction).

Trace: `.aris/traces/research-review/2026-04-29_run01/round3_*`. Resumable thread: `019dd931-50bd-7511-94d8-6ce507348142`.
