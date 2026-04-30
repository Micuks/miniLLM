---
target: ACL 2027 main / EMNLP 2026 main
date: 2026-04-18
status: planning
---

# ACL/EMNLP Submission Plan

## Target venues & timeline

- **EMNLP 2026 main**: submission deadline ~2026-06 (≈ 6-8 weeks from now)
- **ACL 2027 main**: submission deadline ~2027-01-02 (≈ 8 months from now)

Realistic target: **ACL 2027 main** (EMNLP 2026 too tight for full ablation × 3 seeds × 2 benchmarks).

## Narrative thesis (1-sentence)

> "Turn-Delta Credit Assignment + Entropy-Weighted Span GRPO + Difficulty-Adaptive Compute enable a 3B model to match 7B GRPO baselines on Text-to-SQL at 5× lower training compute, by fixing the sparse-reward problem in multi-turn ReAct RL."

## Gaps vs current state

| Requirement | Current | Target for submission |
|---|---|---|
| Benchmarks | Spider 200 | Spider-dev full (1034) + **BIRD-dev (1534) + BIRD-test**  |
| Training steps | 100 | 300-500 |
| Seeds | 1 | **3** (42, 123, 456), report mean ± std |
| Ablation | v5-all only | **5 configs**: v4, v5-adaptive, v5-span, v5-turn, v5-all |
| Baselines cited | SFT, v3, v4 | + **Arctic-R1-7B**, MARS-SQL, MTIR-SQL, Alpha-SQL |
| Statistical tests | none | paired t-test across seeds |

## Compute budget estimate

Per config (300 steps, G=8, ~4 min/step): **20 hours** of GPU time.

| Experiment block | Runs | Hours each | Total GPU-h |
|---|---|---|---|
| Spider: 5 configs × 3 seeds × (SFT + train) | 15 + 3 | 20 | **360** |
| BIRD: 5 configs × 3 seeds | 15 | 25 | **375** |
| SOTA reproductions (Arctic-R1-7B 3-seed) | 3 | 40 | 120 |
| Full Spider-dev eval × 15 models | 15 | 2 | 30 |
| BIRD-dev eval × 15 models | 15 | 2 | 30 |
| **Total** | | | **~900 GPU-hours** |

On RTX 6000 (24 GiB) shared with GRASS: 900h / 12h effective/day = **~75 calendar days**. Tight for EMNLP, fine for ACL.

## Experiment matrix

### Main Table 1: Spider-dev (full, 1034 samples)

| Method | Model | EX_att (easy/medium/hard/overall) |
|---|---|---|
| SFT baseline | 3B | ... |
| GRPO v4 (paper's "no-enhancement" baseline) | 3B | ... |
| **v5-adaptive** (ours) | 3B | ... |
| **v5-span** (ours) | 3B | ... |
| **v5-turn** (ours) | 3B | ... |
| **v5-all** (ours) | 3B | ... |
| Arctic-R1-7B (reproduced) | 7B | ... |
| Arctic-R1 (reported) | 7B | — |
| MARS-SQL (reported) | 7B | — |

### Main Table 2: BIRD-dev (full, 1534 samples)

Same layout as Table 1.

### Ablation Table 3: Feature interactions

| adaptive | span | turn | Spider EX | BIRD EX |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | (= v4) | |
| ✓ | ✗ | ✗ | | |
| ✗ | ✓ | ✗ | | |
| ✗ | ✗ | ✓ | | |
| ✓ | ✓ | ✗ | | |
| ✓ | ✗ | ✓ | | |
| ✗ | ✓ | ✓ | | |
| ✓ | ✓ | ✓ | | |

8 cells × 3 seeds = 24 runs (Spider only). Skip for BIRD to save compute.

### Analysis figures
- Fig 1: Reward curve (v4 vs v5-all) over training steps
- Fig 2: Per-difficulty accuracy breakdown
- Fig 3: Ablation stacked-bar (marginal contribution of each feature)
- Fig 4: Hard-query case study (bimodal reward pattern in v4 → variance-reduced in v5)

## Priority order (what to run first)

### Phase A (immediate, 1 week)
1. Current v5 100-step training completes (in progress)
2. **Eval full Spider-dev** on v5-all checkpoint — get preliminary "big picture" result
3. Implement BIRD loader
4. 1-seed BIRD eval on existing v4 and v5-all models (sanity check)

### Phase B (3 weeks)
5. Train v5-all at 300 steps (single seed) on Spider
6. Run 5-config ablation (single seed) at 300 steps on Spider
7. Eval all 5 on Spider-dev full + BIRD-dev full

### Phase C (4 weeks)
8. Multi-seed (3x) for the best config + v4 baseline on both benchmarks
9. Reproduce Arctic-R1-7B if code available (fallback: cite)

### Phase D (2 weeks)
10. Paper writing (use `/paper-writing` skill)
11. Internal review + revisions
12. Submit

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| GPU contention (GRASS) | Coordinate schedule or offload to vast.ai ($0.40-1/h for RTX 6000) |
| 500 GPU-h budget exceeded | Drop to 2 seeds; drop 3-way ablation cells |
| BIRD too hard for 3B (<30% EX) | Keep 3B as "efficiency" angle, add 7B as scale variant |
| Arctic-R1-7B not reproducible | Cite published numbers with caveat "from original paper" |
| ACL 2027 deadline slip | Fall back to NAACL 2027 (typically 2 months later) |

## Next actions (auto mode)

1. Let current v5 100-step finish → eval on Spider 200 (current task)
2. Start implementing BIRD loader in parallel (can run once GPU free)
3. Plan multi-seed orchestration script (task queue style)
