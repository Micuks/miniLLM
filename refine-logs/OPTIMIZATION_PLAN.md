---
date: 2026-04-25
status: proposed companion plan
scope: throughput, baselines, diagnostics, and transfer validation around SKE-RL
principle: do not expand expensive GRPO runs before cheap gates pass
---

# Optimization Plan Around SKE-RL

This document organizes the non-SKE improvements needed to make the current Text-to-SQL RL story stronger and cheaper to validate.

The main algorithmic bet remains SKE-RL: canonical SQL skeleton equivalence classes used for GRPO advantage estimation. The optimizations below are supporting work: reduce wall-clock, add reviewer-proof baselines, improve diagnostics, and make BIRD transfer credible.

## Current Constraints

- Hardware: single Quadro RTX 6000, Turing `sm_7.5`, 24 GiB.
- Quantization: bnb4 saves memory but is slow on this card; dense fp16/bf16 can be faster when it fits.
- Existing 7B SFT: improves over 3B SFT on Spider EX_attempted, especially Easy/Medium, but Hard remains flat.
- Existing 3B GRPO variants: reward engineering alone did not fix medium-query regression.
- Primary risk: spending GPU on more reward variants before proving the failure mode and credit-assignment mechanism.

## Priority Order

| Priority | Workstream | Why it matters | GPU cost | Blocking? |
|---|---|---|---:|---|
| P0 | Checkpoint-gated eval automation | Prevents wasting 100-step runs that already fail at ck25 | low | yes |
| P0 | Query-class diagnostics | Turns negative/weak RL results into publishable mechanism evidence | low | yes |
| P1 | GRPO dynamic padding / length bucketing | Direct wall-clock reduction; no algorithmic confound | low | no |
| P1 | Skeleton-reward baseline | Defends SKE-RL novelty vs structural reward work | medium | yes before claims |
| P1 | 7B dense fp16 feasibility | Removes bnb4-on-Turing slowdown/confound if it fits | low-medium | no |
| P2 | Stronger 7B SFT sanity | Prevents reviewer claim that GRPO beats a weak SFT | medium | before headline 7B |
| P2 | BIRD evidence-aware eval + coverage | Transfer/stress test, not first-stage proof | low-medium | after Spider gates |
| P3 | BIRD-mini GRPO | Only if Spider mechanism works | high | no |

## Phase A: Run Control and Early Stopping

Goal: every expensive run should produce interpretable signal by checkpoint 25.

### A1. Checkpoint Eval Script

Create a single command that evaluates `ck25/50/75/100` and prints:

- Overall `EX_attempted`
- Overall `EX_all`
- Attempt rate
- Easy / Medium / Hard `EX_attempted`
- Easy / Medium / Hard `EX_all`
- Average turns
- Unknown / no-answer count

Minimum output: one CSV plus one Markdown summary.

### A2. Stop / Continue Gates

For 7B GRPO pilot, compare against 7B SFT single-pass:

| Condition at checkpoint | Action |
|---|---|
| Medium `EX_attempted >= 56.6%` and attempt rate not collapsed | continue |
| Medium down by 3-8pp but attempt rate stable | continue to next checkpoint only |
| Medium down >8pp or attempt rate <75% | stop expanding this line |
| Easy improves but Medium drops | label as easy-overfitting, not capacity rescue |

For SKE-RL:

| Condition | Action |
|---|---|
| `ske_used <20%` sustained after warm-up | abort or fallback to raw GRPO |
| extractor failure >10% on Spider | fix extractor before training |
| `n_classes` mostly 1 | no class signal; abort SKE-RL |
| `n_classes` mostly G | no equivalence sharing; compare to raw GRPO but do not overclaim |

## Phase B: Throughput Optimizations

Goal: reduce wall-clock without changing the scientific question.

### B1. Dynamic Padding for GRPO

Apply the same principle as SFT dynamic padding to GRPO internals:

- Avoid padding prompts/completions to global max length when computing log-probs.
- Pad to longest sequence in the current micro-batch or rollout group.
- Keep Observation tokens masked out with `gen_mask=0`.
- Verify exact loss parity on a small deterministic batch before using in runs.

Expected impact: 1.3-2.0x speedup depending on prompt/trajectory length variance. This should be measured with a 10-step timing smoke.

### B2. Length Bucketing

If batch size or grouped evaluation grows beyond 1 prompt at a time:

- Bucket prompts by tokenized prompt length.
- Optionally bucket by expected difficulty / max turns.
- Do not mix this with algorithmic claims; report as engineering only.

### B3. Dense fp16 on Turing

Because RTX 6000 is Turing, bnb4 can be slower than dense precision.

Feasibility test:

```bash
MAX_STEPS=3 NUM_GEN=4 MAX_COMPLETION=384 QUANT_MODE=none PRECISION=fp16 ...
```

Track:

- peak memory
- seconds per step
- OOM / no OOM
- loss finite

Decision:

| Result | Action |
|---|---|
| fp16 fits and is >=20% faster | use dense fp16 for future 7B GRPO |
| fp16 fits but same speed | prefer fp16 to remove quantization confound |
| fp16 OOM | keep bnb4 for 7B; use dense for 3B |

## Phase C: Baselines Needed for Novelty

Goal: make SKE-RL distinguishable from existing structural reward and SQL skeleton work.

### C1. Raw GRPO Baseline

Keep one clean raw GRPO baseline with the same:

- model
- SFT adapter
- reward profile
- num generations
- max steps
- temperature schedule
- evaluation script

Do not compare SKE-RL against older runs with different reward profiles unless explicitly labeled.

### C2. Skeleton-Reward Baseline

Implement a baseline that adds skeleton similarity as reward, but keeps standard GRPO advantage:

```text
reward = original_reward + lambda * skeleton_similarity(pred_sql, gold_sql)
advantage = standard_group_relative_advantage(reward)
```

This is the key reviewer-control baseline.

Required comparison:

| Run | Reward | Advantage |
|---|---|---|
| A1 raw GRPO | original | standard |
| A2 skeleton reward | original + skeleton reward | standard |
| A3 SKE-RL | original | skeleton-class-aware |
| A4 combined | original + skeleton reward | skeleton-class-aware |

Interpretation:

| Outcome | Meaning |
|---|---|
| A3 > A1 and A3 >= A2 | advantage-level contribution supported |
| A2 > A3 | SKE-RL novelty weak; structural reward enough |
| A4 > both | mechanisms complementary |
| all fail | paper becomes diagnostic / negative result |

### C3. Stronger 7B SFT Control

Before claiming 7B GRPO lift:

- Run or cite a stronger 7B SFT sanity point: rank 64 or longer SFT.
- If compute is tight, run 200-sample eval only.
- Any 7B GRPO gain must be compared against the strongest available 7B SFT, not only rank32/300-step.

## Phase D: Diagnostics for Paper Value

Goal: locate the failure mode instead of only reporting aggregate scores.

### D1. Query-Class Tags

Tag every Spider eval example with:

- number of JOINs
- aggregation present
- GROUP BY present
- HAVING present
- ORDER BY / LIMIT present
- nested query present
- set operation present
- schema table count
- gold skeleton frequency bucket

Use these tags to report SFT vs GRPO vs SKE-RL deltas.

### D2. Trajectory Error Taxonomy

For 100 sampled failures, classify:

- no answer / malformed answer
- SQL syntax error
- schema linking error
- wrong join path
- missing condition
- wrong aggregation
- wrong grouping
- over-exploration / turn exhaustion
- correct intermediate action but wrong final answer

This is especially important if SKE-RL is null: it still yields a credible failure analysis.

### D3. Coverage-Aware Reporting

Always report both:

- `EX_attempted`: correct among attempted executable/evaluable answers
- `EX_all`: correct over all samples

GRPO can improve `EX_attempted` while reducing attempt rate. That is not a real agent improvement unless `EX_all` also holds.

## Phase E: BIRD Transfer Plan

Goal: use BIRD as a transfer/stress test after Spider mechanism is validated.

### E1. Data and Prompt Fixes

Before any BIRD claims:

- Ensure `data/bird` is present.
- Include BIRD `evidence` in the user prompt.
- Confirm execution uses real BIRD SQLite DB paths.
- Run a 50-sample smoke eval before full BIRD.

### E2. Skeleton Coverage on BIRD

Run coverage before training:

```bash
python scripts/analyze_skeleton_coverage.py \
  --datasets bird_train bird_dev \
  --extractors strict loose \
  --output results/bird_skeleton_coverage.json
```

Gate:

| Metric | Pass threshold |
|---|---:|
| BIRD strict coverage | >=75% |
| BIRD loose coverage | >=90% |
| extractor failure examples understood | required |

If strict fails but loose passes, BIRD claims should be about the purity/coverage trade-off, not strict skeleton equivalence.

### E3. BIRD-mini Only After Spider Gate

Do not run BIRD GRPO until:

- Spider SKE-RL Phase 1.1/1.2/1.5 pass
- 25-step SKE smoke is stable
- raw vs skeleton-reward vs SKE-RL comparison exists on Spider

Then run BIRD-mini as conditional validation.

## Suggested Execution Order

1. Let the current running diagnostic finish; do not interrupt GPU.
2. Evaluate any existing `grpo-7b-dense-v1/checkpoint-25`.
3. Implement checkpoint eval aggregation and query-class tagging.
4. Add skeleton-reward baseline.
5. Apply GRPO dynamic padding / timing smoke.
6. Complete SKE-RL gates and 25-step smoke.
7. Run the 4-way novelty baseline table on Spider.
8. Only then run BIRD evidence-aware eval and BIRD skeleton coverage.

## Decision Matrix

| Evidence observed | Paper direction |
|---|---|
| SKE-RL beats raw and skeleton reward on Medium + no EX_all collapse | method paper |
| SKE-RL equals skeleton reward but both beat raw | structural signal matters; method novelty weaker |
| all RL loses to SFT but diagnostics isolate query subclasses | findings/diagnostic paper |
| 7B GRPO helps Medium but SKE-RL does not | capacity/scale story, SKE-RL downgraded |
| BIRD transfer fails after Spider works | claim Spider-bound mechanism, report BIRD as stress-test limit |

## Non-Goals

- Do not add more dense reward variants before skeleton-reward baseline exists.
- Do not start full BIRD GRPO before Spider gates pass.
- Do not claim "first use of SQL skeletons"; claim advantage-estimation over skeleton equivalence classes.
- Do not optimize for `EX_attempted` alone; `EX_all` and attempt rate are mandatory.
