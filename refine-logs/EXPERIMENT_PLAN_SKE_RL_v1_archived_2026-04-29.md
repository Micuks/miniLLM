---
date: 2026-04-24
revision: R4 — staged v1/v2/v3 per external reviewer feedback
parent: FINAL_PROPOSAL.md (R4 SKE-RL)
v1_budget: ~53 GPU-h; off-ramps at every phase
---

# SKE-RL v1 Experiment Plan (Staged)

R3 had 70 GPU-h committed to a 4-way factorial that conflated post-hoc class advantage with skeleton-first generation. R4 splits this into **v1 (post-hoc only)** + **v2/v3 conditional**, addressing reviewer's H-2 critique.

## Phase 0: Implementation (no GPU, ~3 days)

### 0.1 Add sqlglot dependency
```
cd /mnt/data2/wuql/miniLLM
uv add sqlglot
uv lock --upgrade-package sqlglot
```
Smoke test: parse 50 random Spider train SQLs. Expect ≥98% successful parses.

**Verification step (R4 reviewer fix M-5)**: confirm `sqlglot` actually appears in `uv.lock` (not just `pyproject.toml`). Run `grep "name = \"sqlglot\"" uv.lock` to verify. The current repo has `uv.lock` showing as modified — confirm the modification is from sqlglot addition, not unrelated. If unrelated, re-run `uv add sqlglot` to ensure lock is updated.

### 0.2 Implement extractors
- `miniLLM/skeleton.py` — σ_strict + σ_loose
- `tests/test_skeleton.py` — 30+ Spider patterns + 10 BIRD complex patterns
  - canonical S-expr stability under whitespace, capitalization, alias renaming
  - typed slot consistency
  - JOIN-type preservation
  - subquery handling

### 0.3 Implement class-aware advantage
- `miniLLM/agent/ske_advantage.py` (new file) — `compute_class_aware_advantage` per spec in FINAL_PROPOSAL §2.2
  - **Critical**: include the degeneracy fallback to standard GRPO when distinct classes < `--ske-min-classes`
- `miniLLM/train_grpo.py` modifications (~80 LOC):
  - 4 CLI args: `--use-ske-rl`, `--ske-extractor {strict,loose}`, `--ske-beta 0.7`, `--ske-min-classes 3` (raised from 2 per R4 reviewer; min_classes=2 only as ablation)
  - Replace advantage normalization at line ~891 with conditional dispatch
  - Per-step logging: `n_classes`, `ske_used`, `ske_fallback_rate`, `extractor_failure_rate`

### 0.4 Launcher
- `scripts/train_ske_rl.sh` — clones train_grpo.sh, exports SKE flags

---

## Phase 1: Offline Diagnostics (Day 1-2, **0 GPU**)

### Run 1.1 — Coverage on 4 datasets

```bash
python scripts/analyze_skeleton_coverage.py \
    --datasets spider_train spider_dev bird_train bird_dev \
    --extractors strict loose \
    --output results/skeleton_coverage.json
```

**Reports**: coverage rate, distinct skeleton count, top-20 most common skeletons (sanity check), fallback examples for unsupported patterns.

**Gate 1.1**:
- Spider train σ_strict ≥85% AND σ_loose ≥95%: **PASS**
- BIRD train σ_strict ≥75% OR σ_loose ≥90%: **PASS** (low strict bar OK because we have loose fallback)
- ELSE: **FAIL** → tighten extractor or abort

### Run 1.2 — Within-class outcome variance over MODEL ROLLOUTS (R5 reviewer fix)

The previous formulation used gold SQL execution (which is always correct → variance trivially 0 → meaningless gate). Replaced:

```bash
python scripts/analyze_within_class_variance.py \
    --dataset spider_train --extractor strict --min-class-size 3 \
    --adapter outputs/react-sft-v2 \
    --rollouts-per-query 4 --temperature 0.7 \
    --max-classes 100 \
    --output results/within_class_variance.json
```

For each skeleton class z with ≥3 distinct training queries (cap at 100 most frequent classes for compute):
- For each query q in class z: sample K=4 SFT rollouts at training temperature 0.7
- Compute `p_q` = fraction of rollouts that execute correctly against gold result
- Within-class variance = `Var_{q in z}(p_q)`

**Gate 1.2 (hard)**: median within-class variance of `p_q` ≤ 0.15. If queries sharing a skeleton have similar SFT-rollout success rates, equivalence is meaningful. If they have wildly different success rates, the extractor is hiding semantically distinct queries → tighten or abort.

**Supplementary metrics (logged, not gated, R5 fix)**:
- Distribution of `mean_q(p_q)` per class (histogram)
- Fraction of classes with `mean_q(p_q) > 0` (i.e., at least one query in class is sometimes solvable by SFT)
- **Why**: low Var(p_q) can mean (a) genuine equivalence OR (b) all queries uniformly impossible. The supplementary metric distinguishes these. Soft target ≥40% classes with nonzero mean success — below that, equivalence may be clean but uninformative.

**Compute note**: 100 classes × ~5 queries × 4 rollouts ≈ 2000 SFT rollouts. With vLLM at ~10s/rollout = ~5h. Add this as an extra ~5 GPU-h to Phase 1 budget.

---

## Phase 1.5: **Skeleton Diversity in Actual Rollouts** (Day 3, **3 GPU-h** — NEW)

This phase directly addresses reviewer's H-1 critique. It checks whether the SFT model + GRPO temperature actually produces diverse skeletons in rollouts; if not, SKE-RL has no signal.

### Run 1.5.1 — Diversity in G=8 rollouts

```bash
python scripts/diagnose_skeleton_diversity.py \
    --adapter outputs/react-sft-v2 \
    --num-prompts 50 --num-generations 8 --temperature 0.7 \
    --extractor strict \
    --output results/skeleton_diversity_sft.json
```

For each of 50 random Spider train prompts (stratified easy/medium/hard):
- Generate G=8 rollouts at training temperature
- Extract skeleton class for each rollout's final SQL
- Compute: distinct classes per group, fraction of groups with ≥2 distinct, fraction containing **gold-shape skeleton** (σ-equivalent to σ(gold)) AND a stricter "**gold-with-table-set**" rate (gold-shape + table-set match after normalization). Report both — gold-with-table-set is the sanity check on whether shape-match actually correlates with structural correctness.

**Gate 1.5 (calibration per R4 + R5 reviewer)**:
- **G1.5-A (diversity + variance)**: across all difficulties, ≥40% of groups have ≥2 distinct skeleton classes **AND** inter-class reward variance (mean class-mean spread) ≥ 0.1. The variance check filters out "diverse failures" (≥2 classes all at reward ~0).
- **G1.5-B (positive signal, hard floor)**: for **medium queries**, ≥15% of groups contain at least one rollout matching the **gold-shape skeleton** (σ-equivalent). Stricter "gold-with-table-set" rate is reported but not gated.
- **G1.5-B' (positive signal, soft target)**: for medium queries, ≥30% containing gold is the soft target. If actual rate falls in 15-30%, paper claims must be qualified — SKE-RL is mostly sorting among wrong skeletons.

**If G1.5-A or G1.5-B FAILS**: SKE-RL has no signal to share — capacity is binding. PIVOT to "capacity-bound RL at 3B" paper.
**If G1.5-B' soft target also fails (15-30% range)**: proceed but flag in paper that SKE-RL works on wrong-skeleton variance reduction, not gold-class amplification.
**If both PASS strongly (≥30% for medium)**: full credit-assignment claim defensible.

---

## Phase 2: Smoke Pilot (Day 4, ~5 GPU-h)

### Run 2.1 — SKE-RL v1 25-step smoke

```bash
MODEL=Qwen/Qwen2.5-3B-Instruct \
SFT_ADAPTER=outputs/react-sft-v2 \
OUTPUT_DIR=outputs/grpo-ske-smoke \
MAX_STEPS=25 SAVE_STEPS=25 \
USE_SKE_RL=1 SKE_EXTRACTOR=strict SKE_BETA=0.7 SKE_MIN_CLASSES=3 \
NUM_GEN=8 LR=3e-6 \
REWARD_PROFILE=cgfr \
bash scripts/train_ske_rl.sh
```

**Diagnostic checks** (logged per step):
- `loss` non-NaN, decreasing trend
- `n_classes` averages 3-6 (not 1 = collapse, not 8 = no equivalence)
- `ske_used` rate >50% (rest falls back to standard advantage)
- `extractor_failure_rate` <20%

**Gate 2.1**: training stable, advantages non-degenerate. If `ske_used` <20%, σ too strict for current SFT model → switch to σ_loose for Phase 3 strict slot. Also tighten the smoke-pilot gate: `extractor_failure_rate` <**10%** on Spider (R4 reviewer: original 20% was too generous). If higher, investigate failing patterns before Phase 3.

**In-training kill switch (R4 addition)**: monitor `ske_used` rolling 20-step window. After warm-up step 30, if `ske_used` drops below 20% sustained for 20 consecutive steps → halt training, fall back to standard GRPO for the remaining steps (or abort for diagnosis). This formalizes H-1's residual concern: pre-training Phase 1.5 doesn't guarantee in-training diversity persists.

---

## Phase 3: Three-Way Spider Comparison (Day 5-7, ~15 GPU-h)

The decisive v1 experiment. Compute-matched: same NUM_GEN=8, same MAX_STEPS=100, same reward profile.

### Run 3.1 — A1: raw GRPO baseline

```bash
OUTPUT_DIR=outputs/v1-a1-raw-grpo \
MAX_STEPS=100 SAVE_STEPS=25 \
REWARD_PROFILE=cgfr \
bash scripts/train_grpo.sh
```

Cost: ~5 GPU-h. (Existing config; we may already have this from earlier CGFR runs — check before re-running.)

### Run 3.2 — A2: SKE-RL v1 σ_strict

```bash
OUTPUT_DIR=outputs/v1-a2-ske-strict \
MAX_STEPS=100 SAVE_STEPS=25 \
USE_SKE_RL=1 SKE_EXTRACTOR=strict SKE_BETA=0.7 SKE_MIN_CLASSES=3 \
REWARD_PROFILE=cgfr \
bash scripts/train_ske_rl.sh
```

Cost: ~5 GPU-h.

### Run 3.3 — A3: SKE-RL v1 σ_loose

Same as 3.2 with `SKE_EXTRACTOR=loose`. Cost: ~5 GPU-h.

### Run 3.4 — Eval all 3 conditions

```bash
for cond in a1-raw-grpo a2-ske-strict a3-ske-loose; do
    for ck in 25 50 75 100; do
        bash scripts/eval_agent_spider.sh \
            outputs/v1-${cond}/checkpoint-${ck} \
            outputs/eval_v1_${cond}_ck${ck}.json
    done
done
```

Cost: 3 × 4 × ~25min ≈ 5 GPU-h.

### Stratified analysis

```bash
python scripts/analyze_v1_factorial.py \
    --base outputs/eval_v1_*.json \
    --stratify-by difficulty skeleton_class_freq \
    --output results/v1_factorial_analysis.md
```

**Gate 3 (decisive for v1)**:

| Outcome | Action |
|---|---|
| A2 OR A3: medium EX_att ≥ A1 + 3pp | **GO v2**: pursue fill-search prototype |
| A2/A3 within ±3pp of A1 overall | **CONDITIONAL**: investigate per-difficulty; consider v2 if any difficulty improves |
| Both A2 AND A3 worse by >3pp | **HALT**: investigate degeneracy; likely write up as null result |

---

## Phase 4: BIRD Validation (Day 8, ~8 GPU-h, conditional on Phase 3 GO)

### Run 4.1 — A2: σ_strict on BIRD-mini

```bash
DATASET=bird-mini \
OUTPUT_DIR=outputs/v1-a2-bird-strict \
USE_SKE_RL=1 SKE_EXTRACTOR=strict SKE_BETA=0.7 \
MAX_STEPS=100 \
bash scripts/train_ske_rl.sh
```

Cost: ~3 GPU-h.

### Run 4.2 — A3: σ_loose on BIRD-mini

Same with σ_loose. Cost: ~3 GPU-h.

### Run 4.3 — Eval A2/A3/baseline on BIRD eval set

Cost: ~2 GPU-h.

**Reported finding**: σ_strict EX vs coverage rate, σ_loose EX vs coverage rate, baseline. The trade-off curve is the second contribution of the paper.

---

## Phase 5: Multi-Seed Validation (Day 9-10, ~12 GPU-h, conditional on Phase 3 + Phase 4 reasonable)

Run the winning condition (likely A2 σ_strict if it cleared Gate 3) on 3 seeds:

```bash
for seed in 42 123 456; do
    OUTPUT_DIR=outputs/v1-multiseed-seed${seed} \
    SEED=$seed \
    USE_SKE_RL=1 SKE_EXTRACTOR=strict SKE_BETA=0.7 \
    bash scripts/train_ske_rl.sh
done
```

Cost: 3 × 4h = 12 GPU-h.

Report mean ± std on Spider 200 + BIRD-mini.

**Gate 5**: std < 5pp across seeds. Otherwise need more seeds or larger eval.

---

## v2 Roadmap (Conditional on Phase 3 GO)

### v2 Prototype (offline, ~1 day, 0 GPU)

```
python scripts/v2_fill_feasibility.py \
    --skeletons-from-train spider_train \
    --K-attempts 8 \
    --extractor strict \
    --output results/v2_fill_feasibility.json
```

For each unique skeleton in Spider train: sample K=8 candidate fills via constrained decoding from SFT model. Check executability.

**Gate v2-prototype**: ≥70% of skeletons admit at least one parseable+executable fill within K=8 → GO v2 implementation.

### v2 Implementation (~3 days)

Add `R_struct(z) = max over K fills of execution_match` to advantage.

### v2 Factorial (~25 GPU-h)

| Condition | Description |
|---|---|
| **B1** | A2 v1 σ_strict baseline (already have from Phase 3) |
| **B2** | v2 with K=4 fills |
| **B3** | v2 with K=8 fills |

Compute matched per fill-search budget (B1 has 0 extra rollouts; B2 has 4× extra rollouts → reduce B1's training steps proportionally for fairness, OR report uncompensated).

**Gate v2**: B2 OR B3 medium EX ≥ B1 + 3pp.

---

## v3 Roadmap (Conditional on v2 GO with ≥+5pp medium gain)

v3 = skeleton-first generation. Requires:
- New action space (emit `<skel>` then `<fill>` tags)
- Constrained decoding for skeleton phase
- ReAct prompt template revision
- ~500 LOC; 3-5 days

Skip detail until v2 motivates it.

---

## Tracker (`EXPERIMENT_TRACKER.md`)

Maintain per-run rows:
```
Run: <id>
- Started: YYYY-MM-DD HH:MM
- Config: link to scripts/...
- Status: pending | running | done | failed
- Result (Spider 200 EX_att / EX_all): X / Y
- Per-difficulty (E/M/H): A% / B% / C%
- ske_used rate: X% (NA for raw)
- avg n_classes: N (NA for raw)
- Notes: anomalies, training curve obs
```

---

## v1 Total Budget

| Phase | GPU-h | Wall-clock (single GPU) |
|---|---|---|
| 0 (impl) | 0 | 3 days dev |
| 1.1 (coverage) | 0 | 0.5 day |
| 1.2 (within-class via SFT rollouts — R5 fix) | 5 | 0.5 day |
| 1.5 (diversity at SFT temp) | 3 | 0.5 day |
| 2 (smoke) | 5 | 0.5 day |
| 3 (Spider 3-way) | 15 | 1.5 days |
| 4 (BIRD, cond.) | 8 | 1 day |
| 5 (multi-seed, cond.) | 12 | 1 day |
| Reserve | 5 | — |
| **v1 total** | **~53 GPU-h** | **~7-8 days** |

If GPU is shared with 7B SFT (current state): Phase 0+1.1 proceed in parallel; Phases 1.2+ wait.

---

## Decision Tree (Off-Ramps)

```
[Phase 0 impl complete]
        ↓
[Phase 1 coverage]
   pass → continue
   fail → abort SKE-RL, write extractor-limit note, pivot to EPR-X
        ↓
[Phase 1 within-class variance]
   pass → continue
   fail → tighten σ; if can't tighten, abort
        ↓
[Phase 1.5 SFT skeleton diversity]
   pass → SKE-RL has signal → continue
   fail (medium) → CAPACITY hypothesis confirmed → write capacity paper instead
        ↓
[Phase 2 smoke]
   stable → continue
   degenerate → debug; if can't fix, abort
        ↓
[Phase 3 3-way Spider]
   A2/A3 win medium ≥+3pp → GO v2
   no improvement → null result paper (still publishable)
   harm → debug or abort
        ↓
[Phase 4 BIRD]
   reasonable → multi-seed
   bad → restrict paper to Spider
        ↓
[Phase 5 multi-seed]
   std<5pp → ready for paper
   high std → need more seeds or larger eval
```

Off-ramps are intentional. Each is a different paper, but all are honest.
