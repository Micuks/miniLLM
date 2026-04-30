---
date: 2026-04-24
revision: R4 (replaces R3 4-way factorial design)
parent: FINAL_PROPOSAL.md (R4) + EXPERIMENT_PLAN.md (R4)
status: pre-implementation
---

# SKE-RL v1 Experiment Tracker

R3 4-way factorial (C1/C2/C3/C4) is **deprecated**. R4 staged design uses A1/A2/A3 in v1 with optional v2/v3 conditional.

Per-run log. Update after each run completes.

## Phase 0: Implementation  ✅ COMPLETE (2026-04-24)

- [x] **Day 0**: `uv add sqlglot` → sqlglot==30.6.0 in uv.lock
- [x] **Day 0**: Smoke test — 100% parse on 50 random Spider train SQLs
- [x] **Day 1**: `miniLLM/skeleton.py` — σ_strict + σ_loose; canonical S-expr; 290 LOC
- [x] **Day 1**: `tests/test_skeleton.py` — 33 Spider patterns + 10 BIRD × (strict+loose); 98 tests passing
- [x] **Day 2**: `miniLLM/agent/ske_advantage.py` — `compute_class_aware_advantage` with size-weighted baseline + min_classes=3 fallback
- [x] **Day 2**: `train_grpo.py` integration — 5 CLI args, advantage dispatch, in-training kill switch, per-step logging (`ske_used`, `avg_n_classes`, `killed`)
- [x] **Day 2**: `scripts/train_ske_rl.sh`
- [x] **Day 2**: `scripts/analyze_skeleton_coverage.py` — **Gate 1.1 already PASSED** on Spider: 100% strict coverage, 947 classes on 2000 train / 481 classes on 1034 dev
- [x] **Day 2**: `scripts/analyze_within_class_variance.py` — uses SFT rollouts (not gold) per R5 fix; needs GPU
- [x] **Day 3**: `scripts/diagnose_skeleton_diversity.py` — Phase 1.5 gate; needs GPU

**Smoke result**: all imports clean, 98 pytests pass, coverage analysis PASS on Spider.
**Remaining**: Phase 1.2 (within-class variance) + 1.5 (diversity) need GPU (~8 GPU-h total).

## Phase 1: Offline Diagnostics (Day 1-2, 0 GPU)

### Run 1.1 — Coverage analysis  ✅ PASSED (2026-04-24)
- **Status**: completed
- **Output**: `/mnt/data2/wuql/miniLLM/results/skeleton_coverage_smoke.json`
- **Spider train σ_strict / σ_loose**: 100% / 100% (947 / 928 classes on 2000 samples)
- **Spider dev σ_strict / σ_loose**: 100% / 100% (481 / 475 classes on 1034 samples)
- **BIRD train σ_strict / σ_loose**: TBD (BIRD loader not wired; run once available)
- **Gate 1.1 verdict**: **PASS** on Spider; BIRD pending

### Run 1.2 — Within-class variance over MODEL rollouts
- **Status**: pending
- **GPU cost**: ~5h (uses SFT model for K=4 rollouts × ~500 queries)
- **Output**: `results/within_class_variance.json`
- **Median Var_q(p_q) | z target ≤ 0.15**: TBD
- **Gate 1.2 verdict**: TBD

## Phase 1.5: Skeleton Diversity in Actual Rollouts (Day 3, ~3 GPU-h) — NEW

### Run 1.5.1 — SFT rollout skeleton diversity
- **Status**: pending
- **Output**: `results/skeleton_diversity_sft.json`
- **Across all difficulties: % groups with ≥2 distinct classes (gate ≥40%)**: TBD
- **Inter-class reward variance ≥0.1 (gate)**: TBD
- **Medium queries: % groups containing gold-shape skeleton (hard ≥15%, soft ≥30%)**: TBD
- **Gate 1.5 verdict**: TBD
- **Decision**:
  - If gate fails for medium → PIVOT to capacity-confirmation paper
  - If 15-30% medium → proceed but qualify claims
  - If ≥30% medium → full credit-assignment claim defensible

## Phase 2: Smoke Pilot (Day 4, ~5 GPU-h)

### Run 2.1 — SKE-RL v1 25-step smoke (σ_strict, min_classes=3)
- **Status**: pending
- **Config**: USE_SKE_RL=1 SKE_EXTRACTOR=strict SKE_BETA=0.7 SKE_MIN_CLASSES=3 MAX_STEPS=25
- **Output dir**: `outputs/grpo-ske-smoke`
- **Loss trajectory (non-NaN, decreasing)**: TBD
- **Avg n_classes per step**: TBD
- **ske_used rate**: TBD (gate >50%)
- **extractor_failure_rate**: TBD (gate <10%)
- **Gate 2.1 verdict**: TBD

## Phase 3: Three-Way Spider Comparison (Day 5-7, ~15 GPU-h)

### Run 3.1 — A1: raw GRPO baseline (CGFR config, 100 steps)
- **Status**: pending (may reuse existing CGFR run if available)
- **Output dir**: `outputs/v1-a1-raw-grpo`
- **Spider 200 EX_att / EX_all**: TBD
- **Easy / Med / Hard**: TBD

### Run 3.2 — A2: SKE-RL v1 σ_strict
- **Status**: pending
- **Config**: USE_SKE_RL=1 SKE_EXTRACTOR=strict SKE_BETA=0.7 SKE_MIN_CLASSES=3
- **Output dir**: `outputs/v1-a2-ske-strict`
- **Spider 200 EX_att / EX_all**: TBD
- **Easy / Med / Hard**: TBD
- **Avg ske_used rate**: TBD
- **In-training kill switch triggered?** (ske_used <20% sustained for 20 steps after warm-up): TBD

### Run 3.3 — A3: SKE-RL v1 σ_loose
- **Status**: pending
- **Config**: USE_SKE_RL=1 SKE_EXTRACTOR=loose SKE_BETA=0.7 SKE_MIN_CLASSES=3
- **Output dir**: `outputs/v1-a3-ske-loose`
- **Spider 200 EX_att / EX_all**: TBD
- **Easy / Med / Hard**: TBD

### Phase 3 Eval & Stratified Analysis
- **Status**: pending
- **Output**: `results/v1_factorial_analysis.md`
- **Gate 3 verdict**: TBD
  - GO v2 if A2 OR A3 medium EX ≥ A1 + 3pp
  - HALT if both A2 AND A3 worse by >3pp

## Phase 4: BIRD Validation (Day 8, ~8 GPU-h, conditional)

### Run 4.1 — A2: σ_strict BIRD-mini
- **Status**: pending (gated by Phase 3 GO)
- **Output dir**: `outputs/v1-a2-bird-strict`
- **EX**: TBD
- **Coverage rate**: TBD

### Run 4.2 — A3: σ_loose BIRD-mini
- **Status**: pending (gated by Phase 3 GO)
- **Output dir**: `outputs/v1-a3-bird-loose`
- **EX**: TBD
- **Coverage rate**: TBD

### Run 4.3 — BIRD baseline + eval
- **Status**: pending

## Phase 5: Multi-Seed (Day 9-10, ~12 GPU-h, conditional)

### Run 5.{42,123,456} — winning condition × 3 seeds
- **Status**: pending
- **Mean ± std EX_att Spider**: TBD
- **Mean ± std EX_att BIRD**: TBD

## Optional Phase 6: 7B Anchor (gated by 7B SFT)

### Run 6.1 — 7B raw GRPO baseline (if budget remains)
- **Status**: pending (gated by 7B SFT completion)

### Run 6.2 — 7B SKE-RL v1 (if Phase 3 GO)
- **Status**: pending

---

## v2 Roadmap (Conditional on Phase 3 GO)

### v2 Prototype (offline, 0 GPU)
- **Status**: pending
- For each unique skeleton in Spider train: sample K=8 candidate fills via constrained decoding from SFT
- **Gate v2-prototype**: ≥70% skeletons admit ≥1 executable fill within K=8

### v2 Implementation + Factorial (~25 GPU-h)
- B1 = A2 v1 baseline (reuse Phase 3)
- B2 = v2 K=4 fills
- B3 = v2 K=8 fills
- **Gate v2**: B2 OR B3 medium EX ≥ B1 + 3pp

## v3 Roadmap (Conditional on v2 GO with ≥+5pp medium)

Skeleton-first generation: ~500 LOC; 3-5 days. Skip detail until v2 motivates.

---

## OPTIMIZATION_PLAN Companion Progress (2026-04-25)

Companion plan: `refine-logs/OPTIMIZATION_PLAN.md`. Items below are the
non-SKE supporting work — all CPU/file only so the GPU-bound running
experiment was undisturbed.

| Item | Status | Deliverables | Smoke result |
|---|---|---|---|
| **A1** Checkpoint eval aggregator | ✅ done | `scripts/aggregate_eval_reports.py` (CPU); `scripts/eval_checkpoints.sh` (GPU loop) | Verdict on existing CGFR ck25-100 vs 7B SFT: **STOP** all 4 (med drop -23 to -32pp, attempt rate 78-80%). |
| **D1** Query-class tagging | ✅ done | `miniLLM/query_class.py` + `scripts/tag_eval_results.py` + 10 pytests | SFT vs CGFR ck100 breakdown: nojoin -23pp, has_nested -20pp, 1join -10pp. Failure mode is concentrated, not uniform. |
| **C2** Skeleton-reward baseline | ✅ done | `skeleton_similarity()` in `miniLLM/skeleton.py`; `--skeleton-reward-coef` in `train_grpo.py`; `scripts/train_skeleton_reward.sh`; 6 pytests | Coef=0 path is byte-identical to baseline (default off). Identical SQL → 1.0; empty pred → 0.0; equivalence-class match → 1.0. |
| **B1** Dynamic padding helper + parity | ✅ done | `miniLLM/agent/dynamic_padding.py` + 3 pytests | Per-seq vs batched log-prob delta < 1e-5 over uneven-length groups with mid-token masks. **Not yet wired into `train_grpo.compute_log_probs`** — defer integration to avoid confounding the running experiment. |

**Test count**: 98 skeleton + 10 query_class + 6 skeleton_similarity + 3 dynamic_padding = **117 pytests, all green**.

**Immediate diagnostic finding** (no GPU spent): the existing 3B CGFR runs collapse on no-join and nested queries; per-class table now isolates the regression rather than reading it as a flat -25pp medium drop.

**Next steps** (need GPU release):
1. `bash scripts/eval_checkpoints.sh RUN_DIR=outputs/grpo-7b-dense-v1 CHECKPOINTS=25 …` to evaluate the 7B GRPO ck25 already on disk and apply the A2 stop/continue verdict.
2. Phase 1.2 (within-class variance) and Phase 1.5 (skeleton diversity) — same GPU budget as before.
3. Wire `dynamic_padding.pad_group` into `compute_log_probs` after current run finishes; rerun parity test to confirm exact equality on the real 7B model.
4. C2 launcher (`train_skeleton_reward.sh`) is the A2 cell of the four-way novelty comparison.

---

## Aggregate Decision Log

| Date | Phase | Decision | Rationale |
|---|---|---|---|
| 2026-04-24 | R3 selection | Lock SKE-RL | Reviewer 7/10 novelty |
| 2026-04-24 | R4 staged | v1 only (post-hoc) | Reviewer H-1/H-2/H-3 critiques |
| 2026-04-24 | R4 reviewer | "ready for Phase 0" | 5/6 fully resolved + implementation traps noted |
| 2026-04-24 | R5 reviewer | 5 more patches applied | Phase 1.2 use rollouts not gold; tracker updated to v1; min_classes=3 enforced; gold-shape skeleton language; fallback wording |
| 2026-04-25 | OPT_PLAN | A1+D1+C2+B1 file-only items shipped | All CPU/file work, 117 tests green, GPU running 7B exp untouched |
| 2026-04-25 | OPT_PLAN review fixes | 6 reviewer findings patched | (1) eval_checkpoints.sh CLI args fixed (`--interactive --with-execution`); (2) BIRD `evidence` threaded into `build_react_*` (kwarg-only, backward compatible); (3) `--source bird` added with full loader wiring; (4) `--clip-eps-high` added to argparse + asymmetric DAPO clipping in PPO loss; (5) aggregator now derives `attempted` from records when `execution_counts` missing; (6) SKE counters moved past advantage-threshold/valid_indices skips so rolling rate matches steps that actually update. 126 pytests green. |
| 2026-04-26 | OPT_PLAN review R2 | 5 follow-up findings patched | (1) eval_checkpoints.sh only aggregates over reports that actually exist (no more "file not found" on missing checkpoints); (2) analyze_skeleton_coverage.py now emits full `strict/loose_class_counts` maps in addition to top-20; tag_eval_results.py prefers them and adds `--schemas-from spider\|bird` to re-derive schema from db_id (eval records lack schema_ddl, so the freq map matches what coverage saw); (3) `ske_used_history.append` moved past advantage-threshold/valid_indices skips — kill-switch denominator now matches `ske_used_steps`; (4) BIRD loader resolves `dev.json/tables.json/databases/` flat OR one-level-nested layouts (handles official zip's `dev_<DATE>/` extraction); (5) RVDS variance check now uses `raw_rewards` (pre-skeleton-bonus) so the C2 baseline doesn't artificially inflate RVDS-pass rate. 130 pytests green (+4 BIRD path resolver tests). |
| 2026-04-26 | OPT_PLAN review R3 | 3 follow-up findings patched | (1) `compute_class_aware_advantage` no longer counts pseudo-classes (`_no_sql_*`, `_parse_fail_*`) toward `min_classes` — uses `n_real_classes`; new `--ske-max-extractor-failure-rate` (default 0.5) forces fallback when too many parse failures; reproduced reviewer's smoke (G=4 with 75% failure now correctly returns `ske_used=False`, `fallback_reason='too_few_real_classes'`); (2) aggregator verdict adds explicit medium-attempt gate (`STOP (med-attempt X% <75%)`) so easy-skewed runs can't mask a collapsed medium attempted subset; (3) kill-switch check moved BEFORE SKE advantage compute so the trigger step itself uses standard GRPO (not SKE). 136 pytests green (+5 SKE gate tests, +1 aggregator test). |
| 2026-04-26 | OPT_PLAN review R4 | 3 more findings patched | (1) BIRD difficulty stratification: eval_agent.py iterates over simple/moderate/challenging in addition to easy/medium/hard so BIRD eval JSONs no longer drop their by-difficulty splits; aggregator `_diff_block` falls back from easy↔simple, medium↔moderate, hard↔challenging so OPTIMIZATION_PLAN A2 verdicts apply to BIRD reports unchanged. Records-based attempted lookup also matches both naming conventions. (2) Per-step SKE log now reports `real_used` (avg n_real_classes over steps where SKE actually fired — the meaningful diversity signal) and `real_all` (avg over all updating steps). Pseudo-classes excluded from both. New `top_fb=<reason>(N)` field surfaces dominant fallback cause from `ske_fallback_reasons` Counter. (3) Aggregator markdown verdict legend rewritten as ordered gate list including the new med-attempt rule and BIRD mapping note. 137 pytests green (+1 BIRD-shape aggregator test). |
| 2026-04-29 | external review | Story A → Diagnostic Study reframe | Codex xhigh 2-round review (gpt-5.5, session 019dd6aa). SKE-G8 result on full Spider-dev (70.87 EX_att, -1.81 vs Dense75) confirmed SKE underperforms Dense and degrades monotonically with G. Verdict on Story A as written: **borderline reject** at NeurIPS (Soundness 2/4, Contribution 2/4). Recommended pivot: workshop submission framed as "Diagnostic Study of Multi-turn Text-to-SQL RL" with SKE as one of three findings (not lead). Highest-priority diagnostic flagged: format-vs-reasoning partition on existing eval JSONs (zero-cost; tests whether Dense +2pp is protocol/tool-call compliance vs SQL reasoning — most damaging alternative explanation). Full review + ranked experiment plan + claims matrix in `refine-logs/RESEARCH_REVIEW_2026_04_29.md`. |
| 2026-04-29 | research-refine R1-R4 | A+B story refined to READY (9.00/10) | 4 rounds of codex xhigh review (session 019dd71e) on the protocol-vs-semantic + SKE-as-mechanism story. Score 7.35 → 8.35 → 8.90 → 9.00 READY. Final method = leakage-safe lenient extractor + deterministic 3-bucket classifier + signed `lenient_attribution_gap` with bootstrap CIs + SKE single-mechanism test (E1+E2+E3). Outputs: `FINAL_PROPOSAL.md`, `EXPERIMENT_PLAN.md`, `REVIEW_SUMMARY.md`, `REFINEMENT_REPORT.md`. Old SKE-RL v1 proposal archived. |
| 2026-04-29 | post-READY user review (R5) | 5 fixes applied to FINAL_PROPOSAL + EXPERIMENT_PLAN | (1) A0 sanity gate strengthened to `parseable_lenient_rate ≥80%` + 20-record manual audit (old gate was trivially-passing); (2) **E3 DROPPED from C2 pass condition** — per-`skeleton_class` advantage shifts were never persisted by the existing SKE-G4/G8 training runs (verified `train_grpo.py:1119-1350` only keeps aggregate counters); E3 reported as Tier-3 descriptive context only. C2 PASS = E1 AND E2 (binary). (3) Absolute paths + SHA256 + mtime for the 4 input eval JSONs locked into PRE_REGISTRATION; (4) `semantic_gain` bucket tightened to also require `RL.final_correct_strict` so both gain buckets are subsets of the strict-gain transition set; `lenient_only_repair_flag` added as appendix subanalysis; (5) GPU-h vs CPU/IO-h column accounting fixed. **Dense G=8 confound run GATED on C1 PASS + E1 PASS** — saves 28 GPU-h if C1 inverts. |

(Append decision rows as they happen.)

---

## Compute Budget Tracking

| Phase | Estimated GPU-h | Actual GPU-h | Cumulative |
|---|---|---|---|
| 1 (incl 1.2 rollouts) | 5 | — | — |
| 1.5 | 3 | — | — |
| 2 (smoke) | 5 | — | — |
| 3 (3-way Spider) | 15 | — | — |
| 4 (BIRD, cond.) | 8 | — | — |
| 5 (multi-seed, cond.) | 12 | — | — |
| Reserve | 5 | — | — |
| **v1 total** | **~53 GPU-h** | — | — |

Budget cap: 60 GPU-h v1. Off-ramp if cumulative exceeds 80% with no Phase 3 GO.

---

## Implementation Traps Reminder (R4 + R5 reviewer)

1. ✅ Class baseline must be **size-weighted** (not equal-weighted) — fixes singleton-trap
2. ✅ `--ske-min-classes` default = **3** (not 2) — defense in depth against singleton inflation
3. ✅ In-training kill switch on `ske_used` rolling 20-step window <20%
4. ✅ Phase 1.2 uses **MODEL rollouts**, not gold SQL (gold has trivially 0 variance)
5. ✅ Phase 1.5 G1.5-A requires inter-class reward variance ≥ 0.1 (rules out "diverse failures")
6. ✅ Phase 2 `extractor_failure_rate` gate tightened to <10% (was 20%)
7. ✅ Sqlglot must actually be in `uv.lock` after `uv add` — verify before Day 1
8. ✅ Use "gold-shape skeleton" terminology — skeleton matches type/arity, not table-set; report optional stricter "gold-with-table-set" metric for tighter claim
