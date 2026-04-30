---
date: 2026-04-29
parent: FINAL_PROPOSAL.md (R4 READY @ 9.00/10)
status: ready-to-execute
horizon: 3 weeks GPU + 1 week writing
hardware: 1× Quadro RTX 6000 24 GB (Turing sm_7.5)
---

# Experiment Plan — Protocol Beats Structure (Diagnostic Study)

This plan operationalizes `FINAL_PROPOSAL.md`. Designed to be executable with no further design decisions: each experiment block has explicit inputs, outputs, success criteria, GPU cost, and assigned wall day.

## Pre-registration (do this first, lock in writing)

Before any data inspection, write `refine-logs/PRE_REGISTRATION.md` containing (R5 reviewer fix):

1. **Absolute paths + integrity hashes** for all 4 input eval JSONs:
   ```
   /mnt/data2/wuql/miniLLM/outputs/eval_full_dev_sft.json
     size=1617445  sha256=291f54b88a937be2f0994f7d7dcf7f40c6a315e61de0563d882ef6c81bac5fa3
     mtime=2026-04-27T12:23:57+0800
   /mnt/data2/wuql/miniLLM/outputs/eval_full_dev_dense75.json
     size=1576943  sha256=e0b733ab181e980e0344e7b4d21723030280cae820b9930d5ca8438e06255888
     mtime=2026-04-27T16:05:11+0800
   /mnt/data2/wuql/miniLLM/outputs/eval_full_dev_ske75.json
     size=1586069  sha256=7d6cbd3e5f438731b4e1b1448d93845d2baf93683ea66d3aeae2c4328f83bc66
     mtime=2026-04-27T19:54:20+0800
   /mnt/data2/wuql/miniLLM/outputs/eval_full_dev_ske_g8_ck75.json
     size=1566308  sha256=390eb1bf86c9a934fa4e839854cc0a4e40a7f950c4c5ee0964351cba0567d85b
     mtime=2026-04-29T01:02:09+0800
   ```
   All scripts MUST verify SHA256 before reading. Document directory is `/mnt/data1/wuql/dev/miniLLM/refine-logs/`; data + code directory is `/mnt/data2/wuql/miniLLM/`. Never rely on relative `outputs/` lookups.
2. **Bootstrap protocol**: 1000 resamples over Spider-dev records (1034); 95% percentile CIs.
3. **Bucket precedence**: protocol_gain > semantic_gain > no_change. Both gain buckets require `RL.final_correct_strict = True` (R5 fix). Subanalysis flag `lenient_only_repair_flag = (not RL.strict and RL.lenient)` reported in appendix only.
4. **A0 sanity gate (R5 reviewer fix)**: TWO separate criteria, both must pass:
   - **Quantitative**: `parseable_lenient_rate ≥ 80%` on a 50-record sample where `strict_extract_ok = False`.
   - **Qualitative**: manual audit of a 20-record subsample of those — for each, confirm the lenient extractor returned the *intended* final SQL (not Observation text, not a non-final exploratory action). Document the audit results before running A1.
   - **Old gate `lenient_EX(SFT) ≥ strict_EX(SFT)` is removed** because strict was 0 by selection — trivially satisfied.
5. **Pass/fail definitions** for E1, E2, R1, R2, R3.
6. **E3 status (R5 fix)**: NOT a C2 pass condition. The per-`skeleton_class` advantage shifts E3 needs are NOT persisted by the existing SKE training runs (verified against `train_grpo.py` line 1119-1350: only aggregate counters survive to disk). E3 is reported only as a Tier-3 aggregate descriptive comparison (`ske_n_real_classes_sum`, `class_size_distribution` between strata) — for transparency, not as evidence. Re-running SKE arms with new logging is out of budget (~48 GPU-h) and explicitly deferred.
7. **Dense G=8 launch gate (R5 fix)**: Dense G=8 confound run is GATED on C1 result. Only launch if Week 1 reports `lenient_attribution_gap > 0` with 95% CI excluding 0. Otherwise the SKE-specific story has already been weakened by the C1 inversion, and 28 GPU-h is better spent on writing.

This document gets timestamped and committed BEFORE Block A1 runs.

## Resources Already On Disk (R5 reviewer fix: absolute paths verified)

All paths rooted at `/mnt/data2/wuql/miniLLM/` unless noted. Documents at `/mnt/data1/wuql/dev/miniLLM/refine-logs/`.

| Asset | Absolute path | SHA256 (first 12) | Use |
|---|---|---|---|
| 7B SFT eval, full Spider-dev 1034 | `outputs/eval_full_dev_sft.json` | 291f54b88a93 | C1 baseline |
| 7B Dense75 eval, full Spider-dev | `outputs/eval_full_dev_dense75.json` | e0b733ab181e | C1 + C2 |
| 7B SKE-G4-75 eval, full Spider-dev | `outputs/eval_full_dev_ske75.json` | 7d6cbd3e5f43 | C2 |
| 7B SKE-G8-75 eval, full Spider-dev | `outputs/eval_full_dev_ske_g8_ck75.json` | 390eb1bf86c9 | C2 |
| Skeleton extractor + 98 pytests | `miniLLM/skeleton.py`, `tests/test_skeleton.py` | — | classifier flags |
| Per-class diagnostic | `scripts/analyze_eval_by_skeleton.py` | — | Tier-3 descriptive context |
| Per-record query tagger | `miniLLM/query_class.py`, `scripts/tag_eval_results.py` | — | bucket subanalysis |
| Aggregator with verdict gates | `scripts/aggregate_eval_reports.py` | — | sanity views |
| ~~SKE per-`skeleton_class` advantage shift logs~~ | **NOT PERSISTED** (R5 fix) | — | E3 cannot run — see PRE_REGISTRATION §6 |
| Real Spider DBs | `data/spider_data/database/` | — | re-execution |

## Software To Build

| New module | Path | Lines | Tests |
|---|---|---:|---|
| Leakage-safe lenient extractor | `miniLLM/diag/lenient_extract.py` | ~80 | 6 unit tests (Answer-region hit / Answer-region parse-fail / fall-through to last-action / no extraction / parse-only deterministic / never-touches-gold) |
| Per-record classifier | `miniLLM/diag/classifier.py` | ~120 | 8 unit tests (each bucket / precedence / flag combinations / mutual exclusion) |
| Bootstrap CI helpers | `miniLLM/diag/bootstrap.py` | ~50 | 3 tests (CI of mean / proportion / Cliff's δ) |
| Cliff's δ + equivalence test | `miniLLM/diag/equivalence.py` | ~60 | 4 tests (zero effect / large effect / small effect equivalence pass / boundary case) |
| End-to-end diagnostic script | `scripts/diag_protocol_vs_semantic.py` | ~200 | 1 integration test on 10-record fixture |
| SKE join + E3 driver | `scripts/diag_ske_mechanism.py` | ~150 | covered by integration |

Total new code: ~660 LOC + ~22 unit tests + 1 integration test. All testable on CPU with fixtures derived from existing eval JSON records.

## Claim → Experiment Matrix

| Claim | Experiment block | Cost | Pass condition |
|---|---|---|---|
| **C1** dominant: protocol-attribution | A1 + A2 + A3 | ~6 CPU/IO-h + CPU classifier | `lenient_attribution_gap > 0` with 95% CI excluding 0; `transition_bucket_share[protocol_gain] > [semantic_gain]` with non-overlapping CIs |
| **C2** supporting: SKE single mechanism (R5: E3 dropped) | B1 + B2 (mandatory) + B4 (gated) | ~12 CPU/IO-h + 28 GPU-h conditional | **E1 AND E2 must hold**; Dense G=8 (if launched) ≥ Dense G=4 − 1pp |
| (R1) lenient extractor sanity (R5 strengthened) | A0 | <1 CPU/IO-h | `parseable_lenient_rate ≥ 80%` on 50 records AND 20-record manual audit confirms intended final SQL |
| (R2) `semantic_gain` non-empty | implicit in A2 | 0 | ≥10 records in `semantic_gain` for (Dense vs SFT); else flag |
| (R3) Dense G=8 not generic-bad | B4 (gated on C1 result) | ~28 GPU-h | within 1pp of Dense G=4 within bootstrap CI |
| ~~(R4) E3 join coverage~~ | **DROPPED — see PRE_REGISTRATION §6** | — | Per-`skeleton_class` adv shifts never persisted; E3 is Tier-3 descriptive only |

## Week-by-Week Schedule

### Week 1 — Build + dominant contribution (CPU-heavy, ~6 CPU/IO-h, no GPU)

**Day 1 (Mon): Pre-registration + module scaffolding**
- Write `PRE_REGISTRATION.md`. Commit. Timestamp.
- Create `miniLLM/diag/` package skeleton.
- Write 3 modules: `lenient_extract.py`, `classifier.py`, `bootstrap.py` + their tests. ~210 LOC + 17 tests.
- Wall: 1 day.

**Day 2 (Tue): A0 — Lenient extractor sanity (R1, strengthened per R5 reviewer)**
- Subsample 50 records from `/mnt/data2/wuql/miniLLM/outputs/eval_full_dev_sft.json` where `strict_extract_ok = False`.
- Run `lenient_extract` on each; record per-record `(parseable: bool, extracted_sql: str | None, parse_error: str | None)`.
- **Quantitative gate**: `parseable_lenient_rate ≥ 80%`. Else iterate extractor before A1 launches.
- **Qualitative gate**: manually audit a 20-record subsample. For each, confirm that `extracted_sql` is the *intended* final SQL — NOT Observation text echoed by the model, NOT an exploratory non-final action. Document audit results (e.g., "20/20 intended, 0 misclassifications" or "17/20 intended, 3 caught Observation echoes — extractor needs `Observation:` blocking"). If <18/20 intended, tighten extractor and re-audit.
- **Old gate `lenient_EX ≥ strict_EX` is removed** — strict was 0 by selection, so the old gate trivially passed.
- Wall: <1 day. Cost: <1 CPU/IO-h (no GPU; SQLite + sqlglot only).

**Day 3 (Wed): A1 — Full lenient re-execution on Spider-dev**
- Apply `lenient_extract` to all SFT and Dense75 records (1034 each).
- Re-execute lenient_pred_sql against `db_path` for each record; cache results.
- Outputs (absolute paths):
  - `/mnt/data2/wuql/miniLLM/outputs/lenient_eval_sft_full_dev.json`
  - `/mnt/data2/wuql/miniLLM/outputs/lenient_eval_dense75_full_dev.json`
- Cost: ~6 **CPU/IO-h** (R5 reviewer fix — SQLite-bound + sqlglot, no GPU rollout); single-threaded baseline; can multi-process to ~1h wall via `multiprocessing.Pool`.
- Wall: <1 day.

**Day 4 (Thu): A2 — Per-record classifier + transition table**
- Run `scripts/diag_protocol_vs_semantic.py` on (SFT, Dense75) pair.
- Outputs:
  - per-record `(bucket, flags)` JSON
  - aggregate table: `{lenient_attribution_gap, attribution_ratio, transition_bucket_share[*], global_bucket_rate[*]}` with bootstrap CIs
  - `observation_repair_share` subanalysis
- Cost: 0 GPU-h (CPU only).
- **Inspection point**: confirm `semantic_gain` ≥ 10 records (R2). If not, document and proceed with reduced confidence for C2.
- Wall: <1 day.

**Day 5 (Fri): A3 — Bootstrap CIs + sanity write-up**
- Run 1000-resample bootstrap on every headline number.
- Verify all CIs are computed and stored.
- Draft Claim 1 paragraph + 1 figure (the strict-vs-lenient decomposition table).
- Wall: <0.5 day. Buffer for iteration.

**End-of-Week 1 deliverable**: Claim 1 (C1) is empirically resolved one way or the other. Headline number `lenient_attribution_gap` is reported with its CI. Story direction (protocol-dominant vs semantic-dominant) is locked.

### Week 2 — Supporting SKE mechanism (R5 fix: ~12 CPU/IO-h mandatory; +28 GPU-h conditional)

**Inspection gate at end of Week 1**: only proceed to B4 (Dense G=8 confound) if C1 supports the protocol-dominant thesis. B1 + B2 are unconditional (both ~12 CPU/IO-h, no GPU). B3 (E3) is **dropped** as a primary block — the per-`skeleton_class` advantage-shift logs it needs were never persisted. Tier-3 descriptive analysis is folded into B2.

**Day 6 (Mon): B1 — Lenient pass for SKE arms**
- Apply A1 pipeline to SKE-G4-75 and SKE-G8-75 records (1034 each).
- Outputs (absolute):
  - `/mnt/data2/wuql/miniLLM/outputs/lenient_eval_ske75_full_dev.json`
  - `/mnt/data2/wuql/miniLLM/outputs/lenient_eval_ske_g8_full_dev.json`
- Cost: ~12 **CPU/IO-h** (R5 fix; SQLite + sqlglot only).
- Wall: <1 day.

**Day 7 (Tue): B2 — Apply classifier to SKE pairs + Tier-3 descriptive E3**
- Run `diag_ske_mechanism.py` to classify (SKE-G4 vs Dense75) and (SKE-G8 vs Dense75).
- Outputs: per-pair `(bucket_share, lenient_EX)` tables with bootstrap CIs.
- Compute **E1** (semantic_gain bucket share comparison) and **E2** (lenient_EX comparison).
- **Tier-3 descriptive E3 (always reported, NOT a pass condition)**: stratify dev records by outcome density, then for each stratum compute the *aggregate* SKE training-time class statistics that ARE persisted (`ske_n_real_classes_sum / global_step`, `ske_used_steps / global_step`, `ske_fallback_reasons` distribution). Report whether the two strata's `skeleton_class`-frequency distributions match (necessary-but-insufficient check). Document explicitly that this is descriptive context, not equivalence evidence.
- Cost: 0 GPU-h.
- Wall: <1 day.

**End-of-Week 1+B2 inspection point** (after C1 result + E1 + E2): decide whether to launch B4.

**Day 8-10 (Wed–Fri, GATED): B4 — Dense G=8 confound run + eval + classifier**
- **Launch condition (R5 reviewer fix)**: only run if BOTH:
  - C1 PASS (`lenient_attribution_gap > 0`, 95% CI excluding 0)
  - E1 PASS (SKE-G4 `bucket_share[semantic_gain]` ≤ Dense `bucket_share[semantic_gain]`)
  Otherwise: SKE-specific story has already been weakened by C1 inversion or E1 failure; ~28 GPU-h is better spent on writing.
- Launch Dense G=8 GRPO training: same launcher as Dense G=4, MAX_STEPS=75, NUM_GEN=8, no SKE.
  ```bash
  cd /mnt/data2/wuql/miniLLM && \
  RUN_DIR=outputs/grpo-dense-7b-g8 \
  NUM_GEN=8 MAX_STEPS=75 SAVE_STEPS=25 \
  bash scripts/train_grpo.sh
  ```
- Cost: ~24 GPU-h training + ~5 GPU-h Spider-dev eval = ~28 GPU-h (~1.5 days wall).
- After eval, run lenient pass + classifier vs SFT.
- **Pass condition**: `EX_att(Dense G=8) ≥ EX_att(Dense G=4) − 1pp` within bootstrap CI. If FAIL, story shifts to "G effect is generic, not SKE-specific" — documented as failure-mode fallback in `FINAL_PROPOSAL.md`.

**End-of-Week 2 deliverable**: Claim 2 resolved on E1 + E2 (mandatory) + Dense G=8 confound (conditional). E3 status reported as "input data not persisted; cannot test" — honest limitation.

### Week 3 — Writing + figures + buffer

**Days 11-13**: Draft 4-page workshop paper. Section structure:
1. Introduction (1/2 page)
2. Setup (1/2 page) — ReAct + GRPO + Spider-dev + checkpoints
3. Methodology — the diagnostic protocol (1 page) with classifier interface boxout
4. Finding 1 — Dense GRPO gain decomposition (1 page) — Table 1 (headline numbers with CIs) + Figure 1 (bucket distribution)
5. Finding 2 — SKE mechanism (1/2 page) — E1+E2+E3 result + Dense G=8 confound row
6. Discussion + limitations (1/2 page)
+ Appendix: M1-M4 mechanism checklist; full per-record tables; (if found) external-baseline probe

**Day 14 (optional buffer)**: 3B Dense same-launcher rerun (~12 GPU-h) for appendix context, OR external-baseline probe if any HES-SQL/MTIR-SQL/MARS-SQL public per-record artifact exists.

**Days 15-21**: Polish + figure refinement + venue selection (NeurIPS 2026 workshops / TRL workshop / RL-for-LLMs workshop) + submission prep.

## Total Budget (R5 reviewer fix: GPU vs CPU/IO accounting)

| Resource | Used | Available |
|---|---:|---:|
| **GPU-h (mandatory)** | **0** (no GPU work in C1 or B1+B2) | ~360 |
| **GPU-h (B4 conditional, gated on C1 pass)** | ~28 | ~360 |
| **CPU/IO-h (mandatory; SQLite + sqlglot re-execution)** | ~18 | unbounded |
| **GPU-h (with optional 3B + external probe appendix)** | ~12 + ~2 | ~360 |
| New code LOC | ~660 | n/a |
| New unit tests | ~22 | n/a |
| Wall time (Week 1-3) | 15 days | 21 days |

Buffer: ~5 days unallocated for iteration, sanity-check rework, or unforeseen issues. **No reliance on multi-seed, BIRD GRPO, or third-capacity-point runs** — those are explicitly out of scope per `FINAL_PROPOSAL.md`.

## Decision Tree at Each Inspection Point

```
End of Day 2 (A0 sanity)
  ├─ parseable_lenient_rate ≥ 80% AND ≥18/20 manual audit intended → PROCEED to A1
  ├─ parseable rate < 80%                                          → tighten extractor; rerun A0
  └─ audit shows extractor catching Obs/non-final SQL              → tighten Answer-region anchor;
                                                                       re-audit; block until PASS

End of Day 4 (A2 inspection)
  ├─ semantic_gain ≥ 10 transitions in (Dense vs SFT) → PROCEED to Week 1 finish
  └─ semantic_gain < 10                                → flag; reduce C2 contrast confidence;
                                                          C1 stands; still proceed

End of Day 5 (A3 — Claim 1 result; KEYSTONE GATE for Dense G=8)
  ├─ lenient_attribution_gap > 0, CI excludes 0   → C1 PASS protocol-dominant thesis
  │                                                  → schedule B4 (Dense G=8) gated on E1
  ├─ lenient_attribution_gap ≈ 0, CI includes 0   → C1 INVERTS to semantic-dominant; reframe
  │                                                  → SKIP B4; reallocate 28 GPU-h to writing
  └─ lenient_attribution_gap < 0, CI excludes 0   → ANOMALY; investigate extractor; SKIP B4

End of Day 7 (B2 — E1+E2 result)
  ├─ E1 AND E2 both hold                  → C2 PASS (E3 reported as Tier-3 descriptive)
  │                                          → if C1 also PASS, launch B4 (Day 8-10)
  ├─ E1 OR E2 fails                       → C2 FAIL; SKE story is "we cannot demonstrate
  │                                            the protocol-bottleneck mechanism cleanly";
  │                                            SKIP B4; honest reporting
  └─ Both fail                            → SKE story weakened to engineering complaint;
                                              SKIP B4; reduce SKE section to a paragraph

End of Day 10 (B4 — Dense G=8 result, only if launched)
  ├─ EX_att(G=8) ≥ EX_att(G=4) − 1pp within CI  → SKE-specific G-degradation story holds
  └─ EX_att(G=8) < EX_att(G=4) − 1pp            → Generic G effect; reframe as
                                                    "rollout-budget tradeoff"
```

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Lenient extractor too permissive (catches Observation/non-final SQL) | Medium | High — biases protocol-attribution upward | Strengthened A0 gate (R5): parseable_rate ≥ 80% **AND** 20-record manual audit |
| `semantic_gain` <10 records | Medium | High for C2 contrast | Honest CI reporting; report effect-size with full uncertainty; don't overclaim |
| Dense G=8 also degrades | Low-Medium | Medium (no longer keystone — gated on C1) | Documented reframe to "rollout-budget tradeoff"; only launch if C1 passes |
| ~~E3 inconclusive~~ | — | — | E3 dropped from C2 pass condition (R5 fix) — input data not persisted |
| E1 / E2 fails | Medium | High for SKE narrative | C2 reduced to short paragraph; main paper still rests on C1 |
| C1 inverts to semantic-dominant | Medium | Medium — story reframes, doesn't die | See "Branching Plan If C1 Inverts" below |
| GPU contention with concurrent experiments | Low | Medium (only if B4 launches) | Coordinate left-pane experiments before Day 8 |
| Workshop deadline shift | Low-Medium | High | Track venue deadlines from Day 1; have backup venue |

## Branching Plan If C1 Inverts (semantic-dominant)

If `lenient_attribution_gap` is ~0 with CI including 0, the protocol-dominant thesis fails. The diagnostic itself is still novel — but the story changes:

- **New thesis**: "Agentic GRPO does deliver semantic SQL improvements; structural priors like SKE-RL still fail because they cannot target the right subset of semantically-improvable queries."
- **C2 reframes**: SKE doesn't help because semantic_gain is concentrated in queries with high within-skeleton-class outcome variance — class baseline gives no information about who in the class will repair. (M3 candidate from the appendix M1-M4 checklist.)
- **Workshop venue unchanged**.
- **B4 (Dense G=8) is SKIPPED** — 28 GPU-h reallocated to writing + 1.5B Dense appendix or external probe.

If `lenient_attribution_gap` is negative with CI excluding 0 (lenient gain > strict gain), the story has a new angle: lenient view exposes semantic gain that strict's protocol-failure cliff was hiding. Genuinely novel diagnostic. Reframe in writing phase. **B4 SKIPPED.**

## Out of Scope (do NOT do, per FINAL_PROPOSAL non-goals)

- Multi-seed averaging of any condition.
- BIRD GRPO training run.
- 1.5B or 14B capacity points.
- v2 SKE (skeleton-first generation).
- Reproducing HES-SQL or any other external baseline (only consume their public artifacts if available).
- Adding new RL methods, new reward variants, or new advantage estimators.
- Toolkit-as-lead packaging.

These are documented as non-goals in the anchor. Adding any would require a new round of `/research-refine`.
