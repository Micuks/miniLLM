---
date: 2026-04-24
revision: R4 (post external-review refinement; addresses 3 High + 3 Medium critiques)
status: SELECTED — staged v1/v2/v3, only v1 committed initially
problem_anchor: |
  Multi-turn ReAct GRPO at 3B on Text-to-SQL fails on medium queries (-18 to
  -28pp vs SFT). Five reward/sampling interventions (dense, sparse, CGFR, RVDS,
  CGFR+RVDS) failed identically. Reward-shape invariance suggests the bottleneck
  has a structural component, but does NOT alone rule out 3B capacity. SKE-RL
  is therefore framed as a *necessary credit-assignment fix*, not a sufficient
  cure for capacity-bound failure.
method_thesis: |
  Replace GRPO's per-rollout advantage normalization with advantage normalization
  over discrete program-equivalence classes (canonical α-renamed AST templates).
  No change to action space, prompts, decoding, or reward in v1.
prior_art_status: |
  R3 reviewer scored core mechanism 7/10. Existing sql_structure_reward in repo
  is token/clause overlap (not AST equivalence) and operates on reward, not
  advantage. SKE-RL's contribution is precisely the advantage-sharing-over-
  canonical-AST-classes mechanism, not structural reward shaping.
---

# SKE-RL — Skeleton-Class-Aware Advantage Estimation for Multi-Turn Text-to-SQL GRPO

## 0. What Changed in R4 vs R3

External reviewer flagged 3 High + 3 Medium issues. Refinement summary:

| R3 Issue | R4 Resolution |
|---|---|
| **H-1** "unit of optimization" claim too strong; doesn't rule out capacity | Reframed as *necessary not sufficient*. SKE-RL works only if rollouts produce ≥2 distinct skeleton classes per group with non-trivial within-class variance. Phase 1.5 gate makes this explicit. |
| **H-2** Mixed two algorithms (post-hoc extraction vs skeleton-first generation) | Staged into v1 (post-hoc only), v2 (+ fill search), v3 (+ skeleton-first). v1 commits no changes to action space. |
| **H-3** R_struct = max-over-fillings requires inverse rendering | Removed from v1. R_struct in v1 = same as R_fill (binary execution match of the rollout's own SQL). max-over-fillings deferred to v2 with explicit feasibility prototype before committing. |
| **M-4** Class-shared advantage degenerates with G=8 small groups | Explicit fallback: when distinct skeleton classes in group < threshold (default **3**, raised from 2 per R4 review), use standard GRPO advantage. Class baseline is **size-weighted** (not equal-weighted) to prevent singleton inflation. Fallback **reduces exposure on low-diversity steps**, but does NOT bound harm on SKE-used steps where miscalibrated classes can still produce wrong gradients. Skeleton entropy is logged but not added to loss in v1. |
| **M-5** sqlglot not in deps | Explicitly added to dependency list. Estimated +1 day for dep + integration. |
| **M-6** Novelty must be sharper vs existing sql_structure_reward | Contribution narrowed to: "*advantage normalization over canonical AST equivalence classes* — orthogonal to reward shape; no change to reward function in v1." |

## 1. Problem & Anchor (Refined)

**Anchor (do not drift):** at 3B, multi-turn ReAct GRPO over raw SQL strings fails on medium queries regardless of five reward variants tried. This invariance is *consistent with* a credit-assignment problem (lexical noise dominates structural signal), but is *also consistent with* a capacity bottleneck (3B can't sample structurally-correct medium programs at all). These are not mutually exclusive.

SKE-RL targets the credit-assignment hypothesis. **Falsifiability gate**: if 3B rollouts at our current SFT/GRPO temperature rarely produce a **gold-shape (structurally plausible) skeleton class** for medium queries, SKE-RL has no signal to share — capacity is then the binding constraint and SKE-RL is dead. Phase 1.5 measures this directly before committing to training. Note: "gold-shape" means σ-equivalent to σ(gold), which captures arity/types/JOIN structure but NOT table-set; we report the stricter "gold-with-table-set" rate alongside as a sanity check.

We do NOT claim SKE-RL solves capacity. We claim:
1. If the credit-assignment hypothesis is wrong, SKE-RL fails the Phase 1.5 gate and we know
2. If the gate passes and SKE-RL improves medium EX, the credit-assignment mechanism is real
3. If the gate passes but SKE-RL doesn't improve eval, the credit-assignment mechanism exists but is not the binding bottleneck → capacity is

All three outcomes are publishable findings.

## 2. Method (v1 only — minimum viable)

### 2.1 Skeleton extractor σ: SQL → canonical key (or None)

**Implementation**: use `sqlglot` (NEW dependency, see §6.1) for AST parsing.

```python
def extract_skeleton(sql: str, schema_columns: dict[str, list[(str, str)]]) -> str | None:
    """Returns canonical S-expression string, or None if parse fails / pattern unsupported."""
    # 1. parse sqlglot.parse_one(sql, dialect="sqlite")
    # 2. walk tree, replace:
    #    - column refs → "col_<sqlite_type>" via schema lookup; unknown → "col_unk"
    #    - table refs → "tbl_<n_cols>"; unknown → "tbl_unk"
    #    - literals → "?<inferred_type>"  (str/int/float/date/null)
    # 3. preserve: top-level op, JOIN type, aggregate fn signatures, ORDER/GROUP arity, set ops
    # 4. emit canonical S-expr; alphabetize within commutative slots (AND/OR operands)
    # 5. return string. Hashing happens at the call site.
```

Two extractor variants for analysis (only one used per training run):

- **σ_strict**: above; strict typed slots; subqueries kept as `(subq ...)` recursively
- **σ_loose**: numeric types collapsed to `col_num`; subqueries flattened to `?subq`; HAVING merged with WHERE

### 2.2 v1 algorithm — Post-hoc Class-Aware Advantage (NO action-space change)

Standard GRPO is unchanged in: prompts, decoding, action space, reward function. **Only the advantage computation changes.**

```python
def compute_class_aware_advantage(rewards, completions, schema):
    # Step 1: extract skeleton class for each rollout's final SQL
    classes = [extract_skeleton(extract_final_sql(c.gen_text), schema) for c in completions]
    # None means parse failure; treated as a singleton "fallback" class
    classes = [c if c is not None else f"_fallback_{i}" for i, c in enumerate(classes)]

    # Step 2: count distinct classes
    distinct_classes = set(classes)
    
    # Step 3: degeneracy fallback — if too few classes, use standard GRPO.
    # Default raised to 3 (R4 reviewer fix M-4): with 2 classes split 7+1,
    # singleton outlier rewards inflate equal-weighted class baseline. Min 3
    # makes singleton dominance arithmetically harder.
    if len(distinct_classes) < args.ske_min_classes:  # default 3
        return standard_advantage(rewards), {"ske_used": False, ...}

    # Step 4: per-class mean reward + class size
    class_to_indices = defaultdict(list)
    for i, c in enumerate(classes):
        class_to_indices[c].append(i)
    
    class_means = {c: torch.tensor([rewards[i] for i in idxs]).mean().item()
                   for c, idxs in class_to_indices.items()}
    class_sizes = {c: len(idxs) for c, idxs in class_to_indices.items()}
    
    # Step 5: SIZE-WEIGHTED class baseline (R4 reviewer fix M-4 singleton trap).
    # Weighting by class size means a 7-rollout dominant class shapes the
    # baseline more than a singleton outlier, preventing baseline distortion.
    total_size = sum(class_sizes.values())
    class_baseline = sum(class_means[c] * class_sizes[c] for c in class_means) / total_size
    # std also size-weighted for consistency
    class_var = sum(class_sizes[c] * (class_means[c] - class_baseline)**2 for c in class_means) / total_size
    class_std = max(class_var ** 0.5, 1e-6)
    
    # Step 6: per-rollout advantage = class-level advantage + within-class deviation
    advantages = []
    for i, c in enumerate(classes):
        class_adv = (class_means[c] - class_baseline) / class_std
        within_class_dev = rewards[i] - class_means[c]
        # within-class normalized by intra-class std (or 1.0 if singleton)
        members = class_to_indices[c]
        if len(members) > 1:
            intra_std = max(torch.tensor([rewards[j] for j in members]).std().item(), 1e-6)
            within_adv = within_class_dev / intra_std
        else:
            within_adv = 0.0
        advantages.append(args.ske_beta * class_adv + (1 - args.ske_beta) * within_adv)
    return advantages, {"ske_used": True, "n_classes": len(distinct_classes), ...}
```

**v1 properties:**
- 0 extra rollouts (uses existing G=8)
- 0 changes to decoding, prompts, action space, reward function
- ~150 LOC: extractor (~80), advantage replacement in train_grpo (~40), CLI args + logging (~30)
- New dependency: `sqlglot` (well-maintained, MIT, no GPU)

### 2.3 What v1 explicitly DOES NOT include (deferred to v2/v3)

- ❌ Skeleton-first generation (changes action space — moved to v3 if v1 succeeds)
- ❌ max-over-fillings R_struct (requires inverse rendering — moved to v2 with feasibility prototype gate)
- ❌ Skeleton entropy bonus on the loss (logged but not in objective in v1; moved to v2)
- ❌ Constrained decoding (orthogonal direction; not bundled)
- ❌ Multiple extractors at training time (pick one σ per run; compare across runs)

## 3. Three Hypotheses Mapped to v1 Outcomes

| v1 outcome on Spider 200 | Hypothesis allowed |
|---|---|
| Phase 1.5 gate fails (no class diversity) | Capacity hypothesis confirmed; SKE-RL irrelevant |
| Gate passes, v1 EX_att ≥ baseline + 3pp on medium | Credit-assignment mechanism real; pursue v2/v3 |
| Gate passes, v1 EX_att ≈ baseline | Mechanism exists but isn't binding; capacity also matters; reframe paper as "two-factor analysis" |
| Gate passes, v1 EX_att < baseline | Class-aware advantage *hurts*; document failure mode (likely degeneracy or coarse equivalence) |

Each outcome is publishable. The paper writes itself differently in each branch.

## 4. v2 / v3 Roadmap (Conditional)

### v2: + Within-Class Fill Search (only if v1 medium Δ ≥ +3pp)

Add R_struct = "any feasible fill works":
- **Feasibility prototype required first**: write a script that, given a skeleton and a schema, samples K=4 candidate literal fills via constrained decoding from the SFT model and checks executability. If <70% of skeletons admit at least one parseable fill within K=8 attempts → kill v2.
- This prototype is an offline analysis on Spider train; ~1 day of work, no GRPO involvement.
- Only after the prototype passes, modify advantage to use `R_struct(z) = max over K fills of execution match`.

### v3: + Skeleton-First Generation (only if v2 medium Δ ≥ +5pp)

Add a structured generation phase:
- New action space: emit `<skeleton>` tag first, then `<fill>` tag with literals
- Constrained decoding for skeleton phase (grammar-bound)
- Modify ReAct prompt template, generation loop, parsing
- This is a real ~500 LOC change including prompt template changes, parser, and constrained decoder integration.
- Estimated 3-5 days; do NOT undertake unless v2 strongly motivates it.

The v3 staging directly addresses reviewer's H-2 critique: skeleton-first generation is treated as its own algorithmic step, not bundled into v1.

## 5. Reviewer Critiques Addressed (Detail)

### H-1: "Causal diagnosis under-supported"

**Refinement**: §1 explicitly states SKE-RL doesn't claim to rule out capacity. The Phase 1.5 gate (skeleton diversity in actual rollouts) is the falsification mechanism. If 3B rarely samples correct medium-query skeletons, the gate fails and we have *empirical evidence* for capacity rather than credit-assignment.

This makes the paper a *two-factor decomposition*, not a single-mechanism claim.

### H-2: "Mixing two algorithms"

**Refinement**: v1 is purely post-hoc class-aware advantage; same prompts, same decoding, same action space, same reward. The 4-way factorial in §6.2 of the original plan is rewritten in §7 as a **3-way staged factorial** that does not include skeleton-first generation in v1.

### H-3: "max-over-fillings core feasibility risk"

**Refinement**: dropped from v1 entirely. R_struct in v1 = R_fill = standard execution-match reward. v2 adds the max-over-fillings only after a separate offline feasibility prototype passes a 70% bar.

### M-4: "Class-shared advantage degenerates with G=8"

**Refinement**: explicit fallback to standard GRPO when distinct classes in group < `--ske-min-classes` (default **3**, raised from 2 per R4 reviewer). Class baseline is **size-weighted** (not equal-weighted across classes) to prevent singleton outliers from inflating the baseline. Logged: per-step `n_classes`, fallback rate. We expect 30-50% of steps to fall back early in training; acceptable as long as SOME steps benefit. Skeleton entropy bonus is logged but NOT added to loss in v1.

**In-training kill switch (R4 addition)**: monitor `ske_used` rate over rolling 20-step window. If after the first 30 warm-up steps `ske_used` drops below 20% sustained for 20 consecutive steps, the mechanism is not firing → halt training and either fall back entirely to standard GRPO for the remaining steps or abort for diagnosis. This formalizes H-1's residual concern: SFT-time diversity (Phase 1.5 gate) doesn't guarantee in-training diversity.

### M-5: "sqlglot not in deps"

**Refinement**: explicitly added. Implementation milestone: `uv add sqlglot` + smoke test on 50 Spider train queries before any extractor work begins.

### M-6: "Existing structural reward exists"

**Refinement**: explicit differentiation in §0:
- `sql_structure_reward` (existing, in repo) = **token/clause overlap reward**, contributes to scalar per-rollout reward
- SKE-RL v1 = **canonical-AST-equivalence-class advantage normalization**, operates on the GRPO advantage step *after* rewards are computed

These are orthogonal mechanisms. v1 leaves `sql_structure_reward` and the entire reward function untouched. The novelty is at the advantage-estimation step, not at the reward.

## 6. Implementation Plan (v1 Scope)

### 6.1 Dependencies

Add to `pyproject.toml`:
```toml
[project]
dependencies = [
    # ... existing ...
    "sqlglot>=21.0,<26.0",  # SQL AST parsing for SKE-RL
]
```

Run: `uv add sqlglot` then `uv lock --upgrade-package sqlglot`.

### 6.2 New files

| Path | Purpose | LOC est |
|---|---|---|
| `miniLLM/skeleton.py` | σ_strict + σ_loose extractors + canonical S-expr emitter | ~250 |
| `tests/test_skeleton.py` | 30+ unit tests covering Spider patterns + 10 BIRD-specific | ~200 |
| `scripts/analyze_skeleton_coverage.py` | Phase 1 offline diagnostic | ~80 |
| `scripts/analyze_within_class_variance.py` | Phase 1.5 diagnostic | ~60 |
| `scripts/train_ske_rl.sh` | launcher | ~70 |

### 6.3 Modified files

| Path | Change | LOC est |
|---|---|---|
| `miniLLM/train_grpo.py` | add `--use-ske-rl`, `--ske-extractor`, `--ske-beta`, `--ske-min-classes`. Replace advantage normalization step (around line 891) with `compute_class_aware_advantage` when flag set. Add per-step logging of `n_classes`, `ske_used`, `ske_fallback_rate`. | ~80 |
| `pyproject.toml` | add sqlglot dep | 1 |

**Total NEW code v1: ~660 LOC** (260 prod + 200 test + 200 script). Realistic estimate: **2.5-3 days clean implementation**.

### 6.4 Implementation milestones (v1)

| Day | Deliverable | Gate |
|---|---|---|
| Day 0 | sqlglot installed, smoke test on 50 SQLs | parses ≥98% |
| Day 1 | `skeleton.py` σ_strict + σ_loose; unit tests pass | tests green |
| Day 2 | Phase 1 coverage analysis script run on Spider/BIRD train+dev | report generated |
| Day 2 end | **Phase 1 GO/NO-GO**: see §7.1 | proceed if pass |
| Day 3 | `compute_class_aware_advantage` integrated into train_grpo.py; smoke pilot 25 steps | loss not NaN, n_classes logging works |

## 7. Experiments (v1 Only)

### 7.1 Phase 1: Diagnostics (1.1 = 0 GPU; 1.2 = ~5 GPU-h after R5 fix)

**Run 1.1 Coverage**:
```
python scripts/analyze_skeleton_coverage.py \
    --datasets spider_train spider_dev bird_train bird_dev \
    --extractors strict loose
```
Reports: coverage rate, distinct skeleton count, top-20 most common skeletons, fallback examples.

**Gate 1.1**: σ_strict coverage ≥85% Spider train; σ_loose ≥95%. If <50% on BIRD strict → only loose for BIRD.

**Run 1.2 Within-class outcome variance — over MODEL ROLLOUTS, NOT gold** (R5 reviewer fix):

The original formulation `σ²(gold_executes_correctly | z)` was meaningless — gold SQL by definition executes correctly, so variance is ~0 and the gate trivially passes. Replaced with:

```
python scripts/analyze_within_class_variance.py \
    --dataset spider_train --extractor strict --min-class-size 3 \
    --adapter outputs/react-sft-v2 \
    --rollouts-per-query 4 --temperature 0.7
```

For each skeleton class z with ≥3 distinct training queries:
- For each query q in z: sample K=4 SFT rollouts at training temperature; compute fraction of rollouts that execute correctly = `p_q`
- Within-class variance = `Var_{q in z}(p_q)` — i.e., do queries that share a skeleton have similar SFT-rollout success rates?

**Gate 1.2 (hard)**: median within-class variance of `p_q` ≤ 0.15. **Interpretation**: if low, queries in the same skeleton class are similarly hard for SFT (equivalence is meaningful). If high, the skeleton hides queries of wildly different inherent difficulty (equivalence too coarse → tighten extractor or abort).

**Gate 1.2-supplementary (logged, not gated, R5 reviewer addition)**: also report distribution of `mean_q(p_q)` per class AND fraction of classes with `mean_q(p_q) > 0` (i.e., at least one query in the class is sometimes solvable by SFT). **Why**: low Var(p_q) can mean either (a) genuine equivalence — queries similarly solvable — OR (b) all queries uniformly impossible (Var ~0 trivially). The supplementary metric distinguishes these cases. We expect ≥40% of classes to have nonzero mean success; if much lower, equivalence may be technically clean but practically uninformative.

### 7.2 Phase 1.5: **Skeleton diversity in actual GRPO rollouts** (NEW — addresses H-1)

**Run 1.5.1**: with the existing SFT adapter, generate G=8 rollouts at training temperature (0.7) for 50 random Spider train prompts. Extract skeleton for each rollout's final SQL. Report:
- distinct classes per group: distribution
- fraction of groups with ≥2 distinct classes
- fraction of groups where ANY rollout produced the **gold-shape skeleton** (matching σ(gold) — i.e., same arity, types, JOIN structure, aggregate signatures, BUT NOT same table-set or column-roles)
- fraction with **gold-with-table-set skeleton** (stricter: same as above + table-set matches gold's via name normalization). This stricter metric is reported alongside but NOT the gate; it's a tighter sanity check on whether "matching gold-shape" actually corresponds to structural correctness.
- per-difficulty breakdown (easy / medium / hard)

**Gate 1.5 (calibration tightened per R4 reviewer second-pass)**:
- **G1.5-A (diversity)**: ≥40% of groups have ≥2 distinct skeleton classes **AND** inter-class reward variance (mean class-mean spread) ≥ 0.1 — variance check rules out "diverse failures" (multiple classes all at reward ~0) which gives SKE-RL no signal to amplify
- **G1.5-B (positive signal, hard floor)**: for medium queries, ≥15% of groups contain at least one rollout matching the **gold-shape skeleton** (σ-equivalent to gold) — minimum hard gate. **Reported alongside**: stricter "gold-with-table-set" rate as sanity check on whether shape-match correlates with structural correctness
- **G1.5-B' (positive signal, soft target)**: for medium queries, ≥30% gold-containing groups is soft target. If actual rate falls in 15-30%, paper claims must be qualified ("SKE-RL mostly sorts among wrong skeletons; gains likely from variance reduction over wrong approaches, not gold-class amplification")

**This gate is the key falsifier added in R4**. If G1.5-A or G1.5-B fails for medium queries, SKE-RL is dead before any training — write up capacity-confirmation paper.

### 7.3 Phase 2: Smoke pilot (5 GPU-h)

```
USE_SKE_RL=1 SKE_EXTRACTOR=strict SKE_BETA=0.7 SKE_MIN_CLASSES=3 \
MAX_STEPS=25 bash scripts/train_ske_rl.sh
```

Diagnostic: training loss non-NaN; logged `n_classes` averages reasonable; `ske_fallback_rate` < 70%.

### 7.4 Phase 3: 3-Way Comparison (15 GPU-h)

| Condition | Description |
|---|---|
| **A1: raw GRPO baseline** | existing CGFR config, 100 steps |
| **A2: SKE-RL v1 σ_strict** | post-hoc class advantage, β=0.7 |
| **A3: SKE-RL v1 σ_loose** | same but loose extractor |

Compute matched. Eval Spider 200 at ck25/50/75/100.

**Gate 3 (decisive for v1)**:
- A2 OR A3 medium EX_att ≥ A1 + 3pp → go to v2
- A2/A3 not worse than A1 by >3pp on overall EX_att → no harm; consider v2 anyway as exploration
- A2 AND A3 both worse than A1 by >3pp → v1 hurts; investigate degeneracy; likely abort

### 7.5 Phase 4: BIRD validation (8 GPU-h, conditional on Phase 3 GO)

Same 3-way on BIRD-mini. Reports trade-off (σ_strict vs σ_loose) on harder query distribution. This is itself a contribution — empirical mapping of equivalence-purity-vs-coverage.

### 7.6 Phase 5: Multi-seed (only if Phase 3 GO + Phase 4 reasonable)

Top condition × 3 seeds × Spider+BIRD. ~12 GPU-h.

### 7.7 v1 total budget

| Phase | GPU-h |
|---|---|
| 1.1 (coverage, no GPU) | 0 |
| 1.2 (within-class variance via SFT rollouts — R5 fix) | 5 |
| 1.5 (skeleton diversity at SFT temp) | 3 |
| 2 (smoke) | 5 |
| 3 (3-way Spider) | 15 |
| 4 (BIRD, conditional) | 8 |
| 5 (multi-seed, conditional) | 12 |
| Reserve | 5 |
| **Total v1** | **~53 GPU-h** |

Down from R3's 70. R5 reviewer note: Phase 1.2 now uses model rollouts (not gold), adding 5 GPU-h. Numbers consistent across FINAL_PROPOSAL, EXPERIMENT_PLAN, EXPERIMENT_TRACKER.

### 7.8 v2 / v3 budgets (only if v1 motivates)

- v2 prototype + factorial: +25 GPU-h
- v3 prototype + factorial: +40 GPU-h
- Optional 7B anchor on v1 winning condition: +20 GPU-h

## 8. Paper Story (R4)

**Working title**: "Skeleton-Class-Aware Advantage Estimation for Multi-Turn Text-to-SQL GRPO at Constrained Scale: A Two-Factor Analysis"

**Narrative arc** (refined to address H-1):
1. We show GRPO at 3B fails on medium queries across five reward variants
2. Two hypotheses remain: capacity, or credit-assignment mismatch
3. We design SKE-RL to test the credit-assignment hypothesis directly
4. Phase 1.5 (skeleton diversity in rollouts) **separately measures the capacity binding**
5. v1 results give us a 4-cell decision matrix:
   - gate fails → capacity
   - gate passes + improvement → credit
   - gate passes + no improvement → both, in different proportions
   - gate passes + harm → mechanism exists but interaction is bad
6. Whichever cell, we publish

**Two contributions** (refined per M-6):
- **Methodological**: advantage normalization over canonical AST equivalence classes (orthogonal to reward shape)
- **Diagnostic**: empirical decomposition of capacity vs credit-assignment in 3B multi-turn SQL agents

**Acknowledged limitation (R4 reviewer note)**: Phase 3 keeps `sql_structure_reward` (the existing token/clause overlap reward) on for all conditions A1/A2/A3. This isolates the advantage-normalization change but does NOT answer "does SKE-RL still help when `sql_structure_reward` is disabled?" Future work — note explicitly in paper.

**Target venue**: ACL 2027 main / NeurIPS 2026 main. ACL Findings as backup.

## 9. Risk Register (Updated)

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| sqlglot can't parse some Spider/BIRD SQL | Medium | Low | Fall back to surface-string class for unparseable; report fallback rate |
| Phase 1.5 fails for medium queries | High | Method-defining | Gate is the point — if it fails, paper becomes capacity story instead |
| Class diversity too low at G=8 → mostly fallback | High | Medium | Fallback **reduces exposure on those specific low-diversity steps** by reverting to standard GRPO. Does NOT prevent miscalibrated equivalence on SKE-used steps. Report `ske_used` rate; if very low, the experiment is effectively standard GRPO + tiny SKE perturbation |
| σ_strict coverage <85% Spider | Medium | Medium | Use σ_loose; report trade-off |
| v1 hurts performance | Medium | High | Standard-advantage fallback **only reduces exposure on low-diversity steps** (when class count < min_classes). On SKE-used steps, miscalibrated equivalence classes can still produce systematically wrong gradients. Mitigate via: (a) in-training kill switch on `ske_used` rolling rate, (b) early-checkpoint eval to catch divergence before step 100, (c) Phase 1.2 within-class-variance gate as upstream defense. Fallback is exposure-reduction, not harm-bound. |
| 7B SFT succeeds → v1 less interesting | Low | Low | v1 is still a valid 3B finding |

## 10. v1 Off-Ramps (Decision Discipline)

1. **Day 2 end**: if Gate 1.1 or 1.2 fails → abort SKE-RL entirely; document and pivot
2. **Day 3 end**: if Gate 1.5 fails for medium → write up capacity-confirmation paper instead
3. **Phase 3**: if Gate 3 fails → don't pursue v2; document v1 negative result
4. **Phase 4**: if BIRD fails badly across both σ → restrict claims to Spider

These off-ramps are the difference between "method validation" and "publication-bias confirmation". Bake them in.

## 11. Composition

v1 is alone; no GRM/PORG/ATE/EPR-X composition.

If v1 succeeds and v2 fails to add value, EPR-X (R3 backup, 6/10) becomes the secondary intervention worth trying.
