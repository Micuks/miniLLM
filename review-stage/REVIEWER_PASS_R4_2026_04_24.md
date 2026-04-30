---
date: 2026-04-24
revision: R4 review pass
reviewer: external (second pass)
scope: addresses all 6 original critiques; evaluates Phase 1.5 gate calibration
---

# SKE-RL R4 — Second Reviewer Pass

## H-1: Causal Diagnosis Under-Supported

**Verdict: PARTIAL — substantially improved, one residual gap**

What changed: the "unit of optimization" claim has been explicitly softened to "necessary not sufficient." The proposal now frames capacity and credit-assignment as non-exclusive and lists all four Phase 1.5 / Phase 3 outcome cells as publishable findings. This is the right epistemic posture.

What is sound: the two-factor framing is clean. The 4-cell decision matrix (gate fail / gate pass + improvement / gate pass + flat / gate pass + harm) gives the experiment genuine falsifiability. The paper no longer rests on a single mechanistic claim.

Residual gap: Phase 1.5 uses the *SFT* adapter to measure rollout diversity, not the *GRPO-checkpoint* adapter that will actually be running at training step 1. These are not the same distribution. GRPO begins from the SFT checkpoint but then updates — at step 1 it is identical to SFT, but the gate is meant to predict whether the mechanism fires *over the course of training*, not just at initialization. A model that starts with low skeleton diversity might gain diversity as GRPO updates; conversely, a model showing reasonable SFT diversity might collapse to a single high-reward skeleton class within the first 10 GRPO steps. The gate, as designed, only validates the pre-training regime. This is a residual but manageable gap: log `n_classes` and `ske_used` rate continuously through training and treat sustained collapse as an in-training kill switch. The proposal notes this logging but does not formalize it as a gate. It should.

New issue introduced: none beyond the above.

---

## H-2: Mixed Two Algorithms

**Verdict: YES — fully resolved**

What changed: v1 is unambiguously post-hoc only. The action space, prompts, and decoding are frozen. Skeleton-first generation is exiled to v3 and explicitly gated on v2 showing ≥5pp medium gain — a bar that must be cleared first. The 70 LOC concern is moot for v1 (the revised LOC estimate is 260 production LOC, which is realistic for what is actually being implemented). The staged v1/v2/v3 structure is clean and each stage has a discrete entry gate.

What is sound: the staging is correctly motivated. v3 is treated as a distinct algorithmic variant with its own implementation milestone (~500 LOC, 3-5 days), not a cheap add-on. This is honest.

New issues: none. H-2 is closed.

---

## H-3: R_struct = Max-Over-Fillings Core Feasibility Risk

**Verdict: YES — fully resolved for v1, with a caveat for v2**

What changed: max-over-fillings is dropped from v1 entirely. R_struct in v1 equals R_fill equals the standard binary execution match on the rollout's own SQL. No inverse rendering, no schema-aware binding, no type inference required in v1.

What is sound: the offline feasibility prototype gate for v2 (≥70% of skeletons admit ≥1 executable fill within K=8 constrained-decoding attempts) is the right filter. If the prototype fails, v2 is dead before any GRPO resources are spent.

Caveat for v2: the prototype measures executability of *constrained-decoding* fills from the SFT model, not from arbitrary fill generators. This is a reasonable proxy, but it is not the same as measuring inverse-rendering feasibility in general. If the constrained decoder can only produce fills for structurally simple skeletons, the v2 benefit may be limited to easy queries where SKE-RL's v1 advantage mechanism already fires reliably. This is a v2 concern, not a v1 blocker. Flag it in the paper if v2 is pursued.

New issues: none for v1.

---

## M-4: Class-Shared Advantage Degenerates at G=8

**Verdict: YES — adequately addressed, with a monitoring obligation**

What changed: explicit fallback to standard GRPO when distinct skeleton classes in a group fall below `--ske-min-classes` (default 2). The expected fallback rate is acknowledged as 30-50% early in training. Per-step logging of `n_classes`, `ske_used`, and `ske_fallback_rate` is specified.

What is sound: the fallback is the correct engineering response. It bounds worst-case harm: when SKE-RL cannot fire, the model receives standard GRPO updates rather than pathological advantages. The `ske_used` rate is a proxy for how often the mechanism is actually contributing.

Remaining concern: the advantage formula in §2.2 uses a class baseline computed as a simple mean of per-class means, weighting each class equally regardless of how many rollouts it contains. With G=8 and 2 classes, this means a class with 1 rollout and a class with 7 rollouts receive equal weight in the baseline. The singleton class's class advantage is determined entirely by that single reward sample, which is high-variance. If that singleton happens to have an unusually high reward (e.g., a lucky correct-execution outlier), it will inflate the class baseline and depress the within-class advantage of the dominant 7-rollout class. Concretely: the dominant class gets a negative class advantage not because it is structurally wrong, but because a singleton outlier shifted the baseline. This is a bug that the degeneracy fallback does not catch because 2 classes ≥ the minimum threshold. Consider weighting the class baseline by class size (harmonic mean of class counts, or just size-weighted mean), or applying the fallback at a higher threshold such as `--ske-min-classes 3`. This is a medium-severity implementation trap.

New issue introduced: the singleton-inflated-baseline problem described above is not addressed in the proposal. It may not be common enough at G=8 to matter empirically (most groups will likely be 4+4 or 6+2 splits, not 7+1), but it should be flagged in code comments and logged.

---

## M-5: sqlglot Not in Deps

**Verdict: YES — fully resolved**

What changed: sqlglot is explicitly added to pyproject.toml with a version range (`>=21.0,<26.0`), and a smoke test milestone (Day 0, ≥98% parse rate on 50 Spider train queries) is specified in both the proposal and experiment plan.

What is sound: the dependency specification is concrete. The version range is appropriately constrained — sqlglot has had breaking API changes across major versions and pinning to `<26.0` is prudent. The smoke test is a real gate, not a formality.

Implementation note: the experiment plan references `uv add sqlglot && uv lock --upgrade-package sqlglot`. Verify the git status shows `uv.lock` is already modified (it is, per the current repo status) — confirm sqlglot was actually added and is not just a plan artifact. If the lock file was modified for a different reason, run the smoke test explicitly before proceeding.

New issues: none.

---

## M-6: Novelty Unclear vs Existing sql_structure_reward

**Verdict: YES — adequately sharpened**

What changed: the contribution is now explicitly scoped to "advantage normalization over canonical AST equivalence classes, orthogonal to reward shape." The existing `sql_structure_reward` (token/clause overlap, operates on the scalar reward before advantage computation) and SKE-RL v1 (operates on the advantage after rewards are computed) are now correctly positioned as operating at different stages of the GRPO update. v1 leaves the reward function untouched.

What is sound: the distinction is real. Token/clause overlap reward shaping and equivalence-class advantage normalization are mechanistically orthogonal. A reviewer who objects that both are "structural" can be answered precisely: structural reward shaping changes what scalar feedback the model receives; structural advantage normalization changes how that feedback is distributed *across rollouts within a group* without changing its aggregate sum. This is a meaningful difference in the RL formulation.

Remaining sharpness question: the proposal claims the contribution is orthogonal but does not report an ablation that isolates the two. In the Phase 3 design, all three conditions (A1, A2, A3) use the same reward profile (CGFR). This means `sql_structure_reward` is present in all conditions and is not varied. While this is correct for isolating the advantage-normalization change, a reviewer may ask: does SKE-RL still help when `sql_structure_reward` is disabled? The current experiment cannot answer this. This is a future-work question, not a blocker, but note it explicitly in the paper.

New issues: none that are blocking.

---

## Phase 1.5 Gate Calibration

The gate has two thresholds: (1) ≥40% of groups across all difficulties have ≥2 distinct skeleton classes, and (2) for medium queries specifically, ≥15% of groups contain at least one rollout matching the gold skeleton.

**Threshold (1): ≥40% groups with ≥2 distinct classes**

This is too lenient as a general gate and too strict as a diagnostic. At G=8, a random baseline model with near-uniform token probabilities would likely produce ≥2 distinct AST parse results in most groups — surface lexical variation alone generates structural divergence even when the model has no understanding of SQL semantics. A 40% bar does not distinguish "the model is exploring structurally diverse hypotheses" from "the model is generating noisy output that happens to parse differently." The threshold is measuring diversity of parse outcomes, not semantically meaningful skeleton diversity.

A sounder alternative: require ≥40% of groups to have ≥2 distinct classes AND require that the inter-class reward variance exceeds some minimal threshold (e.g., mean class mean spread ≥ 0.1). This filters out the case where 2 classes both have ~0.0 reward — the model is producing diverse failures, which gives SKE-RL no signal to amplify. The proposal already computes class means in §2.2; this check adds no new infrastructure.

**Threshold (2): ≥15% of medium groups contain gold skeleton**

This threshold is calibrated in the right direction — it directly measures whether the positive signal exists — but 15% is a very low bar. If only 15% of medium-query groups contain a gold-skeleton rollout, then 85% of medium training steps provide SKE-RL with no positive class signal at all. In those steps, SKE-RL may still fire (if some non-gold skeleton classes have higher reward than others), but it is not doing what the credit-assignment hypothesis requires: identifying the correct structural approach and reinforcing it. Effective learning from 15% of steps is possible but slow.

A more informative calibration: report the threshold that corresponds to SKE-RL being able to reinforce the gold class in at least 30-50% of medium training steps. At G=8 and a 15% group rate, the expected number of medium steps where SKE-RL can amplify gold-class advantage is very low. If the SFT model produces gold-class rollouts in only 15% of medium groups, a more honest assessment is: SKE-RL will mostly be sorting among wrong skeletons on medium queries. That may still be useful (reducing variance among wrong approaches) but it is not the credit-assignment mechanism the proposal is selling.

Recommendation: keep the 15% threshold as a minimum hard gate but add a soft target of ≥30% and report the actual rate prominently. If the rate is 15-30%, qualify the paper's claims accordingly.

---

## Final Verdict

**R4 v1 is ready to enter Phase 0 implementation.** The three high critiques are resolved; the medium critiques are resolved with one remaining implementation trap (singleton-inflated class baseline at G=8 with 2 classes) and one Phase 1.5 gate calibration concern (diversity thresholds should include reward-variance filtering). Neither is a blocker for implementation, but both should be addressed before the first Phase 3 training run.

**Implementation traps to watch in Phase 0:**

1. The class baseline in `compute_class_aware_advantage` should be size-weighted, not equal-weighted across classes, or the `--ske-min-classes` default should be raised to 3 to avoid singleton-distorted baselines.
2. The `ske_used` rate should be logged as a hard kill-switch condition, not just a diagnostic: if `ske_used` drops below 20% sustained over 20 steps after the first 30 warm-up steps, the mechanism is not firing and training should either fall back entirely to standard GRPO or halt for diagnosis.
3. Phase 1.5 Gate 1.5 threshold (1) should require non-trivial inter-class reward variance, not just ≥2 distinct parse classes.
4. Confirm sqlglot is actually in the lock file (not just planned) before starting Day 1 extractor work.
5. The smoke test gate for `extractor_failure_rate <20%` is generous. If it exceeds 10% on Spider, investigate which patterns are failing before committing to Phase 3.

**The most damning remaining concern is not a blocker but a measurement quality issue: the Phase 1.5 gate, as calibrated, may pass on the basis of noisy parse diversity rather than semantically meaningful structural exploration, causing the experiment to proceed to Phase 3 even when the capacity hypothesis is the true binding constraint.**

