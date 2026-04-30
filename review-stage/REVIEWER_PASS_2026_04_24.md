# Reviewer Pass — 2026-04-24

This proposal is trying to rescue a regime that already looks empirically dead. The problem is not lack of creativity. The problem is lack of identification. All three interventions can produce superficially nicer training curves while still failing to answer the only question that matters: is 3B failing because GRPO has no usable signal, or because 3B simply does not have the capacity margin to survive multi-turn RL on medium queries? As written, the proposal repeatedly risks mistaking "changed the optimization dynamics" for "explained the failure mode."

---

## GRM

### Novelty Audit

There is no serious novelty story here. I could not find a direct arXiv paper on "off-policy GRPO," but the closest prior work is already uncomfortably near: ReST for language modeling (arXiv:2308.08998), RAFT (arXiv:2304.06767), and the rejection-sampling-plus-RL workflow in Llama 2 (arXiv:2307.09288). All three already live in the space of harvesting good samples, reusing them off-policy or semi-offline, and using them to stabilize or redirect learning. GRM's delta is mostly an implementation splice: keep a prompt-indexed bank of winning trajectories inside a GRPO batch and clip importance weights. That is a fine engineering trick. It is not a clean algorithmic contribution. **Novelty score: 3/10.**

Structural problem: GRM muddies the causal question it claims to test. If performance improves, a hostile reviewer will say you did not show "reward-signal starvation matters"; you showed that injecting demonstration-like positives from SFT can partially undo damage from RL. Those are not the same claim.

### Methodological Flaw

The weakest link is exactly the one you already half-acknowledge: clipped IS over 100 GRPO steps with a 25-step bank refresh is not credible enough to carry the causal burden. In a language model, the occupancy mismatch is not a mild continuous-control nuisance. It is a combinatorial shift over long trajectories. Once the policy drifts, clipped ratios do not "correct" the stale bank; they mostly zero out the part of the signal you wanted, while still letting the stale sample determine the batch mean and variance structure. That is the worst of both worlds: biased enough to mislead, weak enough to deny responsibility. A reviewer will say the off-policy correction is decorative.

### Claims-Evidence Audit

1. **Medium improves ≥5pp** — the allowed claim is still too strong. One run only shows banked positives can help; it does not isolate reward starvation from imitation rescue. The ablation that closes the gap: prompt-matched bank versus shuffled-bank. If only prompt-matched wins help, you have real evidence. If shuffled wins help too, you are just regularizing toward generic valid behavior.

2. **Marginal gain** — "weak mitigation" is empty because you do not know whether staleness or mechanism is the bottleneck. Needed ablation: fixed bank versus refreshed bank.

3. **No gain** — you still do not get to claim strong evidence for capacity. Failure could come from bank sparsity, IS collapse, or off-policy contamination. Needed ablation: an oracle same-policy bank constructed from the current model over a short window.

4. **Easy degrades, medium flat** — "needs per-bucket gating" is too glib. You need a bucket-gated bank-injection ablation, not a post hoc story.

### Hostile Alternative Explanations

If GRM works, the most damaging explanation is simple: the SFT model already contained the rare successful trajectories, RL destroyed them, and your bank just reintroduced them. That is an indictment of your RL recipe, not evidence for a novel fix. A second hostile read: GRM is covert rejection-sampling distillation. The "RL" branding becomes cosmetic.

### Composition Risks

GRM plus PORG is especially dangerous. Banked trajectories will almost always win the pairwise ordering, which means the ordering mechanism amplifies stale off-policy winners before IS clipping can do anything useful. GRM plus ATE is also incoherent at the objective level: GRM is pulling the policy toward a narrow positive support, while ATE explicitly pushes mass back out. On medium prompts you are literally paying one term to sharpen and another term to flatten. Expect oscillatory KL and unstable effective step size.

---

## PORG

### Novelty Audit

I found no direct prior on pairwise-advantage GRPO for LLMs, but the nearest literature is obvious enough that novelty claims should be modest: DPO (arXiv:2305.18290), RRHF (arXiv:2304.05302), and REBEL (arXiv:2404.16767). PORG is basically the old pairwise-preference/ranking instinct grafted onto intra-batch GRPO rollouts with handcrafted tiebreakers. The "ordered rollout" framing sounds new only if one ignores two years of direct preference optimization and ranking-based alignment. **Novelty score: 4/10.**

Structural problem: the SQL edit-distance tiebreaker is privileged-answer leakage dressed up as "signal shaping." You are not merely ranking failures. You are injecting dense supervision from the gold target into the policy update. A reviewer can correctly say this stops being a clean RL objective and becomes answer-conditioned imitation with a fancy wrapper.

### Methodological Flaw

The hostile attack is obvious: once the policy converges to consistent format-hack patterns, the tiebreakers stop measuring progress and start rewarding cosmetics. Lexicographic ordering does not save you. If most samples satisfy parse success and format validity, the ranking collapses onto edit distance to gold or other brittle surface proxies. Then PORG either degenerates into dense imitation pressure or becomes numerically noisy when near-ties dominate. In both cases, the "all batches informative" story is false in the way that matters.

### Claims-Evidence Audit

1. **Skipped batches drop <5% and eval improves** — not enough. `skipped` is an artifact of the baseline estimator, not a universal pathology metric. PORG can trivially drive it to zero by manufacturing order. Needed ablation: PORG versus random-tiebreak PORG with matched nonzero advantage rate.

2. **Skipped drops but eval flat** — "capacity dominates" is unjustified. Flat eval could mean the extra gradients are simply wrong. Needed ablation: reward-jitter or tiny-noise ordering as a control for "nonzero advantage alone."

3. **Policy converges to surface-form matching** — directionally right but too cheap. Needed ablation: canonicalized-SQL or execution-equivalent tiebreak to show the pathology is truly surface-driven rather than rank-construction artifact.

### Hostile Alternative Explanations

If PORG improves, the immediate reviewer attack: it did not fix zero-variance RL at all; it just densified supervision through edit distance and parser features. Gains come from manually engineered proxies correlated with the gold answer, not from any pairwise GRPO insight. A second damaging read: PORG teaches cleaner SQL-shaped strings, which boosts narrow execution metrics without solving reasoning.

### Composition Risks

With GRM, PORG becomes even less defensible. The banked positive will usually outrank every on-policy sample, making pairwise negatives against the current policy effectively stale-behavior imitation. Because PORG works through relative ordering, clipping bank token ratios does not prevent the stale sample from flipping the sign of everyone else's gradient. This is a clean recipe for gradient poisoning disguised as stabilization.

---

## ATE

### Novelty Audit

I could not find a direct prior on per-difficulty entropy targets for LLM RL. Closest approximations: automatic temperature tuning in SAC ("Soft Actor-Critic Algorithms and Applications," arXiv:1812.05905), the original SAC paper (arXiv:1801.01290), and the maximum-entropy RLHF perspective in "On the Algorithmic Bias of Aligning Large Language Models with RLHF" (arXiv:2405.16455). Honest summary: adaptive entropy is old, adaptive entropy for LLM alignment is not new in spirit, and per-difficulty targets are a reasonable but obvious extension. **Novelty score: 5/10.**

Structural problem: your target variable is not well-defined. "Mean token entropy per difficulty bucket" conflates difficulty, response length, turn structure, action vocabulary, and current sampling distribution. A medium query with a long ReAct scaffold naturally has different entropy statistics than an easy one. That does not mean it deserves a different exploration target. You are treating a mixed descriptive statistic as if it were a stable control variable.

### Methodological Flaw

The weakest link is the moving-target problem. If RVDS, curriculum, or any other training-time reweighting changes bucket frequencies, then the per-bucket dual variables are chasing a nonstationary occupancy distribution. The learned alpha is no longer adapting to "difficulty"; it is adapting to whichever prompts were sampled recently. This makes the per-bucket `H*` story conceptually sloppy and empirically hard to interpret.

### Claims-Evidence Audit

1. **Improved `corr_change`** — not enough to claim per-bucket entropy targeting prevents collapse. `corr_change` is a proxy over training dynamics, not task quality. Needed ablation: fixed-alpha sweep matched for average entropy. Without that, ATE may just be a better regularization strength.

2. **Late-step reward hacking disappears** — under-evidenced because the proposal does not establish a crisp reward-hacking diagnostic beyond late degradation. Needed ablation: explicit matched-checkpoint auditing of format validity, parse rate, and execution mismatch under fixed KL.

3. **Training stable, eval unchanged** — "capacity bottleneck" is too strong. Unchanged eval could mean `H*` is mis-specified. Needed ablation: bucket-wise fixed-target sweeps around the probed baseline, not one arbitrary 1.2–1.5 multiplier schedule.

### Hostile Alternative Explanations

If ATE helps, the reviewer will say it is just extra regularization or an effective learning-rate reduction by another name. Dynamic entropy bonuses often look smart while merely slowing harmful updates. A second alternative: alpha learns bucket frequency artifacts, not true difficulty. Then the "per-difficulty" story is just branding.

### Composition Risks

GRM plus ATE is a direct objective conflict. GRM says "collapse onto known good trajectories." ATE says "maintain exploration, especially on harder buckets." If the medium bucket is exactly where bank hits matter, ATE will inflate entropy on the same prompts where GRM wants low-entropy replication. Expect worse IS ratios, unstable KL, and an ugly dependence on update ordering. If the combination works, the burden is on you to show ATE is not merely masking GRM's brittleness.

---

## Overall Assessment

The intervention with the highest expected reviewer score is **GRM**, but only if it is sold honestly as a diagnostic instrument rather than a novel algorithm. It has the clearest falsification logic and the most direct connection to the observed zero-win pathology. The intervention most likely to be dismissed as incremental is also **GRM**, because it sits squarely in the shadow of ReST, RAFT, and rejection-sampling alignment.

**PORG** is the most review-fragile. The gold-answer tiebreaker is a glaring attack surface and makes the method look less like principled RL and more like hand-engineered dense supervision. **ATE** is conceptually cleaner than PORG, but the target definition is mushy enough that a strong reviewer will call it an adaptive regularizer in search of a mechanism.

The single biggest improvement to the proposal is not another ablation inside 3B. It is adding one hard external anchor — a same-pipeline 7B control as early as possible. Right now every positive 3B result can be dismissed as a local patch, and every negative result still leaves room for "maybe the patch was wrong." A 7B anchor collapses that ambiguity fast.

**Is this worth pursuing given the 7B alternative?** Only in a very narrow sense. One cheap GRM falsification run is worth doing because it has real decision value. The full three-intervention program is not. If compute is limited, the scientifically cleaner move is to spend it on 7B, because 7B has higher publication upside and much better power to separate capacity failure from optimizer pathology.

---

## References

- ReST: arXiv:2308.08998
- RAFT: arXiv:2304.06767
- Llama 2: arXiv:2307.09288
- DPO: arXiv:2305.18290
- RRHF: arXiv:2304.05302
- REBEL: arXiv:2404.16767
- SAC: arXiv:1801.01290
- SAC Algorithms and Applications: arXiv:1812.05905
- On the Algorithmic Bias of Aligning LLMs with RLHF: arXiv:2405.16455
