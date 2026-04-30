# Paper draft patch — R2-aligned Section 1 + Section 5 with BIRD-leaderboard scope tightening

**Purpose**: a focused replacement for Section 1 (Introduction) and Section 5 (Discussion) that is consistent with `FINAL_PROPOSAL_v3_DRAFT.md` (R2-revised, "Stopping Policy vs Conditional Solver") AND incorporates the BIRD leaderboard scope tightening from `REVIEW_BIRD_LEADERBOARD_CHECK_2026-04-30.md`.

**Replaces**:
- Section 1 in `PAPER_DRAFT_v3_intro.md` (which uses pre-R2 "commitment policy" terminology and lacks scope tightening).
- Section 5 in `PAPER_DRAFT_v4_results.md` (which positions the negative G=8 result as "G hurts GRPO" without scoping to multi-turn).

Other paper sections (§2 Setup, §3 Method, §4 Results, §6 Limitations) are largely R2-compatible already and need only mechanical edits (rename "commitment-policy shift" → "stopping-policy shift"; rename "per-turn capability" → "conditional solver capability"; promote `share_shift_sym` to primary; add `R_same paired ΔEX` column to Table 1).

---

## §1 Introduction (replacement)

Multi-turn ReAct-style reinforcement learning has become the default training recipe for agentic Text-to-SQL. Recent results — MTIR-SQL [\[2510.25510\]](https://arxiv.org/abs/2510.25510), ReEx-SQL [\[2505.12768\]](https://arxiv.org/html/2505.12768), HES-SQL, MARS-SQL, Graph-Reward-SQL — report aggregate execution-match (EX) gains in the +1 to +5 pp range over supervised fine-tuning, and the gains are typically attributed to "the model learns to write better SQL" or "the model learns to recover from execution errors". These attributions are read directly off the aggregate EX gain.

We argue this attribution is under-determined. Aggregate EX admits two operationally distinct mechanisms that are invisible to the metric:

1. **Conditional solver**: at fixed turn count, the model's SQL is more often correct.
2. **Stopping policy**: the model resolves more queries in a single turn (where SFT's accuracy is already higher) without improving accuracy at any individual turn count.

Mechanism (2) is invisible because aggregate EX does not condition on turn count. A model whose RL gains are dominantly mechanism (2) has not become a better SQL writer; it has become more willing to commit early. The two mechanisms also have opposite predictions for negative results: a structural advantage prior like SKE-RL [our prior work] that pushes the policy toward "skeleton-complete" trajectories should amplify mechanism (2) regardless of whether mechanism (1) follows.

We introduce a deterministic, leakage-safe two-term decomposition (Shapley-symmetric primary, sensitivity-checked across orderings):

```
total_ΔEX = stopping_policy_term + conditional_solver_term     (exact per resample)
```

with paired-record bootstrap 95% CIs, and a complementary **R_same paired ΔEX** restricted to records where SFT and RL used identical turn counts — composition is held fixed by construction, so any signed signal in this subset cannot be a stopping-policy artifact.

### Scope of claims

The mechanism we identify exists only when the agent's turn count is non-degenerate. We therefore restrict every quantitative claim in this paper to **multi-turn ReAct GRPO** with at most 5 interactions per query. Single-turn GRPO methods (Arctic-Text2SQL-R1 [\[2505.20315\]](https://arxiv.org/abs/2505.20315) is the canonical recent example, achieving 70.5 BIRD-dev EX at 32B with G=16 and a single-shot Answer format) lie outside this scope: there is no "stopping policy" axis when the agent emits at most one Answer. We do not make any claim about how stopping-policy effects would or would not generalize to single-turn settings; the diagnostic is silent there.

### Empirical findings (sketch — full results in §4)

Applied to a 7B Qwen2.5-Instruct ReAct agent on the full Spider-dev split (1034 examples), with one SFT baseline and three GRPO-trained checkpoints (Dense G=4, SKE-RL G=4, SKE-RL G=8) plus one pre-registered confound run (Dense G=8):

- Dense G=4's +2.32 pp aggregate gain over SFT decomposes into a significantly positive stopping-policy shift (+0.83 [+0.07, +1.67]) and a positive conditional-solver point estimate (+1.49 [−0.51, +3.66]). The R_same paired ΔEX is +2.22 pp [+0.37, +4.06] (CI excludes 0) — Dense G=4 is, unambiguously, a better conditional solver.
- SKE-RL G=8's +0.39 pp aggregate gain decomposes into the largest stopping-policy shift in our data (+2.52 [+1.71, +3.50]) AND a significantly negative conditional-solver term (−2.13 [−3.87, −0.46]). The R_same paired ΔEX is +0.78 pp [−0.56, +2.22] (overlaps 0) — SKE-RL G=8 does not improve conditional SQL writing on the records it didn't move.
- A pre-registered prediction in the v3 draft of this work (locked 2026-04-29 before the Dense G=8 run completed) tested whether the conditional-solver regression was specific to SKE-RL's class-baseline advantage. The Dense G=8 confound run **falsified** the SKE-specific framing — Dense G=8 reproduces SKE-G8's pattern within CIs. The mechanism is **shared between Dense and SKE-RL at G=8**, not advantage-estimator-specific.

This last result aligns with — and may quantitatively explain — MTIR-SQL's [\[2510.25510\]](https://arxiv.org/abs/2510.25510) explicit observation that "vanilla multi-turn GRPO suffers from reward collapse" and their resort to "SQL execution rollout expansion and trajectory filtering" as stabilization. They chose G=5 for training; we report what happens at G=8 in our recipe and find a measurable conditional-solver regression that no aggregate metric would surface.

### Contributions

1. **Methodological**: a deterministic Shapley-symmetric decomposition of agentic-RL aggregate metrics into stopping-policy and conditional-solver components, with paired bootstrap CIs and an R_same composition-decontamination check. Reference implementation released.
2. **Empirical (multi-turn ReAct GRPO at 7B)**: G=8 — both with Dense and with SKE-RL advantage estimators — produces a stopping-policy gain that is offset by a conditional-solver loss, washing out aggregate EX gain. G=4 produces a smaller but cleaner gain.
3. **Methodological transparency**: a pre-registered prediction (specific to v3's SKE-RL framing) was falsified by the data. We document the falsification and its absorption into the v3 R2 framing as a methodological choice.
4. **Cross-model-size replication**: the conditional-solver collapse mode appears at 3B as well (R_same paired ΔEX −33.9 pp [−47.5, −20.3]), consistent with the 7B SKE-RL G=8 finding at much larger magnitude.

### Non-contributions

We do NOT claim a new RL training method, a SOTA Spider/BIRD result, a scaling-law claim, or that "G hurts GRPO" universally. The Arctic-Text2SQL-R1 single-turn G=16 result is a clean counterexample to any universal claim and is explicitly outside our scope.

---

## §5 Discussion (replacement)

### What the data show

Within the multi-turn ReAct GRPO regime at 7B, the **dominant variance** in ΔEX between training configurations is along the stopping-policy axis: every RL condition we test (Dense G=4, SKE G=4, Dense G=8, SKE G=8) has `share_shift_sym` 95% CI excluding 0 in the positive direction. The variance in the **conditional-solver axis** is what determines whether aggregate EX wins or loses: G=4 conditions have non-negative `per_turn_sym` point estimates (Dense +1.49, SKE +0.44), while G=8 conditions are significantly negative (Dense −1.91, SKE −2.13). The R_same paired ΔEX disambiguates: Dense and SKE at G=4 improve conditional SQL writing on records they didn't move; SKE at G=8 does not.

The mechanism is **shared between Dense and SKE-RL at G=8** — point estimates are within 0.5 pp of each other on both decomposition terms, and the (1→1) cell same-record EX rates are identical to one decimal (Dense G=8: 70.1%; SKE G=8: 70.0%; SFT baseline: 73.6%). The trade-off is structural to the multi-turn rollout-budget choice, not specific to SKE-RL's class-baseline advantage. This falsifies the v3 hypothesis.

### Position vs published multi-turn agentic-SQL methods

The closest comparable recipe in the published literature is MTIR-SQL [\[2510.25510\]](https://arxiv.org/abs/2510.25510): 4B model, multi-turn ReAct, GRPO, max 6 interactions, **G=5**, with explicit "SQL execution rollout expansion and trajectory filtering to stabilize training in multi-turn tool-use scenarios, effectively mitigating reward collapse". MTIR-SQL's hyperparameter choice and stabilization tricks are independent corroboration of two phenomena we observe: (a) modest G is the working regime for multi-turn ReAct GRPO; (b) vanilla multi-turn GRPO has stability problems serious enough to warrant explicit mitigation.

Our diagnostic provides a quantitative reading on what those stability problems look like. MTIR-SQL's "reward collapse" likely manifests as a stopping-policy / conditional-solver imbalance — the policy commits earlier in the trajectory to claim higher reward density, sacrificing per-turn correctness. Their stabilization tricks (rollout expansion, trajectory filtering) effectively rebalance the two terms. We have not run their recipe with our diagnostic; doing so is the natural next experiment.

ReEx-SQL [\[2505.12768\]](https://arxiv.org/html/2505.12768) and HES-SQL operate in the same multi-turn agentic regime and report comparable aggregate EX. None decompose by turn-progression. Applying our diagnostic to their public eval logs (where available) would test whether the trade-off curve is universal across multi-turn methods or specific to our advantage estimators.

### Position vs single-turn agentic-SQL methods (out of scope)

Arctic-Text2SQL-R1 [\[2505.20315\]](https://arxiv.org/abs/2505.20315) trains GRPO with **G=16** on a single-turn Answer format and tops BIRD-dev (70.5 at 32B, 68.9 at 7B). Single-turn GRPO does not have a stopping-policy axis: the agent emits exactly one Answer and there is no "commit early vs continue exploring" choice for the policy to learn. Our finding is silent on Arctic-R1's regime; we do not predict that G=16 in single-turn GRPO has any analogue of the conditional-solver regression we observe in multi-turn.

### Cross-model-size replication

Re-applying our diagnostic to a previously-archived 3B GRPO run (reward variant unspecified; from our earlier failed-RL pilot) shows the same conditional-solver collapse mode at much larger magnitude: Shapley-sym `per_turn` term −21.28 pp [−34.63, −7.69] (CI excludes 0); R_same paired ΔEX −33.90 pp [−47.46, −20.34]. The 3B trajectory is even more extreme — single-turn share swings 28.0% → 92.5% under RL training — and the conditional solver collapses by 33.9 pp on records where turn count was unchanged. This is a non-trivial cross-model-size confirmation that the mechanism is real, not a 7B G=8 artifact.

### Implications

For agentic-RL practitioners: in multi-turn ReAct GRPO at the 7B scale we tested, G=8 is *not* a free upgrade over G=4. The stopping-policy gain looks like a +2.5 pp aggregate boost at face value, but is offset by a per-turn quality loss of comparable magnitude in the records that didn't shift. Practitioners optimizing aggregate EX who push G upward should expect this trade-off and either (a) stop at G=4-ish, or (b) add MTIR-SQL-style stabilization to keep the conditional-solver term safe.

For researchers reporting agentic-RL gains: aggregate EX is insufficient to characterize what the model has learned. The two-term decomposition is a one-Python-file addition to any existing eval pipeline; we encourage adoption.

### Limitations and what would increase confidence

(See §6 for the full list. Headlines:)

- **One model family (Qwen2.5)**, two model sizes (3B, 7B). A Llama-3 or Mistral baseline at 7B would test family-independence.
- **One training corpus** (Spider). BIRD or BIRD-Interact would test dataset-independence; BIRD-Interact in particular would let us validate the framework on a benchmark designed for multi-turn evaluation.
- **Two G values per estimator** (4, 8). G=2 and G=16 would extend the trade-off curve. G=16 in particular would test whether the conditional-solver loss saturates or accelerates.
- **No causal intervention on G**. We observe correlation between G and the two terms; we do not run a counterfactual pair holding everything else fixed. The closest match is the Dense G=4 ↔ Dense G=8 pair, which is the right comparison but a single one.
- **Recipe is below SOTA**. Our 7B Dense G=4 reaches 66.6% Spider-dev EX. Top BIRD methods report 68-70+ at 7B. This may reflect under-training, suboptimal reward shape, or the absence of MTIR-SQL-style stabilization. The diagnostic itself is robust to this — the decomposition is meaningful regardless of absolute level — but the negative G=8 result is not directly comparable to a SOTA-trained system. Replicating MTIR-SQL's recipe with our diagnostic is the canonical fix.

---

## Citations to add (with arXiv IDs and URLs)

```bibtex
@article{arctic_text2sql_r1_2025,
  title={Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL},
  author={...Snowflake AI Research...},
  journal={arXiv preprint arXiv:2505.20315},
  year={2025},
  url={https://arxiv.org/abs/2505.20315}
}

@article{mtir_sql_2025,
  title={MTIR-SQL: Multi-turn Tool-Integrated Reasoning Reinforcement Learning for Text-to-SQL},
  journal={arXiv preprint arXiv:2510.25510},
  year={2025},
  url={https://arxiv.org/abs/2510.25510}
}

@article{reex_sql_2025,
  title={ReEx-SQL: Reasoning with Execution-Aware Reinforcement Learning for Text-to-SQL},
  journal={arXiv preprint arXiv:2505.12768},
  year={2025},
  url={https://arxiv.org/html/2505.12768}
}

@article{bird_interact_2025,
  title={BIRD-Interact (full title from arxiv:2510.05318)},
  journal={arXiv preprint arXiv:2510.05318},
  year={2025},
  url={https://arxiv.org/abs/2510.05318}
}
```

(Plus existing Spider, GRPO/DeepSeek, Qwen, sqlglot citations from the earlier draft.)
