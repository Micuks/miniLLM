---
date: 2026-04-30
status: internal review (Codex MCP unavailable in this environment)
reviewer: Claude acting as senior ML reviewer (NeurIPS/ICML perspective)
target: FINAL_PROPOSAL_v4.md, PAPER_DRAFT_v3_intro.md, PAPER_DRAFT_v4_setup.md, PAPER_DRAFT_v4_results.md
question: 当前的故事和实验结果支撑什么水平论文？
---

# Internal review — what venue does v4 support?

## TL;DR

**Workshop accept / top-tier conference reject.** Concretely:
- **Strong fit**: NeurIPS/ICLR 2026 workshops (TRL, RL-for-LLMs, Agentic Foundation Models).
- **Plausible fit**: ACL Findings or short paper (with Section 6 expansion).
- **Reject**: NeurIPS/ICML/ICLR main; ACL/EMNLP main.

Score this would receive: **5–6/10 at workshop (likely accept), 3–4/10 at main conference (likely reject)**. The pre-registered falsification + clean decomposition raise it above a typical negative-result paper, but single-model + single-dataset + 2-G points + small absolute effect sizes cap the venue.

## What's strong

### Methodology

1. **Pre-registered prediction with quantitative bounds, then explicit falsification.** This is genuinely rare in the agentic-RL literature. The paper can credibly claim methodological contribution beyond the empirical finding.
2. **Bootstrap protocol is locked and reproducible.** Seed=20260429, 1000 paired resamples, percentile CI, hash-pinned inputs. Reviewer cannot reasonably challenge the statistics.
3. **Two-term decomposition is mathematically clean.** Identity-true on every paired resample; the bootstrap surfaces uncertainty in each term. Tight, interpretable.
4. **Multiple independent confirmations of the same mechanism.**
   - Aggregate (Table 1): G=8 share_shift CI excl 0 ✓, per_turn CI excl 0 ✗
   - Per-difficulty (Table 2): per_turn negative across Medium and Hard for both G=8 conditions
   - Paired transitions (Table 3): shift-down `net Δ` monotonically decreasing with G across both estimators
   - 1→1 cell EX% (Table 3.5): G=8 conditions identical to 0.1pp (70.1 vs 70.0)
   - Trajectory length (§4.5): EX-lost trajectories +211 / +242 chars under both G=8 conditions
   These are not 5 redundant tests of the same effect; they are 5 different views that all point to the same mechanism.

### Empirical alignment

5. **The 2×2 ablation is informative ex post**. Dense G=8 was the right confound to run. The result simultaneously confirms (a) the share-shift mechanism is real, (b) the per-turn regression is a generic G effect, (c) advantage-estimator choice is operationally neutral at the G we test.

6. **Aggregate point estimates align with the mechanism story without cherry-picking.** Dense G=4 is the only condition with a positive per-turn point estimate; it's also the only condition that wins net. This is what the mechanism predicts.

## What's weak

### Generalization

1. **Single model size (7B Qwen2.5-Instruct).** The G-vs-capability trade-off claim sounds general but rests on one model. A reviewer will ask: does the trade-off saturate, reverse, or amplify at 13B / 1.5B / 32B? You cannot answer this.

2. **Single dataset (Spider-dev).** Spider has well-studied artifacts (table-aliased joins, narrow question style, deterministic SQL). BIRD or sql-create-context would test whether the mechanism is dataset-specific. You don't have those numbers.

3. **Only two G values (4, 8).** The "trade-off" framing implies a curve; you have two points. A G=2 baseline would test whether share_shift is monotonic in G or has a sweet spot. A G=16 run would test whether per_turn loss saturates or accelerates. Reviewers will flag this. "Pre-registered binary prediction" only partially defends — the framing in §5 talks about a "scaling trade-off" that you have not actually scaled.

4. **Absolute effect sizes are small.** Dense G=4's +2.32 pp aggregate gain over SFT is modest. Reviewers comparing to recent agentic-SQL papers (HES-SQL, MTIR-SQL claim 5–10 pp gains) will note that the paper's "winning" condition isn't competitive on aggregate metrics. Counter-argument: that's the paper's *point* — aggregate metrics hide the mechanism — but it weakens the narrative that the mechanism is important enough to study.

### Methodology

5. **Decomposition is descriptive, not causal.** §6 admits this. A reviewer who wants causal evidence will note that you observe correlation between G and the two terms; you do not intervene on G in a counterfactual way. The natural causal experiment is: train Dense G=4 to step 75 / step 100 / step 200 and see if share_shift increases with training duration at fixed G — that would establish that G drives the share_shift via training dynamics, not via random sampling effects.

6. **Single PPO epoch, skip-old-logprob.** This is a real implementation choice. A reviewer might ask: does multi-epoch PPO change the trade-off? You explicitly disallow `--ppo-epochs > 1` (NotImplementedError). This is fine for a workshop but a main-conference reviewer will challenge.

7. **Mechanism is inferred from a 2-condition contrast** (G=4 vs G=8). Within each G, advantage-estimator choice (Dense vs SKE-RL) was *also* a single choice. So the 2×2 is really 4 unrelated points; the "G effect is independent of estimator" claim rests on point estimates being similar, not on a high-power test of estimator-G interaction.

### Narrative

8. **The paper is fundamentally a negative-result story.** "Adding more rollouts hurts capability" is not what reviewers expect from an RL-improvements paper. Reviewers reading abstracts will subconsciously file it as "they tried bigger group size and it didn't work" — even though the diagnostic is the contribution.

9. **The original v2 (protocol-vs-semantic) and v3 (SKE-specific) framings were both wrong.** This shows methodological honesty (which is good) but also that the diagnostic *required two pivots* before producing a publishable claim. A reviewer will be slightly skeptical that v4 is the right framing rather than v5-pending.

10. **"Decomposition is novel" is asserted, not shown.** §1 claims "no published agentic Text-to-SQL RL paper reports any of [these metrics]". A reviewer will demand a more thorough literature scan. You haven't actually re-implemented HES-SQL or MTIR-SQL to confirm they produce the same pattern when decomposed.

11. **The "portable to other agentic-RL settings" claim is unsupported by data.** It's a one-paragraph speculation. Reviewers will ask: did you try it on web agents / code agents? You will say no.

### Unforced errors

12. **`commitment_ratio` is suppressed for G=8 because total ΔEX has CI overlapping 0.** A pedantic reviewer will note this is convenient (you suppress the ratio precisely where it would be ill-defined), but the suppression rule was pre-registered, so it's defensible.

13. **Per-difficulty CIs on Hard are wide** (±5pp on per_turn_gain). The narrative leans on Hard for the SKE-G8-specific story but the data don't strongly support it (point estimate −2.84, CI overlaps 0).

## Mock NeurIPS workshop review

```
Title: Group Size Trades Commitment for Capability — A Falsified Pre-registered
       Decomposition of Agentic GRPO on Text-to-SQL

Summary: The paper introduces a two-term decomposition (share_shift + per_turn_gain)
of aggregate execution-match gains in agentic GRPO and applies it to a 2×2
ablation {Dense, SKE-RL} × {G=4, G=8} on Spider-dev. The central finding is that
group size G amplifies a commitment-policy shift but degrades per-turn capability,
and this trade-off is independent of the advantage estimator. The authors made a
quantitative pre-registered prediction (Dense G=8 isolates share-shift without
per-turn loss) and report its falsification, leading to a reframe.

Strengths:
+ Pre-registered prediction + transparent falsification is methodologically rare
  in agentic-RL literature.
+ Bootstrap protocol is rigorous; CIs reported on every headline number.
+ Multiple independent analyses converge on the same mechanism story.
+ Decomposition formula is mathematically clean and immediately portable.

Weaknesses:
- Single model size (7B) and single dataset (Spider). Generalization claim
  rests on assertion, not evidence.
- Only 2 G values — "scaling trade-off" framing implies a curve but the data
  give 2 points. G=2 and G=16 are obvious next experiments.
- Absolute effect sizes are small. The "winning" condition (Dense G=4) gains
  +2.32 pp over SFT, well below recent agentic-SQL claims.
- Decomposition is descriptive; no causal intervention on G.
- The "advantage-estimator-independent" claim relies on point-estimate similarity,
  not on a high-power test for interaction.

Questions for authors:
1. What does the trade-off curve look like at G=2 and G=16? If the per_turn
   regression is monotonic in G, that strengthens the claim significantly.
2. Does the trade-off survive at a different model size? A 1.5B run is feasible.
3. Have you applied the decomposition to public eval logs from HES-SQL or
   MTIR-SQL? If yes, what does it show?
4. The decomposition is mathematically forced (an identity). What is the
   contribution beyond the formula itself?

Score: 5/10 (Weak Accept at workshop)
Confidence: 4/5
What would move toward Strong Accept:
  - One additional G value (G=2 OR G=16; not both)
  - One additional dataset (BIRD-mini or sql-create-context)
  - A re-decomposition of a public agentic-SQL paper's eval logs
```

## Mock NeurIPS main-conference review

```
Score: 3-4/10 (Reject)
Confidence: 4/5
Key blocker: single model × single dataset × 2 G points × small absolute effects.
The decomposition is clean but the empirical scope is too narrow for a main
conference. The pre-registered falsification is appreciated but not by itself
sufficient at the main-conference bar.
What would move toward Accept:
  - 3+ model sizes
  - 2+ datasets
  - 4+ G values (with monotonicity test)
  - Decomposition applied to ≥1 published agentic-RL benchmark for cross-validation
  Total compute estimate: ~150-300 GPU-h. Feasible on a research lab budget.
```

## Honest venue forecast

| Venue | Verdict | Confidence |
|---|---|---|
| NeurIPS/ICML/ICLR main | Reject | High |
| NeurIPS workshop (TRL, RL-for-LLMs) | Accept | Medium-high |
| ICLR workshop (Agentic Foundation Models) | Accept | Medium-high |
| ACL/EMNLP main | Reject | High |
| ACL Findings | Accept (after expansion) | Medium |
| ACL/EMNLP short | Accept | Medium |
| ACL/EMNLP workshop (TacITRL, NLP4Programming, etc.) | Accept | High |

**Recommended target: NeurIPS 2026 TRL workshop or ICLR 2026 Agentic Foundation Models workshop.** Deadlines typically open in mid-2026; submission timeline fits.

## Minimum experiment package for venue lift

If the user has 2-4 weeks and ~150-200 GPU-h available, the highest-leverage
experiments to push from workshop to ACL Findings / short are:

1. **Dense G=2 (10 GPU-h training + 5 GPU-h eval)**
   Tests share_shift monotonicity below G=4. Pre-register prediction:
   share_shift_term ∈ [-0.5, +0.5] pp (CI overlaps 0) — i.e., G=2 is below the
   threshold where commitment shift kicks in. If confirmed, you have a 3-point
   scaling curve (G=2, 4, 8) with predicted shape.
   Risk: if G=2 has share_shift > 0.5, the trade-off has a different curve than
   "monotonic in G", weakening the "G knob" framing.

2. **Re-eval an existing 3B Dense GRPO checkpoint at G=4 with the decomposition**
   You already have 3B SFT and 3B GRPO from the earlier (archived) experiments.
   Run the diag pipeline on those eval JSONs. Does the trade-off appear at 3B
   too? This costs 0 GPU-h (just CPU re-analysis on existing JSONs). High value
   per GPU minute.

3. **Decomposition on one public eval artifact (HES-SQL preferred)**
   If HES-SQL's authors release per-record SQL predictions on Spider-dev (likely
   true), apply the diag pipeline to their data. Shows portability and that the
   mechanism is observable in published baselines.
   Cost: 1-2 days of artifact-search + CPU work. 0 GPU-h.

4. **(if 2-3 succeed) Add G=16 SKE-RL or G=16 Dense (~30 GPU-h)**
   Tests whether per_turn regression saturates or amplifies further. Pre-register:
   share_shift_term > +2.6 pp; per_turn_gain_term < -2.5 pp.

If only #1 and #2 fit the budget: the paper's claim becomes "the trade-off is
monotonic in G across model sizes (3B + 7B) and across G values (2, 4, 8) with
pre-registered binary predictions confirmed at each step". That is plausibly
ACL-Findings tier.

## Recommendations to the user

1. **Submit to a workshop first.** Use the falsification + decomposition as the
   main story. Workshop bar is the right target for current data.
2. **Run #2 (3B re-decomp) immediately** — it's free CPU and either supports
   the story or honestly limits it. If 3B shows the same pattern, the claim
   that the trade-off is generic gets a second model size at 0 GPU-h cost.
3. **Don't oversell portability** in the workshop version. Keep §5 honest:
   "we conjecture this generalizes to other agentic-RL settings" instead of
   asserting it does.
4. **Tighten Section 1**. The current intro implicitly compares to
   HES-SQL/MTIR-SQL gains; consider acknowledging up front that aggregate
   gains are modest in our setup and the contribution is interpretive rather
   than a new SOTA.
5. **Consider whether to lead with falsification or with decomposition.**
   Workshop reviewers may find the methodological story (pre-registered
   falsification) more novel than the empirical claim (G knob). The current
   draft buries the falsification in §4.2; consider putting it in the abstract.
