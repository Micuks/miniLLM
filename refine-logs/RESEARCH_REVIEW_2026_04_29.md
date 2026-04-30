---
date: 2026-04-29
reviewer: gpt-5.5 via codex exec, xhigh reasoning
session: 019dd6aa-5d15-7ed2-977a-6fdf5c9ac819
trace: .aris/traces/research-review/2026-04-29_run01/
status: actionable
---

# Research Review — Story A on Capacity-bounded Text-to-SQL RL

External senior-ML review (NeurIPS/ICML level). Two rounds, xhigh reasoning. Self-contained — does not require reading the trace files.

## TL;DR

**Story A as currently framed ("capacity, not credit") will not survive main-track review.** Single capacity step + single seed + missing BIRD + ambiguous G=8 control → mock review verdict: borderline reject (Soundness 2/4, Contribution 2/4).

**Recommended pivot**: reframe as **"Diagnostic Study of Multi-turn Text-to-SQL RL"** — workshop submission in 2 weeks, then optionally harden for main later. SKE negative result becomes one of three findings (alongside 7B-only RL benefit and the "what RL actually changes" diagnostic), not the lead.

**Highest-EV next 3 weeks** (single 24GB Quadro RTX 6000):
1. Zero-cost diagnostics: per-class breakdown, KL/entropy/format-error curves, format-vs-reasoning partition (uses existing eval JSONs).
2. ~1.2 run-eq: Dense G=8 control (disambiguates "G effect" from "SKE effect").
3. ~0.5 run-eq: 3B Dense GRPO same-launcher rerun (re-anchors the 3B cliff).
4. If time: 1.5B SFT+Dense (third capacity point), then BIRD-mini.
5. **Skip**: 14B Dense (OOM risk on 24GB), 3-seed multi-condition packages (~7-10 run-eq, infeasible).

## Round 1: Mock review verdict

| Score | Value |
|---|---|
| Soundness | 2/4 |
| Presentation | 3/4 |
| Contribution | 2/4 |
| Confidence | 4/5 |
| Recommendation | **borderline reject** |

**With BIRD + seeds + reframe**: borderline accept / weak accept at ACL/EMNLP; still borderline at NeurIPS unless mechanism is unusually strong.

### Strengths
- Important practical problem (RL for tool-using SQL agents).
- Full Spider-dev evaluation (1034 examples), not sampled.
- SKE-RL is a reasonable, well-motivated intervention.
- Engineering appears careful (tested parsing, gating).

### Weaknesses
- Main causal claim overstrong: only two model sizes, one credit-assignment method.
- Single seed → +2pp gain hard to trust.
- No BIRD → Spider-only generality.
- G=8 degradation ambiguous without Dense G=8 control.
- 3B comparison partly relies on archived runs from a different launcher.

### Reviewer questions to preempt
1. Does Dense GRPO also degrade from G=4 to G=8?
2. Are gains concentrated in specific SQL/query classes?
3. Are improvements from format compliance or genuine reasoning?
4. KL / temperature / checkpoint sensitivity?
5. Spider train/dev contamination risk for Qwen2.5?

## Story A → Diagnostic Study reframe (concrete)

### New one-sentence headline
> Multi-turn Text-to-SQL RL gains are real but brittle: in our ReAct-GRPO pipeline, improvement appears only at 7B, structural advantage shaping fails to help, and the dominant bottleneck must be diagnosed through protocol, reward, and trajectory-level failure modes rather than assumed to be credit assignment.

### New §1 thesis
Not: *"Does SKE-RL solve credit assignment?"*
But: *"When ReAct-style GRPO helps or fails for Text-to-SQL, what observable bottleneck explains the outcome — model capacity, reward shape, tool/protocol compliance, rollout diversity, or structural credit assignment?"*

### New section structure
1. **Introduction** — multi-turn Text-to-SQL RL not plug-and-play; naive intuitions fail.
2. **Setup** — ReAct loop, SQLite execution, GRPO, EX_att/EX_all (compact).
3. **Finding 1: RL helps at 7B but not reliably at 3B** — load-bearing: 3B archived + current rerun, 7B Dense +2pp. **Do not call this "capacity law".**
4. **Finding 2: Reward shaping does not rescue the weak policy** — 3B reward variants. Caveat archived ones in main, full configs in appendix.
5. **Finding 3: Structural credit assignment is not the missing piece here** — SKE design + gates + G=4/G=8.
6. **Finding 4: What actually changes after RL** — *the most important section*: format/tool-call errors, invalid SQL, execution errors, repair-after-observation rate, per-query-class wins/losses.
7. **Discussion** — practical guidance: evaluate agentic RL with diagnostics, not only aggregate EX. SKE as useful negative control.
8. **Limitations** — single seed, Spider-heavy, two sizes, SKE only advantage-side.

### Survives without BIRD / third capacity point?
- **Workshop**: yes.
- **ACL/EMNLP main**: weak but possible only if Finding 4 diagnostics are strong.
- **NeurIPS/ICML main**: no — do not attempt without those additions.

## Ranked experiment plan (3 weeks, 1×24GB)

Cost unit: `1.0 run-eq = 24 GPU-h` (one 7B G=8 train at 75 steps). One full Spider-dev eval ≈ `0.21 run-eq`.

| Rank | Exp | Cost | Lift | Verdict |
|---:|---|---:|---|---|
| 1 | (h) Per-class diagnostic tables | ~0 | large | Do first. Highest value/hour. |
| 2 | (i) KL/entropy/reward-var/format-error curves | ~0 | large | Essential — turns anecdote into mechanism. |
| 3 | (j) Reasoning-vs-format-compliance partition | ~0–0.4 | large | Directly attacks biggest alternative explanation. |
| 4 | (c) Dense G=8 control | ~1.2 | large | Needed to interpret SKE G=8 collapse. |
| 5 | (a) 3B Dense same launcher | ~0.4–0.6 | medium | Re-anchors the 3B cliff. Cheap. |
| 6 | (g) 1.5B Dense + SFT | ~0.25–0.4 | small-med | Third capacity point. Mostly confirms low-end failure. |
| 7 | (e) 7B Dense BIRD-mini SFT+GRPO+eval | ~1.5–2.5 | med-large | Useful but mini-BIRD attackable as cherry-picked. |
| 8 | (d) Dense G=2 + SKE G=2 | ~0.7–0.9 | small | Nice curve, not essential. |
| 9 | (b) 7B Dense+SKE 3-seed × 4-cond | ~7–10 | large | **Too expensive** for 3 weeks/1 card. |
| 10 | (f) 14B Dense+SFT | ~3–5 + OOM risk | large if works | **Don't attempt** unless feasibility proven. |

**Highest-EV package under 3 weeks**: 1+2+3 immediately → 4 → 5 → 6 if time → 7 only if BIRD-mini already stable.

## The format-vs-reasoning experiment (highest-priority diagnostic)

Reviewer flagged this as the **most damaging alternative explanation**: Dense GRPO's +2pp may come from format/tool-call compliance, not SQL reasoning. Cheapest distinguishing experiment uses existing eval JSONs:

### Win/loss transition table
Classify each example into:
- `format_failure`: no valid final SQL/action/answer
- `tool_failure`: malformed action, exceeded turns, exec syntax fail
- `sql_executable_wrong`
- `sql_non_executable`
- `observation_repair`: first SQL wrong/error, later final correct
- `semantic_repair`: final SQL changes skeleton/tables/conditions after observation
- `no_repair_correct`: final SQL right without meaningful tool use

Then: of SFT-wrong / Dense-right examples, what fraction is each category?

### Lenient extraction sanity check
- Strict metric: current EX_all
- Lenient: extract any plausible final SQL from the transcript and execute it
- **If Dense gain disappears under lenient extraction → it was mostly protocol/format**
- If Dense gain remains on executable SQL and hard query classes → it is semantic

Cost: near-zero from existing logs; one re-eval pass per model otherwise.

### If Dense mostly fixes format/tool compliance
Reviewer's verdict: **more interesting, not less**. Inverts the field's assumption.
- Title: *"Agentic GRPO Improves Text-to-SQL by Aligning Tool-Use Protocols More Than SQL Reasoning"*
- Core argument: Dense GRPO improves EX_all by reducing invalid trajectories; EX_att or lenient SQL execution improves less. SKE fails because it targets SQL structure *after* valid SQL exists, while the bottleneck is earlier (producing valid, executable trajectories). Larger model helps because it has enough base competence to exploit protocol rewards.

## Can SKE negative result LEAD a paper?

Currently: **no, only supporting evidence**. To lead, needs mechanism. Four candidate mechanisms (the reviewer's list):

1. **Variance-reduction failure** — SKE doesn't reduce advantage variance enough to matter, or it does but shrinks useful signal.
2. **Class-fragmentation** — many classes singleton/small even at G=8; effective baseline size remains poor; more G adds low-quality diversity rather than within-class comparison.
3. **Wrong abstraction** — same skeleton contains semantically different difficulty (schema linking, literals); cross-skeleton comparison is actually informative.
4. **Optimization** — SKE changes advantage scale/sign distribution; more rollouts produce more near-zero advantages for exploratory-but-useful trajectories; KL/entropy curves show collapse.

Negative-with-mechanism narrative example:
> Skeleton-conditioned baselines fail because SQL skeleton is not a sufficient control variate for Text-to-SQL reward: the main variance comes from schema grounding, literal selection, and protocol validity, not AST shape.

This *can* lead a paper. "We tried X and it didn't work" cannot.

## Competitive landscape (reviewer-sourced; some arxiv IDs unverified)

The 2025-2026 Text-to-SQL RL space is crowded. **Cannot win on SOTA.** Differentiation must be on diagnostics + negative result with mechanism + tested infrastructure.

⚠️ The reviewer cited specific arxiv IDs; some look plausible (e.g. 2505.12768 = May 2025, 2510.25510 = Oct 2025) but at least one (2601.17699) appears to be a hallucination (Jan 2026 ID with no obvious match). **Verify each citation independently before using.**

### Direct competitors (named by reviewer — verify before citing)
- **ReEx-SQL** (arxiv:2505.12768): execution-aware RL, intermediate DB feedback, 7B Spider/BIRD gains.
- **MTIR-SQL** (arxiv:2510.25510): multi-turn tool-integrated RL, GRPO mods, no KL, 4B Spider/BIRD.
- **MARS-SQL** (arxiv:2511.01008): multi-agent ReAct + validation agent, multi-turn RL.
- **SQL-Trail** (arxiv:2601.17699): multi-turn RL with adaptive turn budget, 7B/14B. ⚠️ ID looks suspicious.

### Reward / structure competitors
- **Graph-Reward-SQL** (ACL Findings EMNLP 2025): graph-matching stepwise reward.
- **HES-SQL** (arxiv:2510.08896): GRPO + skeleton-completeness reward + latency reward. **Direct overlap** with skeleton-based guidance — must position SKE as advantage-side (not reward-side) to differentiate.
- **LearNAT** (arxiv:2504.02327): AST-guided decomposition + margin-aware RL/DPO.

### Other RL Text-to-SQL
- CSC-SQL (Findings IJCNLP 2025), ExeSQL (Findings EMNLP 2025), Sparks of Tabular Reasoning (TRL workshop 2025).

## Claims-vs-evidence matrix

| Claim | Current status | Needed |
|---|---|---|
| C1: 3B GRPO fails across rewards | Plausible, not clean | Current-launcher rerun of 3B Dense + best archived variant; archive others in appendix |
| C2: 7B Dense improves +2pp | Promising, not pub-stable | 3 seeds, confidence intervals, fixed checkpoint protocol |
| C3: SKE-G4 underperforms Dense | Defensible single-run | 2-3 seeds; same-G control for Dense |
| C4: G=8 worsens SKE | Interesting, not causal | Seeds; add G=2 or Dense G=8; report variance and class counts |
| C5: BIRD transfer | Unsupported | Run 7B SFT + Dense + maybe SKE-G4 on BIRD dev |
| C6: Multi-seed reproduction | Missing | 3 seeds for headline conditions |

Cost in 7B-run-equivalents:
- 3B current rerun package: ~1–1.5×
- 7B 3-seed Dense/SFT/SKE package: ~6–9×
- Dense G=8 control: ~2× (if rollout cost doubles)
- BIRD SFT+Dense+SKE: ~2–4×
- **Minimal paper-hardening package**: ~5–7×
- **Strong main-track package**: ~10–15×

## Three-week decision

Reviewer's pick: **(iv) workshop submission now + build v2 for main later**.

> "In 3 weeks on one 24GB card, you cannot produce the seeds/BIRD/third-scale evidence needed for a credible ACL/EMNLP main submission. The highest-EV move is to run the zero/low-cost diagnostics, add Dense G=8 and 3B Dense same-launcher, write the honest diagnostic paper, and release the toolkit. Use the workshop submission to establish priority and reviewer feedback, then decide whether v2 deserves BIRD + seeds + larger-scale compute."

## Action items (in execution order)

### Week 1 (zero/low-GPU diagnostics)
- [ ] Build win/loss transition table on existing 7B SFT vs 7B Dense75 eval JSONs (per-record categorization)
- [ ] Lenient-extraction sanity check (re-extract any plausible SQL from transcripts; re-execute)
- [ ] KL / entropy / reward-variance / format-error curves from existing training logs
- [ ] Per-class diagnostic table on existing eval JSONs (the `analyze_eval_by_skeleton.py` output)

### Week 2 (low-cost training)
- [ ] Dense G=8 control run (~1.2 run-eq)
- [ ] 3B Dense GRPO same-launcher rerun (~0.5 run-eq)
- [ ] Begin draft skeleton + Finding 1+2+3 sections

### Week 3 (write + optional adds)
- [ ] Finish draft including Finding 4 (the protocol diagnostic)
- [ ] Optional: 1.5B SFT+Dense (~0.4 run-eq) if time
- [ ] Optional: BIRD-mini if (a) loader is stable and (b) Week 1-2 results suggest it adds value
- [ ] Polish; pick workshop venue

### Workshop venue candidates (verify deadlines)
- NeurIPS 2026 workshops on Foundation Model Interventions / Agentic AI / Negative Results
- ICLR 2027 workshops (deadline typically ~Jan 2027)
- EMNLP/ACL workshops on RL for LLMs / agent evaluation
- TRL workshop (Tabular Reasoning) — direct fit given Sparks of Tabular Reasoning precedent

## Biggest unarticulated risk

**The +2pp 7B Dense gain may not be robust.** Possible failure modes:
- Seed noise
- Checkpoint selection bias
- EX_att / EX_all definition brittleness
- Spider-specific gain with no BIRD transfer
- Pretraining contamination
- **Dense GRPO improving formatting/tool-call compliance rather than reasoning** (the format-vs-reasoning experiment must be done early)
- SKE underperforming because of implementation/hyperparameter mismatch, not because credit assignment is irrelevant

If the +2pp shrinks under seeds or fails on BIRD, the paper becomes a negative engineering report. Still useful, but workshop-only.

## Auxiliary infrastructure as artifact

The ~10K LOC infrastructure (skeleton extractor + tests, class-aware advantage, BIRD nested-layout resolver, evidence-aware prompts, query-class tagger, checkpoint aggregator, dynamic-padding helper) is releaseable but **not as a standalone main-conference systems paper**. Best use: bundle as supplementary artifact for the empirical paper. Possibly a separate workshop/demo/toolkit paper if polished (pip install, examples, repro scripts, fixtures).

## Verbatim trace

Full Round 1 + Round 2 trace at `.aris/traces/research-review/2026-04-29_run01/round{1,2}.txt`. Resume the codex session via `codex resume 019dd6aa-5d15-7ed2-977a-6fdf5c9ac819` if more rounds are needed (e.g., to draft mechanism analysis for the SKE negative result, or to sketch §6 protocol-diagnostic table).
