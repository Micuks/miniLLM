## CATS

### 1. Novelty score: 5/10

The idea is no longer cleanly novel. The classical root is already obvious: COMA (Foerster et al., AAAI 2018) made counterfactual credit assignment mainstream in RL, and Model-Free Counterfactual Credit Assignment (Mesnard et al., ICLR 2021) pushed the same agenda for variance reduction via future-conditioned counterfactual reasoning. More importantly for April 24, 2026, there is now a directly threatening LLM-era paper: **Counterfactual Credit Assignment for Policy Optimization** (Khandoga et al., published March 3, 2026 on OpenReview SPOT), which explicitly identifies important reasoning spans by counterfactual masking and then upweights them during policy optimization. That is uncomfortably close in spirit to “ablate a step, measure performance drop, reweight credit.”

What still keeps CATS above outright rejection is the intervention granularity. Khandoga et al. operate on reasoning spans/tokens and answer probability, whereas CATS intervenes on full **action-turns** in a multi-turn ReAct trajectory and measures downstream **execution success** under a fixed environment. In Text-to-SQL, that is a meaningful adaptation, because turn-level tool calls and execution observations are structurally different from pure CoT spans. But as a NeurIPS/ICML novelty claim, this is now a specialized instantiation of a known counterfactual-credit theme, not a fresh family.

### 2. Closest prior work potentially missed

**Counterfactual Credit Assignment for Policy Optimization** (Khandoga et al., 2026) is the closest miss. If a reviewer has seen it, they can say your core move is already established: counterfactually perturb a generated reasoning unit, estimate importance from the induced performance drop, and reweight policy optimization accordingly. I would also expect some reviewers to cite **Abstract Counterfactuals for Language Model Agents** (Pona et al., NeurIPS 2025) as supporting evidence that counterfactual analysis for LM-agent actions is already an active line, even if that paper is more about analysis than RL training.

### 3. Single deadliest attack

“Your pivotality score is not causal credit; it is off-manifold fragility. Teacher-forcing the prefix and resampling one turn from `pi_old` creates unnatural continuations, so the measured drop mostly tracks entropy, lexical brittleness, or cache-conditioning artifacts rather than true contribution.”

That attack is deadly because it goes straight at identifiability. If this lands, CATS becomes a heuristic ablation weight, not a causal-credit paper.

### 4. Verdict

**Conditional.** I can imagine acceptance only if you beat the obvious attacks head-on: entropy-matched resampling controls, paraphrase-preserving same-intent interventions, comparison against simple entropy weighting, and evidence that turn surgery predicts execution-critical turns better than token-level counterfactual baselines.

## TIPS

### 1. Novelty score: 4/10

This area filled up fast in 2025-2026, and your current formulation sits too close to that wave. The strongest overlap is **ST-PPO: Stabilized Off-Policy Proximal Policy Optimization for Multi-Turn Agents** (submitted September 18, 2025; visible on OpenReview for ICLR 2026), which explicitly argues that **token-level importance sampling is misaligned with multi-turn environments** and introduces **turn-level importance sampling** as one of its two core stabilization mechanisms. That is already most of your headline. Separately, the “multi-turn agent RL should operate at turn granularity rather than trajectory granularity” story is now heavily occupied by the Wei/Zeng/Hong line of work on turn-level reward design and credit assignment (2025 workshop paper, 2025 NeurIPS MTI-LLM poster, 2026 ICLR submission), plus GTPO/Turn-PPO style formulations in adjacent agent settings.

Your specific twist is “factor the GRPO objective with per-turn PPO clipping and stop-gradient on deterministic observations.” That is technically respectable, but it reads like an implementation-specific refinement of the same turn-level PPO/GRPO migration that others are already making. Classical RL also weakens the claim: per-decision importance sampling is old, and variance-reduction through exploiting structure in factored decision processes is not new either.

### 2. Closest prior work potentially missed

**ST-PPO** is the direct threat. Its abstract already says token-level IS is misaligned in multi-turn agents and that turn-level importance sampling is the fix. A second, more superficial problem: the acronym **TIPS** is already taken by **TIPS: Turn-level Information-Potential Reward Shaping for Search-Augmented LLMs** (Xie et al., ICLR 2026 poster). Different mechanism, but the naming collision will not help.

### 3. Single deadliest attack

“This is Turn-PPO/ST-PPO rewritten in GRPO notation. Stop-gradient on `s_t` is an implementation detail, not a paper-level algorithmic contribution.”

If a reviewer writes that sentence, the novelty discussion is basically over unless you have a theorem or a genuinely new estimator.

### 4. Verdict

**No.** As written, I do not think this clears the novelty bar for NeurIPS/ICML, even with good experiments. The contribution is too easy to collapse into existing turn-level PPO stabilization work.

## EPR

### 1. Novelty score: 6/10

This is the only one that currently looks like it can plausibly clear the line. The relevant prior-work buckets are all real, but none of them closes the gap completely. **Let’s Verify Step by Step** (Lightman et al., ICLR 2024) and the broader PRM literature established process supervision. **Free Process Rewards without Process Labels** (Yuan et al., ICML 2025) showed that response-level labels can induce implicit process rewards without explicit process annotation. **Process-based Self-Rewarding Language Models** (Zhang et al., 2025) extended the self-rewarding paradigm toward stepwise signals. In Text-to-SQL specifically, **Graph-Reward-SQL** (EMNLP Findings 2025) introduced stepwise reward modeling, and execution-guided/self-corrective systems such as **LitE-SQL** (EACL Findings 2026) already use execution feedback in the pipeline.

But EPR’s exact move is still distinct: before rollout, ask the frozen model to predict the **expected execution result shape**, then reward prefixes whose execution traces move toward that self-elicited structural target. That is neither standard PRM, nor standard self-rewarding, nor ordinary execution guidance. It is a self-generated, execution-grounded, structurally coarse process target. I do **not** know a direct prior paper that already does this exact thing for Text-to-SQL or agent RL. That uncertainty matters: I am not claiming the idea is uncontested, only that I do not know a closer match.

The reason I stop at 6 rather than 7 is that the self-rewarding family has already normalized “model supplies its own supervision,” so the paper will not feel unprecedented. The novelty is in the **target type** and the **execution-prefix coupling**, not in self-supervision itself.

### 2. Closest prior work potentially missed

The closest conceptual miss is probably **Process-based Self-Rewarding Language Models** (Zhang et al., 2025): same general direction of self-generated, process-level supervision. On the SQL side, a reviewer could also invoke **Graph-Reward-SQL** as evidence that stepwise Text-to-SQL reward design is already live, even though its mechanism is different and much more label-anchored.

### 3. Single deadliest attack

“You are rewarding the model for agreeing with its own hallucination. If the self-elicited target shape is wrong, your dense reward becomes anti-learning, and any gains may come from generic regularization rather than meaningful process supervision.”

That is the right attack because it questions not just novelty, but whether the mechanism is epistemically sane.

### 4. Verdict

**Yes.** With the right empirical support, this is the one I would let into the room. But the support has to be unusually sharp: accuracy of the self-elicited shape predictor, calibration of structural targets, ablations against random/wrong targets, and evidence that prefix reward helps more when the predicted shape is correct than when it is merely plausible.

## PIB

### 1. Novelty score: 4/10

The classical root is **VIME** (Houthooft et al., NeurIPS 2016): exploration bonus from information gain over uncertain dynamics. On its own that would not kill PIB; importing VIME-style information gain into multi-turn LLM agents could still have been fresh. The problem is that the 2026 agent-RL literature moved there already. The biggest threat is **Information Gain-based Policy Optimization (IGPO)** (published January 26, 2026; ICLR 2026 poster), which explicitly models each turn as incremental information acquisition and defines turn-level rewards from the marginal increase in probability of the correct answer. There is also **InfoPO** (published March 2, 2026), which frames multi-turn interaction as uncertainty reduction and rewards turns whose feedback changes later action distributions relative to a masked-feedback counterfactual. Even **CDE** (2025/2026) contributes to the sense that intrinsic exploration bonuses for LLM RL are no longer exotic.

PIB’s own delta is narrower: rather than answer-probability gain or perplexity/value variance, it uses **entropy collapse over sampled success/fail lookahead continuations**. That is a variant of the same information-gain/exploration family, not a new family. Worse, the proposal calls it a “posterior” but does not actually maintain a serious posterior; it estimates one from a tiny Monte Carlo tree of model completions. That weakens both novelty and credibility.

### 2. Closest prior work potentially missed

**IGPO** is the cleanest missed prior. If a reviewer knows that paper, they can say the big idea “credit turns for reducing uncertainty about the right final answer” is already taken in multi-turn agent RL. **InfoPO** is an equally relevant supporting citation because it also operationalizes information gain for turn-level credit assignment via counterfactual masking.

### 3. Single deadliest attack

“This is not posterior information gain. It is entropy over a tiny, highly biased set of sampled continuations, so the bonus is dominated by Monte Carlo noise and greedy-completion artifacts.”

That line is devastating because it attacks the mathematical honesty of the framing.

### 4. Verdict

**No.** Too much overlap with 2026 information-gain agent RL, and the proposed estimator looks noisy enough that even strong experiments may be read as lucky shaping rather than a principled contribution.

## Final Ranked Recommendation

Ranked on novelty for a **3B multi-turn ReAct GRPO Text-to-SQL** paper:

1. **EPR** — 6/10
2. **CATS** — 5/10
3. **TIPS** — 4/10
4. **PIB** — 4/10

### Which ideas reach the 6+/10 threshold?

Only **EPR** reaches it, and barely. It is the only one whose central mechanism still feels meaningfully under-explored after accounting for 2025-2026 process-reward and self-rewarding work.

### Minimal pivots for the 4-5/10 ideas

**CATS (5/10):** the minimal pivot is to stop presenting it as heuristic “pivotality” and instead make it an actual **interventional estimator**. Concretely: match counterfactual replacements by action type and entropy, estimate an average treatment effect on downstream execution-state transitions rather than just terminal success, and show invariance to paraphrase-level replacements. That changes the claim from “ablation weight” to “causal turn-effect estimator.”

**TIPS (4/10):** the minimal pivot is to exploit determinism in a way ST-PPO does not already claim. The cleanest route is a **GRPO-specific marginalized estimator** with a theorem: under deterministic tool/environment observations, show lower variance than sequence-level GRPO and isolate exactly what is stop-grad’ed and why. Without that theorem-plus-baseline package, this remains rebranding.

**PIB (4/10):** the minimal pivot is to make the uncertainty object **Text-to-SQL specific** rather than generic answer success. For example, maintain a posterior over schema-link hypotheses or SQL skeleton equivalence classes, then reward turns that eliminate hypothesis mass. That is much harder to dismiss as “another info-gain reward for agents.”

### Deadliest empirical experiment for the 6+/10 idea

For **EPR**, the killer experiment is an **adversarial target-shape validity study**. On the same training setup, compare:

1. self-elicited target shape,
2. deliberately corrupted target shape,
3. random target shape,
4. oracle target shape if you can define one without contaminating training.

Then stratify performance by whether the initial self-elicited shape was accurate. If EPR only helps when the predicted target is genuinely informative, you have a mechanism. If gains survive equally well under corrupted/random targets, a hostile reviewer will conclude the method is just dense noise regularization wearing a process-reward costume.

Bottom line: **push EPR first**. Keep **CATS** alive only if you are willing to harden it into a real causal-estimation story. **TIPS** and **PIB** are both already too occupied by adjacent 2025-2026 work to be safe lead ideas without a sharper pivot.
