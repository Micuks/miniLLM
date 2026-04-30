This R3 pass is better targeted than R2. It actually attacks the right weaknesses. But “better targeted” is not the same as “novel enough for NeurIPS/ICML 2026.” The bar here is not “interesting mechanism” or “could improve a 3B Text-to-SQL agent.” The bar is whether a hostile reviewer, already aware of the 2025-2026 process-reward and credit-assignment wave, would feel they are seeing a genuinely new methodological object rather than a sharpened variant of an occupied line.

## EPR-X

### 1. Novelty score: 6/10

This is an incremental but respectable advance over the old EPR, not a clean new family. The closest background is now crowded: process supervision and verifier-guided dense rewards are no longer exotic after **Let’s Verify Step by Step** (Lightman et al., 2023/ICLR 2024), **Free Process Rewards without Process Labels** (Yuan et al., ICML 2025), and **Process-based Self-Rewarding Language Models** (Zhang et al., ACL Findings 2025). On the Text-to-SQL side, stepwise or partial reward design is already active in **Reasoning-SQL** (COLM 2025), **Reward-SQL** (arXiv 2025), and **Graph-Reward-SQL** (EMNLP Findings 2025). So the generic claim “we give structured intermediate rewards for SQL reasoning” is already dead.

What keeps EPR-X at 6 instead of 5 is the exact coupling: self-elicited structural target, adversarial target controls in the same minibatch, and a falsifiability gate that explicitly tries to block rewarding self-hallucinated structure. That package is more disciplined than standard PRM dressing. But it is still fundamentally a shaped reward built from a low-dimensional target template plus a hand-built verifier. A reviewer will not experience this as a conceptual break. They will experience it as “process reward made less sloppy.”

### 2. Closest prior work potentially missed

The most dangerous prior bucket is not one paper but the convergence of three lines: **implicit process reward modeling** from outcome labels, **process-based self-rewarding**, and **Text-to-SQL process reward frameworks** such as Reward-SQL and Graph-Reward-SQL. If I had to name the single closest miss, it is probably **Reward-SQL: Boosting Text-to-SQL via Stepwise Reasoning and Process-Supervised Rewards** (Zhang et al., 2025), because it already makes process reward central in Text-to-SQL RL. I did not find a paper that already combines self-elicited target shape, falsifier gating, and adversarial target contrasts in this exact way. Search on that exact combination was inconclusive.

### 3. Single deadliest attack

“Your self-elicited target is decorative. The gains come from the hand-coded falsifier and from contrastive supervision against oracle/corrupt/random targets. In other words, this is schema-aware structural distillation wearing an RL badge.”

That attack is deadly because it collapses the claimed contribution from “new reward mechanism” to “engineered validity prior plus auxiliary ranking loss.”

### 4. Verdict at NeurIPS/ICML 2026 with strong empirical support

Still below the bar as a standalone novelty claim. Strong experiments could make it publishable as a systems-heavy Text-to-SQL paper or a compelling component inside a larger paper. They do not magically turn it into a 7/10 novelty contribution.

### 5. Minimal pivot to push to 7+

You need to prove the self-elicited target is the irreducible source of gain. Minimal pivot: replace the heuristic falsifier with a learned or formally specified schema-grounded validity checker that is frozen across methods, and show a compute-matched decomposition where falsifier-only, contrastive-only, and full EPR-X separate cleanly. If the self-elicited target still dominates after that, the story gets much harder to dismiss.

## ITE

### 1. Novelty score: 5/10

This is the most intellectually serious version of the old CATS idea, but the area around it is now heavily occupied. **VinePPO** (Kazemnejad et al., ICML 2025) made “credit assignment in LLM RL actually matters” mainstream. **Abstract Counterfactuals for Language Model Agents** (Pona et al., NeurIPS 2025) already argued that token-level interventions are the wrong abstraction for LM agents. Then 2026 got even worse for your novelty position: **Counterfactual Credit Assignment for Policy Optimization** (Khandoga et al., March 2026) explicitly does counterfactual masking to estimate causal importance, and **SSVPO** (ICLR 2026) pushes step-level credit assignment with stronger fairness language and theory. Add **SCAR** and the broader Shapley/segment-credit literature, and the landscape is already saturated with “don’t give every step equal credit.”

ITE’s remaining novelty is narrow: intervention at the level of ReAct turns rather than tokens, paraphrase-controlled same-intent counterfactuals rather than masking, and execution-state outcomes rather than terminal correctness only. That is a meaningful specialization for Text-to-SQL agents. It is not enough to look like a new family. It looks like the obvious “agentified” refinement of a line that already exists.

### 2. Closest prior work potentially missed

The closest single threat is **Counterfactual Credit Assignment for Policy Optimization** (Khandoga et al., 2026). A second serious threat is **Abstract Counterfactuals for Language Model Agents** (Pona et al., 2025), because it directly occupies the “high-level intervention on LM-agent actions” rationale. I did not find a direct prior already using paraphrase-preserving interventions plus execution-state ATE on multi-turn Text-to-SQL trajectories. That exact corner appears open. The problem is that it is a corner of an already crowded room.

### 3. Single deadliest attack

“Your intervention is not a causal intervention on decision content. It is a fragile language perturbation chosen by the same policy, so the estimated effect mostly measures prompt sensitivity and manifold drift, not the contribution of the underlying act.”

That is lethal because if the paraphrase filter fails to preserve latent intent, the whole causal estimator collapses into stylistic robustness analysis.

### 4. Verdict at NeurIPS/ICML 2026 with strong empirical support

Reject on novelty if sold as a new causal turn-credit method. Borderline accept only if reframed as a domain-specific estimator for agentic Text-to-SQL with unusually strong identifiability evidence. Right now the novelty claim is too easy to absorb into the 2025-2026 credit-assignment wave.

### 5. Minimal pivot to push to 7+

The minimal viable pivot is to stop intervening in raw language and intervene in an explicit abstract action representation: e.g. executable act schemas or normalized reasoning-act tuples with paraphrases only used as realizations. Then show that the estimator predicts execution-critical turns better than token masking, VinePPO-style rollout credit, and Shapley baselines. Without that abstraction layer, you are still stuck in “counterfactual wording tricks.”

## SKE-RL

### 1. Novelty score: 7/10

This is the first one that actually smells like a paper, not a repair patch.

The relevant prior work is real and should scare you: **RESDSQL** (AAAI 2023) already decouples skeleton parsing from schema linking in Text-to-SQL; **RB-SQL** uses SQL skeleton retrieval; **Plan-of-SQLs** pushes structured intermediate plans; **Reward-SQL** and **Graph-Reward-SQL** already reward structured decompositions such as CTE subqueries; and outside Text-to-SQL, **Latent Programmer** (ICML 2021) and broader discrete-latent program synthesis work normalize the idea of predicting a high-level latent structure before filling in details. So “use skeletons” is old. “Use latent program structure” is old. “Do RL on structured abstractions” is not unprecedented in spirit.

What I did not find, and what matters, is the exact move: define a SQL skeleton equivalence class, share GRPO advantage at the class level, and give partial structural credit through within-class feasible filling search. That is not standard Text-to-SQL skeleton prompting, not standard latent-program prediction, and not standard process reward modeling. It is a specific RL object: optimize over structural equivalence classes instead of raw strings. That is a real novelty claim.

I am not giving it higher than 7 because the “equivalence class” depends completely on your extractor. If the equivalence relation is brittle or too permissive, the elegant formulation becomes engineering smoke.

### 2. Closest prior work potentially missed

The closest miss inside Text-to-SQL is probably the combined pressure from **RESDSQL**, **Reward-SQL**, and **Graph-Reward-SQL**: they already tell a reviewer that SQL structure should be decomposed and rewarded explicitly. The closest conceptual miss outside Text-to-SQL is **Latent Programmer**: discrete latent structural codes before full program generation. I did not find a paper already doing skeleton-equivalence-class policy optimization or class-shared advantage estimation for SQL. Search for exact matches was inconclusive, which in this case helps you.

### 3. Single deadliest attack

“Your equivalence classes are fake. Queries with the same stripped skeleton can differ wildly in semantic difficulty, join pathology, and execution behavior, so class-shared advantage smears credit across non-equivalent programs. The max-over-fillings term then turns the whole method into latent search with reward hacking by inner-loop sampling.”

That is the right attack because it targets both the mathematical honesty of the equivalence relation and the possibility that gains come from inner search budget, not from the RL formulation.

### 4. Verdict at NeurIPS/ICML 2026 with strong empirical support

Conditional accept territory. Unlike EPR-X and ITE, this could survive as a standalone contribution if the paper is ruthless about extractor definition, compute-matched ablations, and failure analysis on complex BIRD queries. If the evidence is weak, it dies instantly. But this is the only R3 idea that has a plausible path to “this is actually a new optimization unit.”

## Final Ranking

Novelty ranking:

1. **SKE-RL** — 7/10
2. **EPR-X** — 6/10
3. **ITE** — 5/10

Which idea truly reaches 7/10 and is publishable as a standalone contribution? **SKE-RL only.** That does not mean it is safest. It means it is the only one whose central abstraction is not already mostly claimed by adjacent 2025-2026 work.

For the highest-scoring idea, the single most informative experiment is a **compute-matched factorial isolation study** on BIRD:

1. raw SQL GRPO,
2. skeleton-first generation with identical filler/search budget but no class-shared advantage,
3. class-shared advantage with `K=1` filler search,
4. full SKE-RL.

Then report results stratified by skeleton frequency, literal variation, and query complexity. If only the full method wins after search budget and decomposition are held constant, you defend the novelty claim. If the gain appears as soon as you add skeletonization or inner fill search, then the “equivalence-class RL” story is mostly branding.

If SKE-RL hits 7+, the AST-template-coverage assumption is **not realistically clean for BIRD**. Spider is template-friendlier. BIRD is not. Once you include nested aggregation, correlated subqueries, date arithmetic, HAVING logic, set operators, and function-heavy predicates, a strict AST-template family will miss too much. But if you loosen the extractor enough to claim >95% coverage, the equivalence classes become semantically dirty and the advantage-sharing story weakens. My brutal read: for BIRD, high coverage and clean equivalence are in direct tension. You can probably get one. You do not get both for free.
