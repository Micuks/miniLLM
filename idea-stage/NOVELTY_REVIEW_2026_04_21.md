---
date: 2026-04-21
type: novelty + positioning review
reviewer: Claude Sonnet 4.6 (senior-reviewer mode) + literature search
scope: Full project state, pivot direction, paper positioning
---

# Novelty, Collision & Paper Viability Review

## TL;DR

**The reward-hacking finding is REAL but PARTIALLY SCOOPED.** The single-turn SQL community has already documented dense-reward hacking (Reward-SQL, Arctic-R1, Think2SQL). Our differentiated contribution is the **multi-turn ReAct agent context** and the observation that **both dense AND sparse rewards fail**, just differently — dense → variance collapse/template-locking; sparse → bimodal signal / early peak then drift. If we frame the paper around this dual failure + diagnostic methodology rather than "dense bad, sparse good", we have a genuinely novel story that the existing single-turn literature cannot claim.

---

## 1. Literature Collision Assessment

### Papers that overlap with our findings

| Paper | arXiv | Finding | Overlap with us |
|---|---|---|---|
| Arctic-Text2SQL-R1 | 2505.20315 | Sparse exec reward beats complex dense; explicitly rejects proxy signals | HIGH — endorses our sparse-wins thesis from the other direction |
| Think2SQL | 2504.15077 | Dense rewards help **small** models; sparse better for large | MEDIUM — **cuts against us**: we're 3B (small) and dense hurts us |
| Reward-SQL | 2505.04671 | Dense PRM causes reward hacking (shorter outputs) in single-turn CTE SQL | HIGH — same mechanism, different setting (single-turn CTE, not multi-turn ReAct) |
| Reasoning-SQL | 2503.23157 | Format/structure rewards optimized before correctness (expected, not failure) | LOW — different framing; they treat it as a staged learning process |
| MO-GRPO | 2509.22047 | GRPO reward hacking in multi-objective settings; variance-normalized reweighting | MEDIUM — same phenomenon, different domain (translation/instruction) |
| Practitioner's Guide | 2510.01132 | Dense turn-level rewards accelerate multi-turn RL but stability varies | MEDIUM — our context, but their conclusion is more nuanced |
| Posterior-GRPO | 2508.05170 | Process reward supervision causes hacking; posterior gating fixes it | LOW — code generation, single-turn thinking, different mechanism |

### Critical scooping concern: Think2SQL

Think2SQL (arXiv:2504.15077) directly studies reward density for Text-to-SQL and finds **dense rewards help smaller models** while sparse rewards are better for larger models. We are at 3B — **this directly contradicts our empirical finding** that dense rewards hurt 3B-scale GRPO. This requires either:
(a) An explanation of why our setting differs (multi-turn ReAct vs single-turn generation — dense reward in multi-turn has a format-exploitation attractor that doesn't exist in single-turn)
(b) A direct experimental comparison showing our finding holds even accounting for model size

**This is the biggest gap to address before submission.**

---

## 2. What IS Novel

Despite the overlap, our contribution has genuinely new elements:

### 2a. Multi-Turn ReAct Agent Context (STRONGEST)
All prior dense-vs-sparse work (Arctic, Reward-SQL, Think2SQL) uses **single-turn** SQL generation. Our setting is different:
- Trajectories are 3-5 turns: Thought → execute_sql → Observation → Answer
- Format rewards include Thought/Action/Observation structure — not just SQL validity
- The "format-reward floor" is structurally different: a model can get ~0.69 total reward just by outputting a well-formed ReAct trajectory with any plausible-looking SQL
- This creates a much stronger exploit attractor than single-turn SQL format rewards

**Claim we can make**: Dense rewards in multi-turn ReAct agents create a **richer set of format-exploitation attractors** because the multi-step format (Thought/Action/Observation/Answer) provides multiple independent format signals that can be satisfied independently of SQL correctness.

### 2b. Dual Failure Story (NEW)
Our sparse control experiment (completed at step 100) showed:
- **Dense**: variance collapse → template-locking (gap=0.89, var=0.01, all 8 rollouts at 0.62-0.63)
- **Sparse**: no variance collapse, but bimodal signal (rewards are 0 or 1) → peak corr=0.80 at step 20, then drift to 0.25

This is NOT the story in the literature. The literature implies "sparse good, dense bad." Our data shows **both fail**, with qualitatively different failure modes. This dual failure + comparison is genuinely new for multi-turn agents.

### 2c. Diagnostic Metrics (NEW, METHODOLOGICALLY)
Our proposed diagnostic triplet:
1. **Shallow-correctness gap**: (fmt+val+struct)/3 - corr > 0.5 = hacking signal
2. **Rollout variance collapse**: rmax - rmin < 0.2 = template-locking
3. **Combined signature**: gap > 0.5 AND var < 0.2 = confirmed exploit

No prior paper proposes this as a diagnostic tool. This is a methodological contribution with practical utility (early stopping criterion).

### 2d. Acceleration Effect of v5 Enhancements (NEW INSIGHT)
Our paper's key irony: 3 techniques designed to extract more training signal (TDCA, EWSG, DAC) actually **accelerate reward hacking** via compounding gradient effects. This is a novel negative-result insight that would genuinely help practitioners.

---

## 3. Paper Viability Assessment

### Option A: "Dense Rewards Fail Multi-Turn ReAct Agents" (Findings paper)
**Strength**: Clear negative result + diagnostic contribution  
**Weakness**: Think2SQL says dense helps small models — we need to address this  
**Required evidence**:
- Dense vs sparse training curves (we have both ✓)
- Diagnostic metric applied to both (partially done ✓)
- Explanation of why Think2SQL doesn't apply to our setting (analysis needed)
- Eval comparison: sparse ck25/75/100 vs dense v5 on Spider 200 (running now ✓)

**Acceptance probability at ACL Findings**: 20-28% (from HONEST_ASSESSMENT.md)

### Option B: "Reward Hacking as Dominant Failure Mode in Multi-Turn GRPO" (Findings paper, stronger framing)
**Key reframe**: Don't claim "dense bad, sparse good" — claim "both fail differently in multi-turn agents." This is harder to contest because we have evidence for both failure modes.

**Story arc**:
1. Multi-turn Text-to-SQL ReAct agents fail to improve over SFT via GRPO → "why?"
2. Dense reward: format-exploitation attractor → variance collapse → corr=0 (step 175)
3. Sparse reward: no collapse, but bimodal gradients → corr peaks early then drifts
4. Neither is satisfactory at 3B scale — this is the key empirical contribution
5. Proposed fixes: multiplicative reward (correctness is necessary condition), posterior gating on format reward, early stopping on corr

**This framing is more honest, harder to reject, and potentially more impactful.**

**Acceptance probability at ACL Findings**: 30-38% (estimated uplift from dual-failure evidence)

### Option C: Pivot to EMNLP 2025 Workshop (if insufficient evidence by deadline)
- Target: **RepL4NLP or NewSumm or similar** (lower bar)
- Workshop papers can make preliminary negative results claims without 3-seed significance
- Timeline: much more forgiving

---

## 4. What's Missing Before Submission

### Must-have (for any Findings submission)
1. **Sparse eval results** (ck25, ck75, ck100 on Spider 200) — running now
2. **Explanation of Think2SQL discrepancy** — why dense hurts us at 3B despite Think2SQL's prediction
3. **Diagnostic metrics applied to sparse run** — does variance stay high? Does gap stay low?
4. **Full training curves** for both dense and sparse (extract from logs to CSV/plots)

### Should-have (uplift acceptance probability)
5. **SFT training curves** — what does SFT training dynamics look like? (no reward hacking by design)
6. **Rollout output analysis** — what is the "template" the model collapses to? Show 2-3 concrete examples
7. **Gradient norm analysis** — does TDCA/EWSG compound gradient magnitude? (confirms acceleration claim)

### Nice-to-have (for completeness)
8. **Think2SQL comparison experiment** — run their setup at 3B to reproduce their dense-helps-small finding, then explain the difference
9. **2 more seeds** for dense training (credible mean ± std)

---

## 5. Recommended Paper Outline (Option B framing)

**Title**: "When Both Dense and Sparse Rewards Fail: Reward Hacking Patterns in Multi-Turn GRPO for Text-to-SQL Agents"

**Abstract**: Multi-turn Text-to-SQL agents trained with GRPO fail to improve over SFT baselines at 3B scale. We identify two distinct failure modes: (1) dense rewards create a format-exploitation attractor that causes variance collapse by step 100-175; (2) sparse binary rewards avoid collapse but produce bimodal gradients that peak early and drift. We propose diagnostic metrics — shallow-correctness gap and rollout variance — that identify hacking before evaluation. Our analysis shows that techniques designed to densify gradient signal (TDCA, EWSG, DAC) accelerate dense-reward hacking. We provide practitioner guidance on reward design for multi-turn agent RL.

**Sections**:
1. Intro: RL for Text-to-SQL works at 7B+ but fails at 3B; we study why
2. Background: GRPO, ReAct, reward composition in multi-turn agents
3. Experimental setup: Qwen2.5-3B, Spider, dense vs sparse, all ablations
4. Failure mode 1: Dense reward → variance collapse (step 175 evidence)
5. Failure mode 2: Sparse reward → bimodal gradient → early peak + drift
6. Diagnostic metrics: Shallow-correctness gap + variance as early warning
7. Why existing fixes (TDCA/EWSG/DAC) backfire: compounding gradients
8. Proposed fixes + limitations
9. Related work: position vs Think2SQL, Arctic-R1, Reward-SQL
10. Conclusion

**Target**: ACL 2027 Findings (deadline typically Nov-Dec 2026)

---

## 6. Immediate Action Priority

**This week (while GPU is running sparse evals)**:
1. ~~Launch sparse ck25 eval~~ ✓ (running)
2. Extract training curves for both dense (v5-300) and sparse to CSV
3. Apply diagnostic metrics to sparse training log
4. Draft Section 4 + 5 (failure modes) with concrete evidence
5. Pull rollout examples showing the "template" collapse at step 175

**Next week**:
6. Launch sparse ck75 + ck100 evals
7. Write diagnostic metric formalization (can be a brief subsection)
8. Address Think2SQL discrepancy (key refutation to prepare)
9. Draft full paper skeleton in new framing

---

## 7. Verdict

**Collision risk**: MEDIUM. The single-turn community has noted dense-reward problems. We are not first to observe this.

**Differentiation**: GOOD, if we execute the dual-failure story + multi-turn framing correctly. The "both fail differently" narrative is genuinely new.

**Novelty**: SUFFICIENT for Findings, NOT sufficient for Main without 3 seeds + beating SFT + matched-LR control.

**Recommended action**: Continue current direction (sparse eval → dual failure story), write in Option B framing. Do NOT invest more GPU time in trying to beat SFT — that ship has sailed for this submission cycle. Invest analysis time in understanding and articulating the failure modes.
