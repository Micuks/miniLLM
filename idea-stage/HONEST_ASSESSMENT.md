---
date: 2026-04-20
advisor: GPT-5.4 xhigh
recommendation: Pivot to Findings (Option A+C hybrid)
---

# Honest Assessment: Can We Reach ACL/EMNLP Main?

## Verdict

**Current path main acceptance probability: 3-5%**

"Best RL model below SFT baseline" is a **fatal** result for a methods paper. No rebuttal survives "your core contribution makes things worse."

## Probability Table

| Path | Venue | Probability |
|---|---|---|
| Current RL path, no changes | Main | **3-5%** |
| Fix RL to beat SFT +5pp / 3 seeds | Main | 10-15% |
| Option A (negative result RL) | Main | 8-12% |
| Option A (negative result RL) | Findings | 20-28% |
| Option C (ReAct SFT ablation study) | Main | 12-18% |
| **Option A+C hybrid** | **Findings** | **30-40%** |
| Workshop | Workshop | 65-75% |

## Recommended 2-Week Plan

### Week 1: Diagnostic Post-mortem (no new training)
1. Plot reward curves, entropy, KL divergence across v5-100 and v5-300 full 300 steps
2. Check for reward hacking patterns (short/trivial SQL passing execution)
3. Check gradient norms per turn type — does TDCA actually weight meaningfully?
4. Build clear evidence: is RL salvageable or not?

### Week 2: Decision Gate
- **If RL salvageable** (clear fix identified, +3pp EX on full dev achievable): run 3 seeds at optimized config, target main
- **If RL broken**: commit to Option A+C hybrid Findings paper

## Strongest Pivot: A+C Hybrid for Findings

**Paper: "Why Multi-Turn ReAct Agents Resist GRPO: An Empirical Study"**

Core contributions:
1. **Negative result**: GRPO on 3B Text-to-SQL ReAct agent does not beat SFT despite 3 new techniques (TDCA/EWSG/DAC) designed to address known issues
2. **Diagnostic analysis**: What specifically fails?
   - Reward hacking at step 120+ (reward clusters 0.62-0.75 while corr drops)
   - Sparse reward on hard queries causes bimodal training signal
   - LR scheduling mismatch for multi-turn GRPO
3. **ReAct SFT characterization**: per-difficulty, per-turn breakdown showing where SFT already captures ReAct behavior
4. **Practitioner guidance**: when NOT to use RL for agentic NLP tasks

This has real value to the community. Diagnostic papers on failure modes are cited frequently and are acceptable at Findings.

## Key Insight from Advisor

> "You have a good empirical dataset for a 'lessons learned' paper. That is not a consolation prize — diagnostic papers on why RL fails in structured prediction are cited frequently. The mistake would be spending 8 weeks trying to fix numbers that may not be fixable at this scale, and arriving at the deadline with nothing submittable."

## Action Items

- [ ] **STOP** new training runs
- [ ] Plot training curves (reward/entropy/KL/gradient norms) from v5-100 + v5-300 logs
- [ ] Analyze v5-300 checkpoint-180 outputs for reward hacking evidence
- [ ] Run diagnostic analysis scripts for 2-3 days
- [ ] Make Week 2 decision: continue RL or pivot to Findings
