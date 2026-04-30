---
title: Turn-Delta Credit Assignment for Multi-Turn GRPO in Text-to-SQL Agents
target: ACL 2027 main (8 pages + refs)
authors: TBD
status: skeleton
---

# Paper Skeleton

## Abstract (150 words)

Group Relative Policy Optimization (GRPO) has emerged as the dominant algorithm for RL-tuned Text-to-SQL agents. However, for **multi-turn ReAct agents** that iteratively interact with a database, the standard sequence-level reward suffers from *sparse credit assignment*: all tokens in a long trajectory receive the same advantage signal, even though only a few decisions actually determine correctness. We introduce three composable improvements for multi-turn GRPO: **(1) Turn-Delta Credit Assignment**, which rewards tokens in turns that improve executability; **(2) Entropy-Weighted Span GRPO**, which concentrates gradients on uncertain decisions; and **(3) Difficulty-Adaptive Compute**, which allocates more turns to hard queries. Applied to a 3B Qwen2.5 model on Spider, our methods achieve X% execution accuracy — a +Y pp improvement over vanilla GRPO and within Z pp of 7B baselines. Our analysis shows that each component contributes independently, with Turn-Delta providing the largest single gain on hard queries.

## 1 Introduction (1 page)

### Problem
- Text-to-SQL via ReAct agents: Thought → execute_sql → Observation → Answer
- GRPO trains policy on group-relative rewards
- **Key issue**: on hard queries, most rollouts fail → reward = 0.03 for 7/8 → near-zero variance → wasted training signal

### Existing work
- Arctic-R1 (Snowflake): simple exec reward, 7B = 70B SFT
- MARS-SQL: multi-agent GRPO (3 separate models)
- Reward-SQL, Graph-Reward-SQL: process rewards, AST matching
- MCTS-SQL, Alpha-SQL: test-time search with frozen models
- **Gap**: no method addresses credit assignment for the multi-turn ReAct agent specifically

### Our contribution
1. Turn-Delta Credit Assignment (TDCA): per-turn execution-grounded advantage redistribution
2. Entropy-Weighted Span GRPO (EWSG): decision-token-aware loss weighting
3. Difficulty-Adaptive Compute (DAC): variable max_turns by curriculum tier
4. Ablation showing each component's contribution
5. At 3B scale, achieve X% on Spider / Y% on BIRD

### Roadmap: Sec 2 (related), Sec 3 (method), Sec 4 (experiments), Sec 5 (analysis), Sec 6 (conclusion)

## 2 Related Work (0.5 page)

- **RL for Text-to-SQL**: Arctic-R1, MARS-SQL, MTIR-SQL, Reasoning-SQL
- **Credit assignment**: GTPO/GRPO-S (token-entropy), Reward-SQL (PRM), Execution-Grounded CA
- **Multi-turn agents**: ReAct (Yao et al.), Reflexion, ExeSQL
- **Test-time search**: Alpha-SQL, MCTS-SQL

## 3 Method (2 pages)

### 3.1 Background: Multi-turn GRPO
- Rollout: prompt → G generations → each with (token_i, gen_mask_i)
- Reward: r_i = reward_breakdown(gen_text_i, gold_sql, schema, db)
- Advantage: A_i = (r_i - mean(r)) / std(r)
- Loss: clipped PPO surrogate on mean log-prob

### 3.2 Turn-Delta Credit Assignment (TDCA)

**Motivation**: A trajectory succeeds because of a specific turn (the one that generates correct Answer or successful exec). Reward this turn more.

**Formalization**:
- Each model-generated segment gets a turn weight w_t ∈ {0.5, 0.8, 1.2, 1.5}:
  - 1.5 if the turn contains "Answer:"
  - 1.2 if action executed successfully
  - 0.5 if action failed
  - 0.8 otherwise (thought only)
- Token-level weights: `weight[t] = gen_mask[t] × w_turn(t)`
- Normalized so mean = 1.0 over model tokens

**Effect on loss**:
```
new_lp_mean = Σ (log π(y_t|s) × weight[t]) / Σ weight[t]
```

### 3.3 Entropy-Weighted Span GRPO (EWSG)

**Motivation**: High-entropy tokens = the model's uncertain decisions (table choice, join condition). Push gradients there.

**Implementation**:
- Compute token entropy: H_t = -Σ p(y|s_t) log p(y|s_t)
- Normalize to ∈ [0.5, 2.0]: `e_t = 0.5 + 1.5 × clip(H_t / H_mean, 0, 2)`
- Combined weight: `weight[t] × e_t`

### 3.4 Difficulty-Adaptive Compute (DAC)

**Motivation**: Easy queries need 1 turn; hard queries benefit from multi-turn exploration.

**Rule**: `max_turns(d) = base × m(d)`, m ∈ {0.4, 0.8, 1.4} for {easy, medium, hard}.

### 3.5 Composition
All three are multiplicative on the token weight:
```
weight[t] = gen_mask[t] × turn_weight(t) × entropy_weight(t)
```
Fully orthogonal — can ablate each independently.

## 4 Experiments (2 pages)

### Setup
- **Model**: Qwen2.5-3B-Instruct + ReAct SFT (500 steps on Spider train)
- **GRPO**: DeepSpeed ZeRO-2, G=8, LR=3e-6, KL coef 0.1, 300 steps
- **Benchmarks**:
  - Spider-dev (1034 samples): EM + EX, 3 difficulty tiers
  - BIRD-dev (1534 samples): EX, 3 difficulty tiers
- **Baselines**:
  - SFT (react-sft-v2)
  - GRPO (vanilla, no v5 methods) — = v4
  - Arctic-R1-7B (cited; or reproduced at 3B)
- **Seeds**: 3 (42, 123, 456), report mean ± std, paired t-test

### 4.1 Main Results (Table 1, Table 2)

[Spider-dev table — expect v5-all > v4 > SFT]
[BIRD-dev table — expect v5-all > v4 > SFT]

### 4.2 Ablation (Table 3)

| DAC | EWSG | TDCA | Spider EX | BIRD EX |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | (v4) | |
| ✓ | ✗ | ✗ | | |
| ✗ | ✓ | ✗ | | |
| ✗ | ✗ | ✓ | | |
| ✓ | ✓ | ✓ | | |

### 4.3 Per-difficulty Breakdown (Figure 2)

Bar chart: easy / medium / hard × 4 methods

### 4.4 Training Dynamics (Figure 1)

Reward curve vs steps for v4 vs v5-all, 3 seeds, mean ± 1σ band.

## 5 Analysis (1 page)

### 5.1 Why does Turn-Delta help?
- Show: v4 rewards=[0.03×7, 1.0×1] (bimodal) → v5 rewards=[0.03×1, 0.5-0.8×4, 1.0×3] (graded)
- Advantage variance increases → more useful gradient

### 5.2 Where does Entropy weighting concentrate?
- Analysis: top-10% entropy tokens are mostly table/column names and join conditions
- Shows the method targets the "right" decisions

### 5.3 Failure modes
- Hard queries where all 3 methods fail: pattern analysis
- Case studies from Spider-hard

## 6 Conclusion (0.25 page)

- TDCA + EWSG + DAC = composable, orthogonal improvements for multi-turn GRPO
- 3B model reaches X% on Spider — closes gap to 7B methods by Y pp
- Future: BIRD-test SOTA, MCTS+GRPO combination

## Limitations (0.5 page)

- 3B scale only — haven't tested 7B/13B
- Spider/BIRD only — SQL generalization unclear
- Requires real DB (vs simulator); expensive for huge DBs
- Reward hacking risks (mitigation: KL penalty verified)

## References

TBD — starting list in SOTA_BASELINES.md
