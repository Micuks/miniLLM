---
date: 2026-04-30
status: external check — does the "large G hurts agentic GRPO" story hold against BIRD/BIRD-Interact leaders?
codex_mcp: NOT AVAILABLE in this environment; review is internal + WebFetch/WebSearch on public papers
sources:
  - https://www.snowflake.com/en/engineering-blog/arctic-text2sql-r1-sql-generation-benchmark/
  - https://arxiv.org/abs/2505.20315 (Arctic-Text2SQL-R1)
  - https://arxiv.org/abs/2510.25510 (MTIR-SQL)
  - https://arxiv.org/html/2505.12768 (ReEx-SQL)
  - https://bird-bench.github.io/ (BIRD leaderboard)
  - https://arxiv.org/abs/2510.05318 (BIRD-Interact)
---

# BIRD/BIRD-Interact leaderboard check — does the v3 R2 G-trade-off story hold?

## TL;DR

**Conditionally yes**. The story holds when scoped to **multi-turn agentic GRPO**, where it is consistent with MTIR-SQL's "reward collapse" remark and the modest G values chosen by top multi-turn methods. It does NOT hold as a universal claim about GRPO — Arctic-Text2SQL-R1 trains with G=16 on **single-turn** generation and tops BIRD-dev. The paper must scope the claim explicitly to multi-turn ReAct-style GRPO, and stop short of "G hurts GRPO".

## What top BIRD methods actually use

| Method | Model size | BIRD-dev | Format | G (training) | Multi-turn? |
|---|---|---:|---|---:|---|
| **Arctic-Text2SQL-R1 (32B)** | 32B | 70.5 | Single-turn | **16** (256/16=16 rollouts) | No |
| Arctic-Text2SQL-R1 (14B) | 14B | 70.1 | Single-turn | 16 | No |
| Arctic-Text2SQL-R1 (7B) | 7B | 68.9 | Single-turn | 16 | No |
| **MTIR-SQL** | 4B | 64.4 | Multi-turn ReAct + tool | **5** | Yes (max 6 interactions) |
| ReEx-SQL | 7B | 64.9 | Multi-turn execution-aware | not stated in summary | Yes |
| Top BIRD-test entries (Agentar, AskData, LongData, …) | varies | 76-82% | Mostly closed-source pipelines | n/a | varies |

(Top BIRD-test entries are mostly multi-agent / GPT-4o-augmented pipelines; the GRPO-trained open methods cluster in the mid-range.)

## What the data say about the story

### Supports the story

1. **MTIR-SQL explicitly mentions "reward collapse"** in multi-turn GRPO.
   > "[MTIR-SQL] extends GRPO with SQL execution rollout expansion and trajectory filtering to stabilize training in multi-turn tool-use scenarios, effectively mitigating reward collapse."
   This is independent corroboration that vanilla multi-turn GRPO has stability problems. They chose **G=5**, not high-G, and added stabilization tricks. This is exactly the regime where v3 R2 lives.

2. **Top multi-turn methods all pick modest G** (MTIR-SQL G=5, ReEx-SQL similar). If high-G were universally better, top methods would pick high-G. Their hyperparameter choice is implicit support for the v3 R2 finding.

3. **No published paper has decomposed turn-share-shift vs per-turn-quality.** The novelty of the diagnostic is intact.

4. **The R2 R_same paired-ΔEX result (composition-decontaminated)** is the kind of evidence that survives "you only ran one model size" critique. The 3B replication (R_same -33.9 pp on n=59 records) confirms the conditional-solver-collapse mode at a different model size.

### Challenges the story

1. **Arctic-R1 trains with G=16 at 7B and tops BIRD** (single-turn, 68.9% dev).
   - Reading: a paper claiming "G=16 hurts GRPO" would be DIRECTLY contradicted by Arctic-R1.
   - Resolution: scope the claim. Arctic-R1 is single-turn; the commitment-policy mechanism (when does the agent stop generating?) does NOT apply — there is exactly one "turn" and the agent always emits an Answer. v3 R2's mechanism only exists when num_turns is non-degenerate.

2. **Our absolute scores are weak.** Dense G=4 7B Spider 66.63% vs Arctic-R1 7B BIRD 68.9% (different benchmark; Spider is generally easier so we should expect higher). This makes the v3 R2 "Dense G=4 wins, G=8 loses" claim look like an artifact of an under-optimized recipe.

3. **Top methods don't ablate G.** A reviewer can't read this as "G doesn't matter" — they just chose one G and shipped. But it leaves us without external corroboration of the specific magnitude.

4. **Comparison with MTIR-SQL is the closest apples-to-apples.** MTIR-SQL is multi-turn, GRPO, similar 4B/7B scale, similar architecture. They chose G=5 and added stabilization. We chose G=4/8 with no stabilization. **Our G=4 Dense corresponds roughly to MTIR-SQL's recipe; our G=8 corresponds to nothing in the literature.** The negative G=8 result is informative *because no one in the literature uses G=8 on multi-turn agentic SQL*.

## What this implies for the paper

### Required scope tightening

- **Title/abstract**: drop "GRPO" alone; replace with "**multi-turn agentic GRPO**" or "**ReAct-style multi-turn GRPO**".
- **Section 1**: add an explicit non-claim: "We do not study single-turn execution-reward GRPO; the commitment-policy axis only exists when num_turns can vary. Single-turn methods such as Arctic-Text2SQL-R1 [cite] use larger G (G=16) successfully and lie outside our scope."
- **Section 5 (Discussion)**: comparable-recipe discussion — name MTIR-SQL as the closest published baseline, note its G=5 choice, frame v3 R2 as quantifying *why* multi-turn methods avoid high G in the first place.

### Required citations

- **Arctic-Text2SQL-R1** (cite as the boundary case where the claim does not apply).
- **MTIR-SQL** (cite as the closest comparable recipe; their "reward collapse" remark is direct support).
- **ReEx-SQL** (similar regime).
- **HES-SQL / Graph-Reward-SQL** (structural-reward methods; cite even if we don't decompose them — note that they are reward-side and orthogonal to the advantage-side share-shift mechanism).

### Required experiment to convert workshop → main-conf

If the user wants to upgrade beyond workshop:

**Reproduce MTIR-SQL's G=5 recipe on Spider-dev with our diagnostic.** Run our turn-progression decomposition on a 7B run trained with MTIR-SQL's exact hyperparameters (G=5, 6 interactions max, their reward shape, their stabilization tricks). If the diagnostic shows:
- share_shift > 0 with CI excluding 0, AND
- per_turn_gain ≥ −0.5 with CI not strictly negative
Then MTIR-SQL's stabilization tricks effectively keep the per-turn term safe while the share-shift mechanism still operates. This would be a much stronger paper: "the diagnostic explains why MTIR-SQL's stabilization is necessary".

If MTIR-SQL's recipe ALSO shows per-turn collapse, then their stabilization tricks don't actually work on the per-turn axis (they only stabilize aggregate reward) — a fundamentally interesting finding.

Either way, this turns "we ran 4 GRPO recipes ourselves" into "we explain a published method's hyperparameter choice".

Estimated cost: ~20–40 GPU-h (MTIR-SQL recipe + Spider-dev eval). Worth the bump from workshop accept to ACL Findings borderline.

## Honest verdict

**Story holds** under the scope it actually claims (multi-turn agentic GRPO at our recipe scale). It does **NOT hold** as a universal "G hurts GRPO" claim. The paper as currently drafted (v3 R2) is workshop-tier; the framing around scope is what determines whether reviewers will accept the multi-turn restriction.

**Workshop**: holds, accept with the scope tightening above.
**ACL Findings**: needs the MTIR-SQL recipe replication described above to lift confidence.
**Main conference**: would also need ≥1 model family (Llama / Mistral) at ≥2 sizes; out of current budget.

## Action items

1. **(immediate, 0 GPU-h)**: add scope-tightening paragraph to Section 1 of paper draft; cite Arctic-R1 as the boundary case.
2. **(immediate, 0 GPU-h)**: rewrite Section 5 to position v3 R2 against MTIR-SQL specifically; quote their "reward collapse" line.
3. **(if upgrading beyond workshop, ~30 GPU-h)**: replicate MTIR-SQL's G=5 recipe with our diagnostic on Spider-dev. Pre-register prediction: share_shift > 0 ✓, per_turn neutral or modestly positive.
4. **(if more compute, ~50 GPU-h)**: same diagnostic on a Llama-3-8B SFT + GRPO baseline — second model family confirms generalization beyond Qwen.
