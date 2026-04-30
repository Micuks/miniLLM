# Problem Anchor (immutable)

**Frozen 2026-04-29.** Copied verbatim into every subsequent round.

## Bottom-line problem

Multi-turn ReAct-style RL (GRPO) on Text-to-SQL agents shows uneven and poorly-understood gains: the same recipe fails on 3B Qwen2.5 across four reward variants, succeeds modestly on 7B (Dense GRPO +2.01 EX_att / +2.32 EX_all over 7B SFT on full Spider-dev), and a structural-credit-assignment intervention (SKE-RL, skeleton-equivalence-class advantage estimation) **fails to help and degrades monotonically with G** (G=4 → -1.04 vs Dense; G=8 → -1.81 vs Dense, ≈SFT). The community implicitly attributes agentic-RL gains to "better reasoning" — but our negative SKE result suggests the active mechanism is upstream of SQL structure. **We need to determine what RL is actually changing on these agents and use that explanation to give a clear account of when/why agentic Text-to-SQL RL helps or fails.**

## Must-solve bottleneck

**Diagnostic gap, not optimization gap.** The literature reports aggregate EX_att / EX_all gains for agentic SQL RL but provides almost no decomposition of *what* changed: did the model become a better SQL reasoner, or did it merely become a more compliant tool-using agent (valid Action format, valid SQL syntax, proper Answer tag, repair-after-observation)? Without that decomposition we cannot:

1. explain why structural priors like SKE fail at 7B,
2. explain why 3B fails entirely,
3. predict where RL will help on adjacent agentic tasks,
4. interpret existing leaderboard numbers honestly.

## Non-goals

- **Not** "beat SOTA on Spider/BIRD". We will not win the leaderboard with a 24 GB single-GPU budget against 4B-multi-agent systems (MARS-SQL, MTIR-SQL).
- **Not** "propose a new better RL method that fixes everything". The single new method we touch (SKE-RL) is in fact a negative result and we keep it that way.
- **Not** a benchmark/toolkit paper as the lead claim (toolkit is supplementary artifact only).
- **Not** a 3B-vs-7B-vs-14B scaling-law claim — single 24GB card cannot defend it.

## Constraints

- **Compute**: single Quadro RTX 6000 (Turing sm_7.5, 24 GB). 1×7B GRPO run @ G=8 / 75 step ≈ 24 GPU-h ("1 run-eq"). 1×full Spider-dev eval ≈ 5 GPU-h.
- **Time**: ~3 weeks of GPU + ~1 week of writing remaining (workshop deadline driven).
- **Data**: Spider train/dev (downloaded, real DB), BIRD dev (loader works, evidence threading works, no GRPO run yet).
- **Code**: SKE-RL stack landed (canonical extractor + class-aware advantage + gates + tests, 137 pytests green); per-class diagnostic scripts (`analyze_eval_by_skeleton.py`, `tag_eval_results.py`) exist; `aggregate_eval_reports.py` with verdict gates exists; full Spider-dev eval JSONs for SFT / Dense75 / SKE-G4-75 / SKE-G8-75 already written and on disk.
- **Tooling**: deepspeed Z2, native HF generate / vLLM rollout, sqlglot 30.6 AST, real SQLite execution.
- **Venue posture**: workshop now (NeurIPS 2026 / TRL workshop / RL-for-LLMs workshop), with main-track v2 only if data is ROBUST under additional seeds + BIRD.

## Success condition

The user can show three things in one paper that audiences cannot get from the existing literature:

1. **A protocol-vs-reasoning decomposition** of the +2 pp 7B Dense GRPO gain (lenient-extraction + win/loss transition table). If the gain is mostly protocol/format, name it and own it as a contribution that *reframes how the field reads agentic-RL gains*.
2. **A mechanism-grounded explanation** of why SKE-RL — a structural advantage prior that is the natural extension of HES-SQL skeleton reward into the advantage layer — fails and *gets worse* with more rollouts. The mechanism must be one of {variance-reduction failure, class fragmentation, wrong abstraction layer, optimization side-effect} and must be backed by one targeted measurement (not just speculation).
3. A concrete **practical takeaway**: a recommended evaluation protocol for agentic Text-to-SQL RL (and more broadly, agentic RL with tool use) that makes the protocol-vs-semantic split visible by default. The audience leaves with both a falsified intuition (SKE / structural credit) and a corrected one (whatever the mechanism analysis shows).

The paper passes if a senior reviewer would say: *"This changes how I read agentic-SQL-RL leaderboards, and the negative SKE result is convincingly mechanistic, not just an engineering complaint."*
