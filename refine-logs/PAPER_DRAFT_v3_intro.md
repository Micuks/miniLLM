# Workshop paper draft — Section 1 (Introduction) + Section 3 (Method)

**Working title**: Commitment Beats Capability — Decomposing Agentic GRPO Gains on Text-to-SQL into Strategy-Shift and Per-Turn-Quality Components

**Target venue**: NeurIPS 2026 workshops (TRL / RL-for-LLMs).
**Target length**: 4 pages + appendix.

---

## 1. Introduction

Multi-turn ReAct-style reinforcement learning has become the default training
recipe for agentic Text-to-SQL [1, 2, 3]. Recent results report aggregate
execution-match (EX) gains in the +1 to +5 pp range over supervised
fine-tuning (SFT), and the gains are typically attributed to "the model
learns to write better SQL" or "the model learns to recover from execution
errors". These attributions are read directly off the aggregate EX gain.

We argue that this attribution is under-determined. The aggregate EX gain
admits at least two operationally distinct mechanisms:

1. **Per-turn capability**: at fixed turn count, the model's SQL is more
   often correct.
2. **Commitment-policy shift**: the model resolves more queries in a
   single turn (where SFT's accuracy is already higher), without
   improving accuracy at any individual turn count.

Mechanism (2) is invisible to aggregate EX yet has different downstream
implications. A model whose RL gains are dominantly mechanism (2) has not
become more capable at SQL writing; it has become more decisive about
when to stop exploring. The two mechanisms also have different
predictions for negative results: a structural prior like SKE-RL [4]
that pushes the model toward "skeleton-complete" trajectories should
amplify mechanism (2) regardless of whether mechanism (1) follows.

We introduce a deterministic, leakage-safe two-term decomposition:

```
total_ΔEX = share_shift_term + per_turn_gain_term
```

where `share_shift_term` is the gain attributable to redistributing
queries across turn counts (with EX rates frozen at the SFT baseline),
and `per_turn_gain_term` is the gain attributable to changing the EX
rate at each fixed turn count (with the share weighted by the treatment
distribution). Both terms are reported with paired-record bootstrap 95%
CIs. The decomposition is portable: replace `num_turns` with any
agentic-RL turn-count proxy and the math is unchanged.

Applied to a 7B Qwen-Instruct ReAct agent on Spider-dev (1034 examples),
with one SFT baseline and three GRPO-trained checkpoints (Dense G=4,
SKE-RL G=4, SKE-RL G=8), we find:

- Dense G=4's +2.32 pp gain over SFT splits as `share_shift = +0.90 pp`
  (CI [+0.08, +1.84], excludes 0) and `per_turn_gain = +1.42 pp` (CI
  [−0.64, +3.58]). The commitment-shift component is statistically
  significant; per-turn capability has positive point estimate but a CI
  that overlaps 0 in aggregate.
- The per-turn capability gain is concentrated on **medium-difficulty
  queries** (+3.17 pp, CI [+0.08, +6.24], excludes 0), where SFT
  accuracy is in the meaningful middle band where genuine SQL writing
  improvement matters.
- SKE-RL at G=8 produces the largest commitment shift (+2.64 pp, CI
  [+1.73, +3.59]) but a **statistically significant per-turn capability
  regression** (−2.26 pp, CI [−3.89, −0.60], excludes 0 in negative
  direction). The two terms nearly cancel; aggregate EX is +0.39 pp,
  not significant. Per-record paired analysis confirms: among the 113
  records that shifted to fewer turns under SKE-G8, the "EX gained" and
  "EX lost" sets are exactly equal (11 vs 11), making the shift
  net-neutral.

A single mechanism — commitment-policy shift — therefore explains both
the modest Dense gain AND the SKE-RL failure across G=4 and G=8. The
decomposition surfaces this shared mechanism from the same aggregate-EX
data that earlier analyses left interpretively flat.

Our contributions:

(i) A deterministic, portable, leakage-safe two-term decomposition of
agentic-RL aggregate metrics with paired-record bootstrap 95% CIs.

(ii) Empirical demonstration on 4 7B Spider-dev runs that a commitment-
policy shift accounts for a measured fraction of every gain examined:
0.39 of Dense's, 0.80 of SKE-G4's, and 100% of SKE-G8's (modulo a
negative per-turn term).

(iii) A mechanism-grounded explanation of SKE-RL's failure that is
predictively distinct from "SKE just doesn't work": SKE amplifies the
commitment shift more than Dense, but provides no signal that improves
per-turn quality, and at G=8 the over-commitment hurts per-turn
accuracy enough to wash out the gain.

(iv) Pre-registered prediction for a Dense G=8 confound run that, if
confirmed, isolates the SKE-RL-specific quality regression from the
generic effect of larger group size.

We do **not** claim a new RL training method, a SOTA Spider/BIRD
result, or a scaling law. The contribution is interpretive: the
decomposition turns existing aggregate-EX numbers into mechanistically
distinct claims.

## 3. Method

### 3.1 Decomposition

Let `EX(X, k)` denote the EX-match rate of condition X among records
where condition X used `k` turns, and let `s(X, k)` denote the share of
X's records using `k` turns. Aggregate EX is

```
EX(X) = Σ_k s(X, k) · EX(X, k).
```

For two conditions B (baseline) and T (treatment), the aggregate
difference is

```
ΔEX = EX(T) − EX(B)
    = Σ_k [s(T, k) · EX(T, k) − s(B, k) · EX(B, k)].
```

Adding and subtracting `s(T, k) · EX(B, k)` inside the sum gives the
exact decomposition

```
ΔEX = Σ_k [s(T, k) − s(B, k)] · EX(B, k)        ← share_shift_baseline
     + Σ_k s(T, k) · [EX(T, k) − EX(B, k)].     ← per_turn_gain
```

For binary turn binning (`k ∈ {1, 2+}`), the share_shift_term reduces
to

```
share_shift_term = (s(T, 1) − s(B, 1)) · (EX(B, 1) − EX(B, 2+)),
```

a closed form interpretable as "the fraction of queries shifted to
1-turn, weighted by SFT's 1-turn accuracy advantage over multi-turn".

Both terms are computed at the per-resample level. We use paired
bootstrap (1000 resamples, percentile 95% CI, fixed seed 20260429): the
same record indices are resampled for treatment and baseline so the
covariance structure is preserved. Decomposition exactness
(`total = share_shift + per_turn_gain`) holds for each individual
resample; the bootstrap surfaces uncertainty about each term.

### 3.2 Per-difficulty stratification

The Spider dev split carries a difficulty annotation per record (easy /
medium / hard / extra). We compute the same decomposition stratified by
difficulty. Stratum sizes for Spider-dev: 333 easy, 377 medium, 324
hard, 0 extra (Spider's "extra" is concentrated in the train split).

### 3.3 Per-record paired transitions

For mechanism interpretation, we cross-tabulate `(num_turns(SFT) → num_turns(RL))`
into a 3×3 transition matrix (turn=1, 2, 3+) per pair, recording per
cell: (count, EX rate of treatment within cell). The "shift-down" set
— records where treatment used strictly fewer turns than baseline — is
further partitioned into EX-kept / EX-lost / EX-gained subsets,
yielding the per-record net contribution of the commitment shift.

### 3.4 Why a leakage-safe lenient extractor was rejected

We initially specified a leakage-safe lenient SQL extractor
(first-parseable-in-Answer-region with last-action-fallback) and a
4-bucket classifier (protocol_gain, semantic_gain, no_change,
lenient_only_repair). Empirically, the strict extractor used in the
existing eval pipeline produces the same SQL as the lenient extractor
on 99.9% of records (Table 2; details in Appendix B), so the
lenient/strict axis is operationally null on this dataset. The
turn-progression axis above is the alternative we adopted; the lenient
extractor + bucket classifier remains in the appendix as the diagnostic
we attempted, with full implementation released alongside the paper for
groups working on data with stronger protocol noise (smaller models,
no SFT, or domains with less structured action templates).

---

## Notes for the writing pass

- Section 2 (Setup): half page describing ReAct + GRPO + Spider-dev
  baseline + 4 checkpoint configurations. Cite existing GRPO + Spider
  refs.

- Section 4 (Results, the core): Table 1 = aggregate decomposition (4
  rows: SFT, Dense, SKE-G4, SKE-G8 vs SFT). Table 2 = per-difficulty
  decomposition. Figure 1 = 3-bar grouped chart (share_shift vs
  per_turn_gain) per condition with error bars. Figure 2 = transition
  matrix heatmap.

- Section 5 (Discussion): mechanism summary; pre-registered B4
  prediction; what the decomposition implies for other agentic-RL
  domains.

- Section 6 (Limitations): single 7B model size; single dataset
  (Spider-dev); deterministic SQL evaluator may overweight schema-level
  errors; commitment shift might be a Spider-specific artifact (typical
  Spider question requires 1 SELECT) — discuss BIRD comparison as
  future work.

- Appendix A: full bucket classifier + lenient extractor implementation
  (they're correct and useful, just empirically null on this data).

- Appendix B: extractor-equivalence table per condition.

- Appendix C: full per-difficulty / per-record-transition tables.

## Citations to chase down

- HES-SQL (skeleton reward) — referenced as [4] above
- MTIR-SQL, MARS-SQL — comparison baselines (aggregate EX only)
- Original GRPO papers (DeepSeek, etc.)
- Spider benchmark paper
- SQLite / sqlglot — for tooling

## Outstanding questions for the user

- Pre-registration of the Dense G=8 prediction: the v3 draft has
  `share_shift ∈ [+1.8, +2.4]` and `per_turn_gain ∈ [−0.5, +0.5]`. Want
  to commit those bounds tighter / looser?
- Workshop venue choice: TRL (NeurIPS '26 workshop) is the most natural
  fit; deadline TBD.
- Single-author or co-author with the (de-identified for review)
  collaborator who set up the SFT pipeline?
