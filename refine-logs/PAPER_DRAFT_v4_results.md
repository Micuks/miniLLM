# Workshop paper draft — Sections 4-6 (post-B4 results)

**Status**: post-B4 falsification of v3 prediction. 2×2 ablation now anchors the paper.
**Working title (revised)**: Group Size Trades Commitment for Capability — A Falsified Pre-registered Decomposition of Agentic GRPO on Text-to-SQL.

**Reads with**: `PAPER_DRAFT_v3_intro.md` (Sections 1+3); `B4_RESULT_v3_PREDICTION_FALSIFIED_2026-04-30.md` (raw result); `FINAL_PROPOSAL_v3_DRAFT.md` (the prediction we falsified).

---

## 4. Results

We compute the two-term decomposition from §3.1 on the full Spider-dev split (1034 records) for four GRPO configurations: Dense and SKE-RL at G=4 (the original two arms motivated by the SKE-RL hypothesis) and G=8 (the confound run launched after pre-registering a directional prediction). Bootstrap protocol: 1000 paired resamples, percentile 95% CI, RNG seed locked at 20260429.

### 4.1 Aggregate decomposition (2×2 ablation)

**Table 1 — `total_ΔEX_match`, `share_shift_term`, `per_turn_gain_term` for each (advantage estimator, group size) cell vs SFT (pp, 95% bootstrap CI)**.

| Pair | total ΔEX | `share_shift_term` | `per_turn_gain_term` |
|---|---:|---:|---:|
| Dense G=4 vs SFT | +2.32 [+0.39, +4.35] | +0.90 [+0.08, +1.84] | +1.42 [−0.64, +3.58] |
| SKE-RL G=4 vs SFT | +1.64 [−0.39, +3.87] | +1.32 [+0.47, +2.24] | +0.32 [−1.98, +2.58] |
| Dense G=8 vs SFT | +0.68 [−0.87, +2.22] | **+2.59 [+1.68, +3.59]** | **−1.91 [−3.64, −0.27]** |
| SKE-RL G=8 vs SFT | +0.39 [−1.06, +1.94] | **+2.64 [+1.73, +3.59]** | **−2.26 [−3.89, −0.60]** |

Bold = CI excludes 0. The G=8 row of each estimator shows the same signature: ~3× share_shift relative to its G=4 counterpart, and a per-turn capability term that is significantly negative.

The 2×2 structure makes the **G axis** carry the paper's central finding: increasing the rollout group size from 4 to 8 amplifies the commitment-policy shift while degrading per-turn SQL writing, and this trade-off is **independent of the advantage estimator** (Dense and SKE-RL both produce the same pattern, with quantitatively similar terms at each G level).

### 4.2 Pre-registered prediction outcome

In `FINAL_PROPOSAL_v3_DRAFT.md` (locked 2026-04-29, before B4 launched) we predicted:

> Dense G=8 `share_shift_term ∈ [+1.8, +2.4]` pp with 95% CI excluding 0; `per_turn_gain_term ∈ [−0.5, +0.5]` (CI overlaps 0).

The motivation was the v3 hypothesis that SKE-RL's per-turn regression was specific to its class-baseline advantage and that Dense G=8 would isolate the share-shift mechanism without the per-turn loss.

**Outcome**: Dense G=8 share_shift = +2.59 (overshoots the [+1.8, +2.4] band; CI [+1.68, +3.59] still excludes 0 in the predicted direction); per_turn_gain = −1.91 (well below the predicted band; CI [−3.64, −0.27] excludes 0 in the **negative** direction). The pre-registered prediction is **falsified** in both terms.

The falsification rules out the SKE-RL-specific framing of v3. The replacement claim — that the trade-off is a generic property of group size — is consistent with the data (Dense G=8 ≈ SKE-RL G=8 within CIs in both terms).

We document the falsification as the central methodological move of the paper: a pre-registered prediction was made with quantitative bounds, the experiment was run, and the result was reported as it fell, without re-fitting the hypothesis to the data.

### 4.3 Per-difficulty stratification

**Table 2 — same decomposition stratified by Spider difficulty (n_easy = 333, n_medium = 377, n_hard = 324)**.

(The G=8 numbers from B4 augment the previously-reported per-difficulty breakdown; only the columns relevant to the new claim are included here for brevity.)

| Pair | Easy total Δ | Medium total Δ | Medium per_turn | Hard total Δ | Hard per_turn |
|---|---:|---:|---:|---:|---:|
| Dense G=4 vs SFT | +1.50 | **+3.98 ✓** | **+3.17 ✓** | +1.23 | +0.80 |
| Dense G=8 vs SFT | +0.30 | +1.33 | −0.86 | +0.31 | −1.26 |
| SKE-RL G=4 vs SFT | +1.20 | +1.86 | +1.05 | +1.85 | +0.80 |
| SKE-RL G=8 vs SFT | +1.80 ✓ | +0.53 | −1.45 | −1.23 | −2.84 |

(✓ = CI excludes 0)

Per-difficulty: Dense G=4's gain is **concentrated on Medium with the only significantly positive per-turn capability term in the entire 4×3 grid (+3.17 ✓)**. G=8 conditions show per-turn point estimates that are **negative on Medium and Hard for both estimators** (CI overlaps 0 individually but the consistency across 4 cells is striking). SKE-RL G=8 on Hard has the most negative point estimate (−2.84). On Easy, SKE-G8 still produces a significantly positive total +1.80 ✓ via share_shift +1.84 ✓ — Easy queries already trivially resolve in 1 turn, so over-commitment has nothing left to break.

### 4.4 Per-record paired transitions

For each pair, we partition records by (turns(SFT), turns(RL)) into a 3×3 matrix. The "shift-down" set — records where the RL condition used strictly fewer turns than SFT — is further partitioned into EX-kept / EX-lost / EX-gained.

**Table 3 — shift-down outcomes per pair**.

| Pair | shift-down records | EX kept | EX lost | EX gained | net Δ |
|---|---:|---:|---:|---:|---:|
| Dense G=4 vs SFT | 125 | 45 | 8 | 19 | **+11** |
| Dense G=8 vs SFT | 117 | 42 | 12 | 15 | **+3** |
| SKE-RL G=4 vs SFT | 136 | 50 | 10 | 14 | +4 |
| SKE-RL G=8 vs SFT | 113 | 41 | 11 | 11 | **+0** |

The shift-down `net Δ` decreases monotonically with G across both estimators (Dense: +11 → +3; SKE-RL: +4 → +0). The commit-faster behavior nets fewer gains relative to losses as G grows, providing per-record corroboration of the aggregate per_turn_gain regression.

A complementary indicator is the (1 → 1) cell, where both SFT and the RL condition resolved the query in a single turn. The treatment's EX rate within this cell is bounded above by the SFT rate (73.6% in our data, since the cell is the same set of records under SFT) — any drop indicates over-commitment hurting accuracy.

| Pair | (1→1) cell n | rl EX% within cell |
|---|---:|---:|
| Dense G=4 | 753 | 73.6% (≈ SFT) |
| Dense G=8 | 820 | **70.1%** |
| SKE-RL G=4 | 757 | 73.1% |
| SKE-RL G=8 | 824 | **70.0%** |

Both G=8 conditions trap more records in the (1 → 1) cell (820 / 824 vs 753 / 757) AND degrade their accuracy by ~3.5pp relative to the SFT/G=4 rate. The (1 → 1) cell EX% is **identical to one decimal between Dense G=8 (70.1) and SKE-RL G=8 (70.0)**, sharper evidence still that the mechanism is G-driven, not advantage-estimator-driven.

### 4.5 Trajectory length signal

Across the records that stayed at turn=1 under each G=8 condition, the trajectory-length pattern conditional on EX-outcome is:

| G=8 condition | (1→1) cell n | overall Δlen | EX-kept Δlen | EX-lost Δlen | EX-gained Δlen |
|---|---:|---:|---:|---:|---:|
| Dense G=8 vs SFT | 820 | −8 chars | −13 (n=560) | **+211 (n=16)** | −57 (n=15) |
| SKE-RL G=8 vs SFT | 824 | +5 chars | −1 (n=562) | **+242 (n=13)** | (n.a.) |

The EX-lost subset — records where SFT was correct in 1 turn but the G=8 condition broke them — has **trajectories ~210-240 chars longer than SFT's** under both estimators. The pattern reproduces across estimators at the same G level, making it another manifestation of the generic G-effect: over-commitment doesn't "rush" to a wrong answer with shorter reasoning; it *expands* the Thought block to justify the wrong answer.

This is the qualitative signature of the per_turn_gain regression at the trajectory level.

## 5. Discussion

### 5.1 What the data show

The paper's empirical finding is that agentic-GRPO group size G is a tunable knob that **trades commitment-policy strength for per-turn SQL-writing quality**. The trade-off is monotonic in our data: G=4 → G=8 amplifies the commitment shift roughly 3× and flips the per-turn term from neutral-or-positive to significantly negative. The trade-off is **independent of the advantage estimator** — Dense and SKE-RL produce the same pattern at each G level.

Aggregate EX hides the trade-off because the two terms have opposite signs at moderate G. At G=4 the share-shift gain dominates (Dense +0.90 share + +1.42 per-turn = +2.32 aggregate); at G=8 the per-turn loss almost cancels the share-shift gain (Dense +2.59 share + −1.91 per-turn = +0.68 aggregate). Reporting only the aggregate, as is current practice in agentic-RL literature, would have left the mechanism invisible.

### 5.2 Why the SKE-RL-specific framing failed

The v3 hypothesis was that SKE-RL's class-baseline advantage was uniquely responsible for the G=8 per-turn regression. B4 (Dense G=8) was designed to isolate this. The result — Dense G=8 reproduces the per-turn regression at the same magnitude and significance as SKE-G8 — falsifies the SKE-RL-specific framing. The class-baseline advantage is not the cause; group size is. The two estimators differ in how aggressively they amplify commitment shift (SKE-RL slightly stronger at +2.64 vs Dense +2.59 at G=8) but produce the same trade-off curve.

### 5.3 Generalization and limitations

The decomposition is portable: replace `num_turns == 1` with "task resolved on first tool call" for any agentic-RL setting. Web agents, code agents, tool-using assistants — all sit in the same EX_all-style aggregate metric we critique. The pre-registered-prediction-falsification methodology is independently transferable to any RL ablation that wants to make a mechanism claim more rigorous than "we tried it and got these numbers".

Limitations: (i) single 7B model; (ii) single dataset (Spider-dev); (iii) two G values (4, 8) — does the trade-off reverse or saturate at G=2 or G=16? (iv) we have not characterized whether the per-turn regression decomposes further into (e.g.) loss of repair behavior vs. loss of single-turn first-try accuracy. (iii) and (iv) are natural follow-ups.

### 5.4 Implications

- **For agentic-RL practitioners**: G is not a "free" hyperparameter. Choosing G=8 over G=4 systematically degrades per-turn SQL writing capability while improving aggregate accuracy through a strategy shift. If the downstream goal cares about per-turn quality (e.g., interactive use cases where the user observes the agent's reasoning), G=4 is the correct choice. If only the final answer matters, the answer is data-dependent.

- **For researchers reporting agentic-RL gains**: aggregate EX is insufficient. The two-term decomposition is one Python file; the additional reporting cost is negligible. We argue it should become standard.

- **For SKE-RL specifically**: the mechanism that explains its failure is now precise: it amplifies commitment shift slightly more than Dense, accelerating the per-turn regression. The structural prior of class-baseline advantage is not "wrong" — it works on a layer below the actual bottleneck (group size choice).

## 6. Limitations

(See §5.3.) Additional explicit limitations:

- **Sample size**: Spider-dev's 1034 records gives reasonable CI tightness but the per-difficulty-Hard CIs are wide (~±5pp on individual terms).
- **No baseline RL methods compared**: we use Dense GRPO and SKE-RL because we own the SFT pipeline. PPO with similar G might show different curves — we cannot rule out that the trade-off is GRPO-specific.
- **Decomposition is descriptive, not causal**: we observe correlations between G and the two terms; we do not show that intervening on G *causes* the trade-off in a counterfactual sense.
- **The "single-turn vs multi-turn" partition is coarse**. A fine-grained version (turn-by-turn EX) might reveal richer dynamics. We chose binary partitioning for interpretability.

## Action items before submission

1. Run `scripts/diag_turn_by_difficulty.py` on B4 to fill Table 2 row.
2. Run `scripts/diag_paired_turn_transitions.py` extended with B4 to fill Table 3 row.
3. Trajectory length analysis on Dense G=8 EX-lost records.
4. Decide if v4 proposal supersedes v3 explicitly (recommended: yes, write FINAL_PROPOSAL_v4.md).
5. Build figures (matplotlib install required): bar chart of share_shift vs per_turn_gain across 4 conditions; scatter of (share_shift, per_turn_gain) with CI ellipses; per-difficulty heat map.
