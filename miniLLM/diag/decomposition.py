"""Turn-share / per-turn decomposition with multiple orderings.

The aggregate execution-match delta `total_ΔEX = EX(T) - EX(B)` between a
treatment T and a baseline B is decomposed into a turn-share-shift term and a
per-turn-gain term. The decomposition is exact per-resample; uncertainty is
surfaced via paired bootstrap.

Three orderings are provided; their average is the symmetric Shapley
decomposition. For binary turn binning `k ∈ {1, 2+}`, with notation
``s_B := s(B,1)``, ``s_T := s(T,1)``, ``b1 := EX(B,1)``, ``b2 := EX(B,2+)``,
``t1 := EX(T,1)``, ``t2 := EX(T,2+)``:

    Ordering A  (baseline-weighted share-shift, treatment-weighted per-turn):
        share_A = (s_T - s_B) · (b1 - b2)
        per_A   = s_T · (t1 - b1) + (1 - s_T) · (t2 - b2)

    Ordering B  (treatment-weighted share-shift, baseline-weighted per-turn):
        share_B = (s_T - s_B) · (t1 - t2)
        per_B   = s_B · (t1 - b1) + (1 - s_B) · (t2 - b2)

    Symmetric  (Shapley average of A and B):
        share_sym = (s_T - s_B) · ((b1 - b2) + (t1 - t2)) / 2
        per_sym   = ((s_B + s_T) / 2) · (t1 - b1)
                  + (1 - (s_B + s_T) / 2) · (t2 - b2)

In all three orderings, ``share + per ≡ total = EX(T) - EX(B)`` per resample.
The reviewer-recommended primary is ``Ordering="sym"``; A and B go in the
appendix as a sensitivity check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Ordering = Literal["A", "B", "sym"]


@dataclass(frozen=True)
class DecompTerms:
    """Three numbers that sum (per resample) to ``total_delta``.

    All values are stored as raw deltas in [-1, 1]; convert to percentage points
    with ``*100`` at the print/serialize boundary.
    """

    total: float
    share_shift: float
    per_turn: float


def _binary_inputs(
    turns_t: np.ndarray, em_t: np.ndarray,
    turns_b: np.ndarray, em_b: np.ndarray,
    idx: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Return ``(s_B, s_T, b1, b2, t1, t2)`` on indices ``idx``.

    Empty bins return 0.0 for the EX rate (handled by caller via masking).
    """
    st_t = (turns_t[idx] == 1).astype(np.float64)
    st_b = (turns_b[idx] == 1).astype(np.float64)
    em_t_s = em_t[idx]
    em_b_s = em_b[idx]
    s_T = float(st_t.mean())
    s_B = float(st_b.mean())
    b1 = float((em_b_s * st_b).sum() / max(st_b.sum(), 1.0))
    b2 = float((em_b_s * (1 - st_b)).sum() / max((1 - st_b).sum(), 1.0))
    t1 = float((em_t_s * st_t).sum() / max(st_t.sum(), 1.0))
    t2 = float((em_t_s * (1 - st_t)).sum() / max((1 - st_t).sum(), 1.0))
    return s_B, s_T, b1, b2, t1, t2


def decompose(
    turns_t: np.ndarray, em_t: np.ndarray,
    turns_b: np.ndarray, em_b: np.ndarray,
    idx: np.ndarray | None = None,
    ordering: Ordering = "sym",
) -> DecompTerms:
    """Return the (total, share_shift, per_turn) tuple under chosen ``ordering``.

    The three terms exactly satisfy ``share_shift + per_turn = total`` per
    sample. The ordering choice changes how the gap is *attributed*, not its
    sum. ``idx`` defaults to the full index range.
    """
    idx_arr = np.arange(turns_t.shape[0]) if idx is None else idx
    s_B, s_T, b1, b2, t1, t2 = _binary_inputs(
        turns_t, em_t, turns_b, em_b, idx_arr,
    )

    em_t_s = em_t[idx_arr]
    em_b_s = em_b[idx_arr]
    total = float(em_t_s.mean() - em_b_s.mean())

    if ordering == "A":
        share = (s_T - s_B) * (b1 - b2)
        per = s_T * (t1 - b1) + (1.0 - s_T) * (t2 - b2)
    elif ordering == "B":
        share = (s_T - s_B) * (t1 - t2)
        per = s_B * (t1 - b1) + (1.0 - s_B) * (t2 - b2)
    elif ordering == "sym":
        share = (s_T - s_B) * ((b1 - b2) + (t1 - t2)) / 2.0
        avg_s = (s_B + s_T) / 2.0
        per = avg_s * (t1 - b1) + (1.0 - avg_s) * (t2 - b2)
    else:
        raise ValueError(f"unknown ordering {ordering!r}; expected 'A', 'B', or 'sym'")

    return DecompTerms(total=total, share_shift=share, per_turn=per)


def decompose_all_orderings(
    turns_t: np.ndarray, em_t: np.ndarray,
    turns_b: np.ndarray, em_b: np.ndarray,
    idx: np.ndarray | None = None,
) -> dict[Ordering, DecompTerms]:
    """Convenience: return all three orderings on the same indices."""
    return {
        ord_: decompose(turns_t, em_t, turns_b, em_b, idx=idx, ordering=ord_)
        for ord_ in ("A", "B", "sym")
    }


def bootstrap_decomp(
    turns_t: np.ndarray, em_t: np.ndarray,
    turns_b: np.ndarray, em_b: np.ndarray,
    *,
    ordering: Ordering = "sym",
    mask: np.ndarray | None = None,
    seed: int = 20260429,
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> dict:
    """Paired bootstrap of (total, share_shift, per_turn) under chosen ``ordering``.

    Resamples record indices *jointly* across treatment and baseline so that
    paired structure (same record id, same gold) is preserved. When ``mask`` is
    provided, only indices where ``mask`` is True are eligible — used for
    difficulty-stratified or no-shift-subset bootstraps.

    Returns a dict with point estimates and CI bounds in [-1, 1] (raw, not pp).
    """
    if turns_t.shape != turns_b.shape or em_t.shape != em_b.shape:
        raise ValueError("paired decomp requires equal-length arrays")

    if mask is None:
        eligible = np.arange(turns_t.shape[0])
    else:
        eligible = np.where(mask)[0]
    n_eff = eligible.size
    if n_eff < 5:
        return {"n_records": int(n_eff), "skipped": "too_few_records"}

    point = decompose(turns_t, em_t, turns_b, em_b, idx=eligible, ordering=ordering)
    rng = np.random.default_rng(seed)
    samples = np.empty((n_resamples, 3), dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.choice(eligible, size=n_eff, replace=True)
        d = decompose(turns_t, em_t, turns_b, em_b, idx=idx, ordering=ordering)
        samples[i] = (d.total, d.share_shift, d.per_turn)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(samples, [100 * alpha, 100 * (1 - alpha)], axis=0)

    keys = ("total", "share_shift", "per_turn")
    return {
        "n_records": int(n_eff),
        "ordering": ordering,
        "ci_type": f"percentile_{int(ci*100)}",
        **{
            k: {
                "point": float(getattr(point, k)),
                "ci_lo": float(lo[i]),
                "ci_hi": float(hi[i]),
            }
            for i, k in enumerate(keys)
        },
    }


def r_same_paired_diff(
    turns_t: np.ndarray, em_t: np.ndarray,
    turns_b: np.ndarray, em_b: np.ndarray,
    *,
    seed: int = 20260429,
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> dict:
    """Paired ΔEX restricted to records where ``turn_T(i) == turn_B(i)``.

    On this no-shift subset the per-turn case mix is fixed by construction —
    any remaining sign in the per-turn term cannot be a composition artifact.
    """
    if turns_t.shape != turns_b.shape:
        raise ValueError("paired diff requires equal-length arrays")
    same_mask = turns_t == turns_b
    n_same = int(same_mask.sum())
    if n_same < 5:
        return {"n_records": int(n_same), "skipped": "too_few_records"}

    diffs = (em_t[same_mask] - em_b[same_mask]).astype(np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n_same, size=n_same)
        samples[i] = diffs[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(samples, [100 * alpha, 100 * (1 - alpha)])

    # Also break out by SFT turn count for interpretability
    by_sft_turn: dict[int, dict] = {}
    for k in sorted(set(int(x) for x in np.unique(turns_b[same_mask]))):
        sub_mask = (same_mask) & (turns_b == k)
        n_k = int(sub_mask.sum())
        if n_k == 0:
            continue
        sub_diffs = (em_t[sub_mask] - em_b[sub_mask]).astype(np.float64)
        sub_samples = np.empty(n_resamples, dtype=np.float64)
        for i in range(n_resamples):
            idx = rng.integers(0, n_k, size=n_k)
            sub_samples[i] = sub_diffs[idx].mean()
        sub_lo, sub_hi = np.percentile(sub_samples, [100 * alpha, 100 * (1 - alpha)])
        by_sft_turn[k] = {
            "n_records": n_k,
            "paired_dEX": float(sub_diffs.mean()),
            "ci_lo": float(sub_lo),
            "ci_hi": float(sub_hi),
        }

    return {
        "n_records": n_same,
        "n_total": int(turns_t.shape[0]),
        "share_no_shift": float(n_same / turns_t.shape[0]),
        "paired_dEX": float(diffs.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "by_sft_turn": by_sft_turn,
    }


__all__ = [
    "DecompTerms",
    "Ordering",
    "decompose",
    "decompose_all_orderings",
    "bootstrap_decomp",
    "r_same_paired_diff",
]
