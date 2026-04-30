"""Robustness pack for the v3 turn-share / per-turn-gain decomposition.

Implements the round-2 reviewer-required additions:

1. **Decomposition robustness**: report Ordering A, Ordering B, and the
   symmetric Shapley average of the (share_shift, per_turn) pair, with paired
   bootstrap CIs for each.

2. **R_same paired ΔEX**: paired execution-match difference restricted to the
   no-shift subset ``R_same = {i : turn_T(i) == turn_B(i)}``, where the per-turn
   case mix is held fixed by construction. Eliminates composition contamination
   in the per_turn term.

3. **Difficulty × turn 3×2 stratification**: per-difficulty Shapley-symmetric
   decomposition.

4. **Augmented 3×3 transition matrix**: cells gain ``paired_dEX`` (the EX delta
   on the same records under the cell's transition pattern) and
   ``contribution = (n / N_total) × paired_dEX``.

Inputs (locked SHA256, see ``refine-logs/PRE_REGISTRATION.md`` §1):
    /mnt/data2/wuql/miniLLM/outputs/eval_full_dev_sft.json
    /mnt/data2/wuql/miniLLM/outputs/eval_full_dev_dense75.json
    /mnt/data2/wuql/miniLLM/outputs/eval_full_dev_ske75.json
    /mnt/data2/wuql/miniLLM/outputs/eval_full_dev_ske_g8_ck75.json

Output: ``outputs/robustness_pack.json``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniLLM.diag.decomposition import (  # noqa: E402
    bootstrap_decomp,
    r_same_paired_diff,
)

REPO_ROOT = Path("/mnt/data2/wuql/miniLLM")

INPUTS = {
    "SFT": (
        REPO_ROOT / "outputs/eval_full_dev_sft.json",
        "291f54b88a937be2f0994f7d7dcf7f40c6a315e61de0563d882ef6c81bac5fa3",
    ),
    "Dense75": (
        REPO_ROOT / "outputs/eval_full_dev_dense75.json",
        "e0b733ab181e980e0344e7b4d21723030280cae820b9930d5ca8438e06255888",
    ),
    "SKE-G4-75": (
        REPO_ROOT / "outputs/eval_full_dev_ske75.json",
        "7d6cbd3e5f438731b4e1b1448d93845d2baf93683ea66d3aeae2c4328f83bc66",
    ),
    "SKE-G8-75": (
        REPO_ROOT / "outputs/eval_full_dev_ske_g8_ck75.json",
        "390eb1bf86c9a934fa4e839854cc0a4e40a7f950c4c5ee0964351cba0567d85b",
    ),
}

RL_LABELS = ("Dense75", "SKE-G4-75", "SKE-G8-75")
DIFFICULTIES = ("easy", "medium", "hard", "extra")
SEED = 20260429
N_RESAMPLES = 1000


def verify_hash(path: Path, expected_sha256: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {path.name}: expected {expected_sha256[:12]}…, "
            f"got {actual[:12]}…"
        )


def load(path: Path, sha: str) -> list[dict]:
    verify_hash(path, sha)
    with open(path) as f:
        return json.load(f)["records"]


def per_record_arrays(recs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_recs = sorted(recs, key=lambda r: r.get("index", -1))
    turns = np.array([int(r.get("num_turns", 0)) for r in sorted_recs], dtype=np.int64)
    em = np.array(
        [bool(r.get("execution_match", False)) for r in sorted_recs], dtype=np.float64
    )
    diff = np.array([str(r.get("difficulty", "")) for r in sorted_recs])
    return turns, em, diff


def to_pp(d: dict) -> dict:
    """Convert raw [-1,1] decomposition CI dict to percentage points in-place keys."""
    out = {"n_records": d["n_records"], "ordering": d.get("ordering")}
    for k in ("total", "share_shift", "per_turn"):
        v = d[k]
        out[k + "_pp"] = {
            "point": v["point"] * 100.0,
            "ci_lo": v["ci_lo"] * 100.0,
            "ci_hi": v["ci_hi"] * 100.0,
        }
    return out


def turn_bin(t: int) -> str:
    if t == 1:
        return "1"
    if t == 2:
        return "2"
    return "3+"


def augmented_transition_matrix(
    turns_b: np.ndarray, em_b: np.ndarray,
    turns_t: np.ndarray, em_t: np.ndarray,
) -> dict:
    """3×3 matrix with paired_dEX and contribution per cell.

    Each cell records:
      n             = #records in cell
      em_b_count    = how many of those had EX_B = 1
      em_t_count    = how many had EX_T = 1
      paired_dEX    = (em_t_count − em_b_count) / n   (delta on same records)
      contribution  = (n / N_total) * paired_dEX
    """
    bins = ("1", "2", "3+")
    n_total = int(turns_b.shape[0])
    cells: dict[tuple[str, str], dict] = {
        (a, b): {"n": 0, "em_b_count": 0, "em_t_count": 0}
        for a in bins for b in bins
    }
    for i in range(n_total):
        a = turn_bin(int(turns_b[i]))
        b = turn_bin(int(turns_t[i]))
        c = cells[(a, b)]
        c["n"] += 1
        c["em_b_count"] += int(em_b[i])
        c["em_t_count"] += int(em_t[i])

    rows = []
    for a in bins:
        for b in bins:
            c = cells[(a, b)]
            n = c["n"]
            if n == 0:
                paired_dEX = 0.0
                contribution = 0.0
                em_b_rate = 0.0
                em_t_rate = 0.0
            else:
                paired_dEX = (c["em_t_count"] - c["em_b_count"]) / n
                contribution = (n / n_total) * paired_dEX
                em_b_rate = c["em_b_count"] / n
                em_t_rate = c["em_t_count"] / n
            rows.append({
                "sft_turn": a,
                "rl_turn": b,
                "n": n,
                "share": n / n_total,
                "em_b_rate": em_b_rate,
                "em_t_rate": em_t_rate,
                "paired_dEX": paired_dEX,
                "contribution": contribution,
            })
    contribution_total = sum(r["contribution"] for r in rows)
    return {
        "n_total": n_total,
        "rows": rows,
        "sum_contribution": contribution_total,
        "note": (
            "sum_contribution should equal total_ΔEX (mean(em_t) - mean(em_b)); "
            "rounding < 1e-12 expected."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "outputs/robustness_pack.json"),
    )
    args = parser.parse_args()

    print("[robustness_pack] verifying input hashes …")
    cond: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, (path, sha) in INPUTS.items():
        recs = load(path, sha)
        cond[label] = per_record_arrays(recs)

    sft_turns, sft_em, sft_diff = cond["SFT"]
    n_total = int(sft_turns.shape[0])

    out: dict = {
        "schema_version": 1,
        "seed": SEED,
        "n_resamples": N_RESAMPLES,
        "ci_type": "percentile_95",
        "n_total": n_total,
        "decomposition_robustness": {},
        "r_same_paired": {},
        "difficulty_x_turn_decomp": {},
        "augmented_transition_matrix": {},
    }

    # ---- (1) Decomposition robustness: A / B / sym, full sample ----
    print("\n=== Decomposition robustness (orderings A / B / sym) ===")
    for label in RL_LABELS:
        t_turns, t_em, _ = cond[label]
        sub = {}
        for ord_ in ("A", "B", "sym"):
            d = bootstrap_decomp(
                t_turns, t_em, sft_turns, sft_em,
                ordering=ord_, seed=SEED, n_resamples=N_RESAMPLES,
            )
            sub[ord_] = to_pp(d)
        out["decomposition_robustness"][f"{label}_vs_SFT"] = sub
        print(f"\n  {label} vs SFT (pp):")
        for ord_ in ("A", "B", "sym"):
            v = sub[ord_]
            t = v["total_pp"]
            s = v["share_shift_pp"]
            p = v["per_turn_pp"]
            print(
                f"    {ord_:3s} | total={t['point']:+.2f} [{t['ci_lo']:+.2f},{t['ci_hi']:+.2f}]"
                f" | share={s['point']:+.2f} [{s['ci_lo']:+.2f},{s['ci_hi']:+.2f}]"
                f" | per_turn={p['point']:+.2f} [{p['ci_lo']:+.2f},{p['ci_hi']:+.2f}]"
            )

    # ---- (2) R_same paired ΔEX (composition decontamination) ----
    print("\n\n=== R_same paired ΔEX (no-shift subset) ===")
    for label in RL_LABELS:
        t_turns, t_em, _ = cond[label]
        rs = r_same_paired_diff(
            t_turns, t_em, sft_turns, sft_em,
            seed=SEED, n_resamples=N_RESAMPLES,
        )
        rs_pp = {
            "n_records": rs.get("n_records", 0),
            "n_total": rs.get("n_total", n_total),
            "share_no_shift": rs.get("share_no_shift", 0.0),
            "paired_dEX_pp": {
                "point": rs.get("paired_dEX", float("nan")) * 100.0,
                "ci_lo": rs.get("ci_lo", float("nan")) * 100.0,
                "ci_hi": rs.get("ci_hi", float("nan")) * 100.0,
            },
            "by_sft_turn": {
                str(k): {
                    "n_records": v["n_records"],
                    "paired_dEX_pp": v["paired_dEX"] * 100.0,
                    "ci_lo_pp": v["ci_lo"] * 100.0,
                    "ci_hi_pp": v["ci_hi"] * 100.0,
                }
                for k, v in rs.get("by_sft_turn", {}).items()
            },
        }
        out["r_same_paired"][f"{label}_vs_SFT"] = rs_pp
        d = rs_pp["paired_dEX_pp"]
        print(
            f"\n  {label} vs SFT | n_same={rs_pp['n_records']} ({100*rs_pp['share_no_shift']:.1f}% of {n_total})"
            f" | paired ΔEX = {d['point']:+.2f} [{d['ci_lo']:+.2f},{d['ci_hi']:+.2f}]"
        )
        for k, v in sorted(rs_pp["by_sft_turn"].items(), key=lambda kv: int(kv[0])):
            print(
                f"    sft_turn={k:>2s} n={v['n_records']:>4d}"
                f" paired ΔEX = {v['paired_dEX_pp']:+.2f}"
                f" [{v['ci_lo_pp']:+.2f},{v['ci_hi_pp']:+.2f}]"
            )

    # ---- (3) Difficulty × turn 3×2 stratification ----
    print("\n\n=== Difficulty × turn-bin Shapley-sym decomposition ===")
    for diff_label in DIFFICULTIES:
        mask = sft_diff == diff_label
        n_diff = int(mask.sum())
        if n_diff == 0:
            continue
        out["difficulty_x_turn_decomp"][diff_label] = {
            "n_records": n_diff,
            "by_pair": {},
        }
        print(f"\n  -- {diff_label.upper()} (n={n_diff}) --")
        for label in RL_LABELS:
            t_turns, t_em, _ = cond[label]
            d = bootstrap_decomp(
                t_turns, t_em, sft_turns, sft_em,
                ordering="sym", mask=mask, seed=SEED, n_resamples=N_RESAMPLES,
            )
            if "skipped" in d:
                out["difficulty_x_turn_decomp"][diff_label]["by_pair"][f"{label}_vs_SFT"] = d
                print(f"    {label}: SKIPPED ({d['skipped']})")
                continue
            v = to_pp(d)
            out["difficulty_x_turn_decomp"][diff_label]["by_pair"][f"{label}_vs_SFT"] = v
            t = v["total_pp"]
            s = v["share_shift_pp"]
            p = v["per_turn_pp"]
            print(
                f"    {label:11s} | total={t['point']:+.2f} [{t['ci_lo']:+.2f},{t['ci_hi']:+.2f}]"
                f" | share_sym={s['point']:+.2f} [{s['ci_lo']:+.2f},{s['ci_hi']:+.2f}]"
                f" | per_turn_sym={p['point']:+.2f} [{p['ci_lo']:+.2f},{p['ci_hi']:+.2f}]"
            )

    # ---- (4) Augmented 3×3 transition matrix with paired ΔEX ----
    print("\n\n=== Augmented 3×3 transition matrix ===")
    for label in RL_LABELS:
        t_turns, t_em, _ = cond[label]
        atm = augmented_transition_matrix(sft_turns, sft_em, t_turns, t_em)
        out["augmented_transition_matrix"][f"{label}_vs_SFT"] = atm
        print(f"\n  {label} vs SFT (sum_contribution={atm['sum_contribution']*100:+.3f} pp)")
        print(f"    {'cell':>10s} {'n':>5s} {'share':>7s} {'EX_SFT':>7s} {'EX_RL':>7s} {'ΔEX':>7s} {'contrib':>9s}")
        for r in atm["rows"]:
            cell_lbl = f"{r['sft_turn']}->{r['rl_turn']}"
            print(
                f"    {cell_lbl:>10s} {r['n']:>5d} {r['share']*100:>6.1f}%"
                f" {r['em_b_rate']*100:>6.1f}% {r['em_t_rate']*100:>6.1f}%"
                f" {r['paired_dEX']*100:>+6.2f}"
                f" {r['contribution']*100:>+8.3f}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[robustness_pack] wrote {out_path}")


if __name__ == "__main__":
    main()
