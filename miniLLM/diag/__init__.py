"""Diagnostic protocol for agentic Text-to-SQL RL.

Implements the protocol-vs-semantic gain decomposition from
``refine-logs/FINAL_PROPOSAL.md``. All public functions are deterministic and
leakage-safe (no gold-conditioned candidate selection).
"""
from __future__ import annotations

from .lenient_extract import lenient_extract
from .classifier import classify_pair, per_side_flags
from .bootstrap import bootstrap_ci, bootstrap_proportion_ci, bootstrap_cliffs_delta_ci
from .equivalence import cliffs_delta, cliffs_delta_equivalence
from .decomposition import (
    DecompTerms,
    decompose,
    decompose_all_orderings,
    bootstrap_decomp,
    r_same_paired_diff,
)

__all__ = [
    "lenient_extract",
    "classify_pair",
    "per_side_flags",
    "bootstrap_ci",
    "bootstrap_proportion_ci",
    "bootstrap_cliffs_delta_ci",
    "cliffs_delta",
    "cliffs_delta_equivalence",
    "DecompTerms",
    "decompose",
    "decompose_all_orderings",
    "bootstrap_decomp",
    "r_same_paired_diff",
]
