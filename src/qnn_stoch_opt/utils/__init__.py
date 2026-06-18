"""
Utility helpers for the QNN framework.

Exports:
    compute_quantile_crossing_rate: Fraction of samples that exhibit quantile
        crossing violations in a set of QNN predictions.
    compute_optimality_gap: Relative optimality gap between a heuristic objective
        and a reference (e.g. SAA) objective value.
"""

from qnn_stoch_opt.utils.metrics import (
    compute_optimality_gap,
    compute_quantile_crossing_rate,
)

__all__ = [
    "compute_quantile_crossing_rate",
    "compute_optimality_gap",
]
