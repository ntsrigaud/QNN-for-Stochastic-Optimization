"""
Performance metrics for evaluating the QNN framework.
"""

import numpy as np


def compute_quantile_crossing_rate(predictions: np.ndarray) -> float:
    """
    Compute the fraction of samples where quantile crossing occurs.

    Quantile crossing happens when the predicted quantiles are not
    monotonically non-decreasing across quantile levels for a given sample.

    Args:
        predictions: Array of shape ``(num_samples, num_quantiles)`` containing
            predicted quantile values ordered from the lowest to the highest
            quantile level.

    Returns:
        float: Fraction of samples (in ``[0, 1]``) that contain at least one
        crossing violation, i.e. where any adjacent pair of predicted quantiles
        is strictly decreasing.

    Example:
        >>> preds = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 4.0]])
        >>> compute_quantile_crossing_rate(preds)
        0.5
    """
    if predictions.ndim != 2:
        raise ValueError(
            "predictions must be a 2-D array of shape (num_samples, num_quantiles), "
            f"got shape {predictions.shape}."
        )
    # A crossing occurs whenever q[k] > q[k+1] for any k
    diffs = np.diff(predictions, axis=1)
    crossings = np.any(diffs < 0, axis=1)
    return float(np.mean(crossings))


def compute_optimality_gap(heuristic_obj: float, reference_obj: float) -> float:
    """
    Compute the relative optimality gap between a heuristic and a reference objective.

    The gap is expressed as a percentage:

    .. code-block::

        gap = (heuristic_obj - reference_obj) / |reference_obj| * 100

    A positive gap means the heuristic is worse (higher cost) than the reference;
    a negative gap means the heuristic outperforms the reference.

    Args:
        heuristic_obj: Objective value obtained by the heuristic (e.g. QNN surrogate).
        reference_obj: Reference objective value (e.g. exact SAA solution).

    Returns:
        float: Relative gap in percent.

    Raises:
        ValueError: If ``reference_obj`` is zero (division by zero).

    Example:
        >>> compute_optimality_gap(110.0, 100.0)
        10.0
        >>> compute_optimality_gap(95.0, 100.0)
        -5.0
    """
    if reference_obj == 0.0:
        raise ValueError(
            "reference_obj must be non-zero to compute a relative optimality gap."
        )
    return (heuristic_obj - reference_obj) / abs(reference_obj) * 100.0
