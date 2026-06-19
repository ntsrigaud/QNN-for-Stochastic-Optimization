"""
Endogenous uncertainty module.

Models P(demand_2 | x_1) as a Gaussian whose mean and variance are linear
functions of the stage-1 decision x_1 (specifically, of sum(x_1)) -- a
lightweight parametric approximation of the true endogenous coupling encoded
in :func:`ThreeStageCFLPEvaluator.sample_demand_2
<qnn_stoch_opt.case_studies.cflp_multistage.ThreeStageCFLPEvaluator>`.
"""

from __future__ import annotations

import numpy as np


class ConditionalDemandModel:
    """Gaussian model of P(demand_2 | x_1) fitted by least squares.

    For each customer j, the mean demand is modeled as
    ``a_j + b_j * sum(x_1)`` and the variance is a single constant shared
    across all customers, fitted to the residuals of the mean fit.
    """

    def __init__(self) -> None:
        self.a: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.var: float | None = None

    def fit(self, X1_train: np.ndarray, demand2_train: np.ndarray) -> None:
        """Fit the conditional mean and variance via least squares.

        Args:
            X1_train: Array of shape (N, n) with stage-1 decisions x_1.
            demand2_train: Array of shape (N, m) with observed stage-2 demand.
        """
        s = X1_train.sum(axis=1)
        design = np.stack([np.ones_like(s), s], axis=1)  # (N, 2)
        coeffs, _, _, _ = np.linalg.lstsq(design, demand2_train, rcond=None)
        self.a = coeffs[0]
        self.b = coeffs[1]

        residuals = demand2_train - design @ coeffs
        self.var = float(np.mean(residuals**2))

    def sample(self, x_1: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
        """Draw samples from the fitted N(a + b*sum(x_1), var) per customer.

        Args:
            x_1: Stage-1 decision vector of shape (n,).
            n_samples: Number of demand scenarios to draw.
            seed: Random seed for reproducibility.

        Returns:
            np.ndarray: Sampled demand scenarios of shape (n_samples, m).
        """
        if self.a is None or self.b is None or self.var is None:
            raise RuntimeError("ConditionalDemandModel must be fit() before sample().")

        s = float(np.sum(x_1))
        mean = self.a + self.b * s
        rng = np.random.default_rng(seed)
        shape = (n_samples, len(mean))
        return rng.normal(loc=mean, scale=np.sqrt(self.var), size=shape)

    def save(self, path: str) -> None:
        """Persist the fitted parameters to a ``.npz`` file."""
        if self.a is None or self.b is None or self.var is None:
            raise RuntimeError("ConditionalDemandModel must be fit() before save().")
        np.savez(path, a=self.a, b=self.b, var=np.array(self.var))

    def load(self, path: str) -> None:
        """Load fitted parameters from a ``.npz`` file written by :meth:`save`."""
        data = np.load(path)
        self.a = data["a"]
        self.b = data["b"]
        self.var = float(data["var"])
