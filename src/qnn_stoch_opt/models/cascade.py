"""
Cascade of QNN/IQNN surrogates for multi-stage stochastic optimization.

Generalizes the two-stage QNN framework (Alcantara, Ruiz & Tsay, 2024) to
T stages by chaining T-1 surrogates, one per stage transition, trained by
backward induction: the terminal-stage surrogate is trained first (on
directly observed terminal costs), then each earlier surrogate is trained on
targets that already incorporate the optimal continuation value implied by
the next, already-trained surrogate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from qnn_stoch_opt.data.dataset import create_dataloaders
from qnn_stoch_opt.models.iqnn import IncrementalQuantileNeuralNetwork
from qnn_stoch_opt.models.qnn import QuantileNeuralNetwork
from qnn_stoch_opt.models.trainer import train_model

if TYPE_CHECKING:
    from qnn_stoch_opt.optimization.multistage_optimizer import (
        MultiStageSurrogateOptimizer,
    )


class CascadeQNN:
    """A cascade of T-1 QNN/IQNN surrogates for T-stage stochastic optimization.

    Stage t's surrogate maps stage-t decisions X_t to conditional quantiles
    of the stage-(t+1) value function, which already incorporates the
    optimal recourse available from stage t+1 onward. Training proceeds
    backward from the terminal stage; at inference time, all T-1 surrogates
    are embedded as a single MILP via :meth:`embed_all_stages`.
    """

    def __init__(
        self,
        n_stages: int,
        input_dims: List[int],
        hidden_dims: List[int],
        num_quantiles: int,
        model_type: str = "iqnn",
    ):
        if len(input_dims) != n_stages - 1:
            raise ValueError(
                f"input_dims must have length n_stages-1={n_stages - 1}, "
                f"got {len(input_dims)}."
            )
        if model_type not in ("qnn", "iqnn"):
            raise ValueError(
                f"Unknown model_type: {model_type!r}. Must be 'qnn' or 'iqnn'."
            )

        self.n_stages = n_stages
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_quantiles = num_quantiles
        self.model_type = model_type

        model_cls = (
            QuantileNeuralNetwork
            if model_type == "qnn"
            else IncrementalQuantileNeuralNetwork
        )
        self.surrogates: List[nn.Module] = [
            model_cls(
                input_dim=dim, hidden_dims=hidden_dims, num_quantiles=num_quantiles
            )
            for dim in input_dims
        ]

    def train_backward(
        self,
        stage_data: List[Tuple[np.ndarray, np.ndarray]],
        quantiles: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 10,
    ) -> List[float]:
        """Train surrogates from the terminal stage backward.

        Args:
            stage_data: ``stage_data[t] = (X_t, v_t)`` training pairs for the
                stage-t surrogate, ``X_t`` of shape (N, input_dims[t]) and
                ``v_t`` of shape (N,).
            quantiles: Quantile levels passed through to the pinball loss.
            epochs, lr, patience: Forwarded to :func:`train_model`.

        Returns:
            List[float]: Best validation pinball loss per surrogate, in the
            same order as ``stage_data`` (index t -> stage-t surrogate).
        """
        if len(stage_data) != len(self.surrogates):
            raise ValueError(
                f"stage_data must have {len(self.surrogates)} entries, "
                f"got {len(stage_data)}."
            )

        val_losses = [0.0] * len(self.surrogates)
        for t in reversed(range(len(self.surrogates))):
            X_t, v_t = stage_data[t]
            Y_t = v_t.reshape(-1, 1).astype(np.float32)
            train_loader, val_loader = create_dataloaders(
                X_t.astype(np.float32), Y_t, batch_size=32
            )
            trained_model, val_loss = train_model(
                self.surrogates[t],
                train_loader,
                val_loader,
                quantiles,
                epochs=epochs,
                lr=lr,
                patience=patience,
            )
            self.surrogates[t] = trained_model
            val_losses[t] = val_loss

        return val_losses

    def embed_all_stages(self, optimizer: "MultiStageSurrogateOptimizer") -> None:
        """Embed all trained surrogates into the multi-stage MILP optimizer."""
        for t, surrogate in enumerate(self.surrogates):
            optimizer.embed_stage_surrogate(t, surrogate, self.model_type)
