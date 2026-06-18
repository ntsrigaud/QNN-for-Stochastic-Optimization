import typing
from typing import List

import torch
import torch.nn as nn


class QuantileNeuralNetwork(nn.Module):  # type: ignore[misc]
    """
    Standard Quantile Neural Network (QNN).
    Uses a standard feedforward architecture with ReLU activations in the hidden layers,
    and a linear output layer to simultaneously predict multiple quantiles.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], num_quantiles: int):
        """
        Args:
            input_dim: Dimensionality of the first-stage decision variables (X).
            hidden_dims: List specifying the number of neurons in each hidden layer.
            num_quantiles: Number of quantiles to predict.
        """
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Linear output layer for the quantiles
        layers.append(nn.Linear(prev_dim, num_quantiles))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting the quantiles.

        Args:
            x: First-stage decisions, shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Predicted quantiles, shape (batch_size, num_quantiles).
        """
        return typing.cast(torch.Tensor, self.network(x))
