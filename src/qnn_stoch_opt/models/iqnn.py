from typing import List

import torch
import torch.nn as nn


class IncrementalQuantileNeuralNetwork(nn.Module):  # type: ignore[misc]
    """
    Incremental Quantile Neural Network (IQNN).
    Specifically designed to avoid the quantile crossing phenomenon. The network
    outputs a base value (lowest quantile) and positive increments (using ReLU)
    for higher quantiles. The final quantiles are computed using a cumulative sum.
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

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, num_quantiles)
        self.output_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass ensuring monotonically increasing quantiles.

        Args:
            x: First-stage decisions, shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Predicted ordered quantiles,
            shape (batch_size, num_quantiles).
        """
        h = self.hidden_layers(x)
        out = self.output_layer(h)

        # The first output is the base quantile (unconstrained)
        base = out[:, 0:1]

        # The remaining outputs are increments and must be non-negative (ReLU)
        increments = self.output_activation(out[:, 1:])

        combined = torch.cat([base, increments], dim=1)

        # Cumulative sum enforces the monotonically increasing property
        return torch.cumsum(combined, dim=1)
