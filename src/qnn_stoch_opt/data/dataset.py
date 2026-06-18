from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class StochasticOptimizationDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):  # type: ignore[misc]
    """
    PyTorch Dataset pairing first-stage decisions X with their corresponding
    distribution of second-stage objectives (costs evaluated across scenarios).
    """

    def __init__(self, X: np.ndarray, Y_distributions: np.ndarray):
        """
        Args:
            X: (num_samples, num_x) array of first-stage decisions.
            Y_distributions: (num_samples, num_scenarios) empirical distribution of
                                second-stage costs.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        # We sort the scenarios along the last dimension so that the neural network
        # naturally learns the quantiles from the empirical CDF.
        sorted_distributions = np.sort(Y_distributions, axis=1)
        self.Y_distributions = torch.tensor(sorted_distributions, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y_distributions[idx]


def create_dataloaders(
    X: np.ndarray,
    Y_dist: np.ndarray,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[
    DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    DataLoader[Tuple[torch.Tensor, torch.Tensor]],
]:
    """
    Split the dataset into training and testing sets and construct PyTorch DataLoaders.

    Args:
        X: First-stage decision variables.
        Y_dist: Simulated second-stage cost distributions.
        batch_size: Batch size for the DataLoader.
        train_ratio: Proportion of data to use for training.
        seed: Random seed for shuffling.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and Testing dataloaders.
    """
    np.random.seed(seed)
    dataset_size = len(X)
    indices = np.random.permutation(dataset_size)

    train_size = int(dataset_size * train_ratio)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = StochasticOptimizationDataset(
        X[train_indices], Y_dist[train_indices]
    )
    test_dataset = StochasticOptimizationDataset(X[test_indices], Y_dist[test_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
