import numpy as np
import torch
from qnn_stoch_opt.data.dataset import StochasticOptimizationDataset, create_dataloaders


def test_stochastic_optimization_dataset() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    # We scramble Y_dist to ensure the Dataset sorts it correctly for the CDF
    Y_dist = np.array([[5.0, 2.0, 9.0], [4.0, 1.0, 1.0]])

    dataset = StochasticOptimizationDataset(X, Y_dist)

    assert len(dataset) == 2

    x0, y0 = dataset[0]
    assert isinstance(x0, torch.Tensor)
    assert isinstance(y0, torch.Tensor)

    # Check that Y is sorted to represent the empirical CDF
    np.testing.assert_allclose(y0.numpy(), np.array([2.0, 5.0, 9.0]))

    x1, y1 = dataset[1]
    np.testing.assert_allclose(y1.numpy(), np.array([1.0, 1.0, 4.0]))


def test_create_dataloaders() -> None:
    X = np.random.rand(100, 3)
    Y_dist = np.random.rand(100, 10)

    train_loader, test_loader = create_dataloaders(
        X, Y_dist, batch_size=16, train_ratio=0.8, seed=42
    )

    # 80 training samples => 80/16 = 5 batches
    # 20 test samples => 20/16 = 2 batches (1 batch of 16, 1 of 4)
    assert len(train_loader) == 5
    assert len(test_loader) == 2

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (16, 3)
    assert batch_y.shape == (16, 10)
