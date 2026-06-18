from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from qnn_stoch_opt.models.loss import pinball_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    quantiles: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
) -> Tuple[nn.Module, float]:
    """
    Standard training loop with early stopping for QNN/IQNN models.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = pinball_loss(preds, batch_y, quantiles)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = pinball_loss(preds, batch_y, quantiles)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_loss
