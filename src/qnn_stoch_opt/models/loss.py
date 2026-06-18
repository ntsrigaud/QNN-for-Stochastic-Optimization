import torch


def pinball_loss(
    preds: torch.Tensor, targets: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Quantile Loss (Pinball Loss) across all scenarios and quantile levels.

    Args:
        preds: (batch_size, num_quantiles) Predicted conditional quantiles.
        targets: (batch_size, num_scenarios) Empirical scenario target values.
        quantiles: (num_quantiles,) 1D tensor specifying the quantile levels
            (e.g. 0.1, 0.5, 0.9).

    Returns:
        torch.Tensor: The scalar aggregated pinball loss.
    """
    # Expand dimensions for broadcast matching
    # preds: (batch_size, num_quantiles, 1)
    preds_expanded = preds.unsqueeze(2)
    # targets: (batch_size, 1, num_scenarios)
    targets_expanded = targets.unsqueeze(1)

    # errors: (batch_size, num_quantiles, num_scenarios)
    errors = targets_expanded - preds_expanded

    # quantiles: (1, num_quantiles, 1)
    q = quantiles.view(1, -1, 1)

    # Pinball loss function calculation
    loss = torch.max(q * errors, (q - 1) * errors)

    # Average across the batch (dim 0) and scenarios (dim 2),
    # then sum the losses across the different quantile levels.
    return loss.mean(dim=(0, 2)).sum()
