"""
Neural network surrogate models (QNN and IQNN).

Exports:
    QuantileNeuralNetwork: Standard multi-output QNN using ReLU hidden layers and
        a linear output layer to predict multiple quantiles simultaneously.
    IncrementalQuantileNeuralNetwork: Monotonicity-constrained IQNN that prevents
        quantile crossing by predicting non-negative increments via a ReLU output.
    pinball_loss: Vectorised quantile (pinball) loss for training QNN/IQNN models.
    train_model: Training loop with early stopping for QNN/IQNN models.
    CascadeQNN: Cascade of QNN/IQNN surrogates for multi-stage stochastic
        optimization, trained by backward induction.
"""

from qnn_stoch_opt.models.cascade import CascadeQNN
from qnn_stoch_opt.models.iqnn import IncrementalQuantileNeuralNetwork
from qnn_stoch_opt.models.loss import pinball_loss
from qnn_stoch_opt.models.qnn import QuantileNeuralNetwork
from qnn_stoch_opt.models.trainer import train_model

__all__ = [
    "QuantileNeuralNetwork",
    "IncrementalQuantileNeuralNetwork",
    "pinball_loss",
    "train_model",
    "CascadeQNN",
]
