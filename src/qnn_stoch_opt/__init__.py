"""
QNN-for-Stochastic-Optimization package.

Top-level convenience re-exports from each subpackage.
"""

from qnn_stoch_opt.data.dataset import StochasticOptimizationDataset, create_dataloaders
from qnn_stoch_opt.data.saa_solver import SecondStageEvaluator
from qnn_stoch_opt.data.scenario_generation import (
    generate_normal_scenarios,
    generate_uniform_scenarios,
)
from qnn_stoch_opt.models.iqnn import IncrementalQuantileNeuralNetwork
from qnn_stoch_opt.models.loss import pinball_loss
from qnn_stoch_opt.models.qnn import QuantileNeuralNetwork
from qnn_stoch_opt.models.trainer import train_model
from qnn_stoch_opt.optimization.milp_formulation import QNNtoMILP
from qnn_stoch_opt.optimization.stochastic_optimizer import (
    ConstrSense,
    SolveResult,
    SurrogateOptimizer,
    VarType,
)

__all__ = [
    # data
    "StochasticOptimizationDataset",
    "create_dataloaders",
    "SecondStageEvaluator",
    "generate_normal_scenarios",
    "generate_uniform_scenarios",
    # models
    "QuantileNeuralNetwork",
    "IncrementalQuantileNeuralNetwork",
    "pinball_loss",
    "train_model",
    # optimization
    "QNNtoMILP",
    "SurrogateOptimizer",
    "VarType",
    "ConstrSense",
    "SolveResult",
]
