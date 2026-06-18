"""
Data generation and processing modules.

Exports:
    StochasticOptimizationDataset: PyTorch Dataset pairing first-stage decisions
        with empirical second-stage cost distributions.
    create_dataloaders: Factory that splits data and returns train/test DataLoaders.
    SecondStageEvaluator: Gurobi-based ground-truth evaluator for the recourse problem.
    generate_normal_scenarios: Multivariate-normal scenario sampler.
    generate_uniform_scenarios: Independent-uniform scenario sampler.
"""

from qnn_stoch_opt.data.dataset import StochasticOptimizationDataset, create_dataloaders
from qnn_stoch_opt.data.saa_solver import SecondStageEvaluator
from qnn_stoch_opt.data.scenario_generation import (
    generate_normal_scenarios,
    generate_uniform_scenarios,
)

__all__ = [
    "StochasticOptimizationDataset",
    "create_dataloaders",
    "SecondStageEvaluator",
    "generate_normal_scenarios",
    "generate_uniform_scenarios",
]
