"""
MILP formulations and Gurobi integration for embedding trained surrogate models.

Exports:
    QNNtoMILP: Translates a trained PyTorch QNN or IQNN into exact MILP constraints
        within a Gurobi model using the Big-M method and interval-arithmetic bound
        tightening.
    SurrogateOptimizer: Formulates and solves the first-stage optimisation problem
        by embedding a surrogate neural network.  Supports risk-neutral (expected
        cost), risk-averse (CVaR), and mean-risk (E + λ·CVaR) objectives.
"""

from qnn_stoch_opt.optimization.milp_formulation import QNNtoMILP
from qnn_stoch_opt.optimization.stochastic_optimizer import SurrogateOptimizer

__all__ = ["QNNtoMILP", "SurrogateOptimizer"]
