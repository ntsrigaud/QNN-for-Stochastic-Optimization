from typing import List, Tuple

import gurobipy as gp
import torch.nn as nn
from gurobipy import GRB

from qnn_stoch_opt.optimization.milp_formulation import QNNtoMILP


class SurrogateOptimizer:
    """
    Formulates and solves the first-stage optimization problem by embedding the
    neural network surrogate models to estimate the second-stage costs.

    Supports three objective modes:
      - Risk-neutral:  min c^T x + E[V(x, xi)]
      - Risk-averse:   min c^T x + CVaR_alpha[V(x, xi)]
      - Mean-risk:     min c^T x + E[V(x, xi)] + lambda * CVaR_alpha[V(x, xi)]
    """

    def __init__(self, x_dim: int, x_bounds: List[Tuple[float, float]]):
        self.x_dim = x_dim
        self.x_bounds = x_bounds
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.model = gp.Model("surrogate_optimizer", env=self.env)

        # Define first stage variables
        self.x_vars: List[gp.Var] = []
        for i in range(x_dim):
            lb, ub = x_bounds[i]
            var = self.model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"x_{i}")
            self.x_vars.append(var)

        self.embedder = QNNtoMILP(self.model)
        self.quantiles_vars: List[gp.Var] = []

    def embed_surrogate(self, torch_model: nn.Module, model_type: str = "qnn") -> None:
        """
        Embed the surrogate neural network into the Gurobi model.

        Args:
            torch_model: The trained PyTorch model (QNN or IQNN).
            model_type: Either ``"qnn"`` or ``"iqnn"``.
        """
        if model_type == "qnn":
            self.quantiles_vars = self.embedder.embed_qnn(
                torch_model, self.x_vars, self.x_bounds
            )
        elif model_type == "iqnn":
            self.quantiles_vars = self.embedder.embed_iqnn(
                torch_model, self.x_vars, self.x_bounds
            )
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. Must be 'qnn' or 'iqnn'."
            )

    def set_risk_neutral_objective(self, c: List[float]) -> None:
        """
        Sets a risk-neutral objective: minimize c^T x + E[V(x, xi)].

        The expected second-stage cost E[V] is approximated as the mean of all
        predicted quantiles.

        Args:
            c: Linear cost coefficients for the first-stage variables.
        """
        num_quantiles = len(self.quantiles_vars)
        expected_second_stage = gp.LinExpr()
        for q_var in self.quantiles_vars:
            expected_second_stage.add(q_var, 1.0 / num_quantiles)

        first_stage_cost = gp.LinExpr()
        for i, var in enumerate(self.x_vars):
            first_stage_cost.add(var, c[i])

        self.model.setObjective(first_stage_cost + expected_second_stage, GRB.MINIMIZE)

    def set_risk_averse_objective(self, c: List[float], alpha: float) -> None:
        """
        Sets a purely risk-averse CVaR objective at confidence level alpha:
        minimize c^T x + CVaR_alpha[V(x, xi)].

        CVaR at level alpha is approximated as the mean of the predicted
        quantiles at or above the alpha tail.

        Args:
            c: Linear cost coefficients for the first-stage variables.
            alpha: Confidence level in [0, 1). Higher values are more risk-averse.

        Raises:
            ValueError: If alpha leaves no quantiles in the tail.
        """
        num_quantiles = len(self.quantiles_vars)

        # The tail starts at index: floor(alpha * num_quantiles)
        tail_start_idx = int(alpha * num_quantiles)
        tail_vars = self.quantiles_vars[tail_start_idx:]

        if not tail_vars:
            raise ValueError(
                f"Alpha={alpha} is too high: no quantiles remain in the CVaR tail "
                f"(num_quantiles={num_quantiles})."
            )

        cvar_second_stage = gp.LinExpr()
        for q_var in tail_vars:
            cvar_second_stage.add(q_var, 1.0 / len(tail_vars))

        first_stage_cost = gp.LinExpr()
        for i, var in enumerate(self.x_vars):
            first_stage_cost.add(var, c[i])

        self.model.setObjective(first_stage_cost + cvar_second_stage, GRB.MINIMIZE)

    def set_mean_risk_objective(self, c: List[float], alpha: float, lam: float) -> None:
        """
        Sets a mean-risk objective combining expected cost and CVaR:
        minimize c^T x + E[V(x, xi)] + lambda * CVaR_alpha[V(x, xi)].

        This directly matches the mean-risk formulation from the paper
        (Section 2.1), allowing the risk-aversion level to be tuned via ``lam``.

        Args:
            c: Linear cost coefficients for the first-stage variables.
            alpha: CVaR confidence level in [0, 1).
            lam: Non-negative risk-aversion weight (lambda >= 0).
                 When lam=0, reduces to risk-neutral; as lam increases, the
                 solution becomes more risk-averse.

        Raises:
            ValueError: If lam < 0 or if alpha leaves no quantiles in the tail.
        """
        if lam < 0:
            raise ValueError(f"Risk-aversion weight lam must be >= 0, got {lam}.")

        num_quantiles = len(self.quantiles_vars)

        # Expected second-stage cost: mean of all quantiles
        expected_second_stage = gp.LinExpr()
        for q_var in self.quantiles_vars:
            expected_second_stage.add(q_var, 1.0 / num_quantiles)

        # CVaR tail
        tail_start_idx = int(alpha * num_quantiles)
        tail_vars = self.quantiles_vars[tail_start_idx:]

        if not tail_vars:
            raise ValueError(
                f"Alpha={alpha} is too high: no quantiles remain in the CVaR tail "
                f"(num_quantiles={num_quantiles})."
            )

        cvar_second_stage = gp.LinExpr()
        for q_var in tail_vars:
            cvar_second_stage.add(q_var, lam / len(tail_vars))

        first_stage_cost = gp.LinExpr()
        for i, var in enumerate(self.x_vars):
            first_stage_cost.add(var, c[i])

        self.model.setObjective(
            first_stage_cost + expected_second_stage + cvar_second_stage, GRB.MINIMIZE
        )

    def optimize(self) -> Tuple[List[float], float]:
        """
        Solves the embedded MILP problem.

        Returns:
            Tuple of (optimal first-stage decisions, objective value).

        Raises:
            RuntimeError: If the solver does not find an optimal solution.
        """
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            x_opt = [var.X for var in self.x_vars]
            return x_opt, self.model.ObjVal
        else:
            raise RuntimeError(
                f"Optimization failed with status code {self.model.status}"
            )
