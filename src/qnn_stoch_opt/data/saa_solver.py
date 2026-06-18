import gurobipy as gp
import numpy as np
from gurobipy import GRB


class SecondStageEvaluator:
    """
    Evaluates the second-stage objective (recourse problem) for a given set of
    first-stage decisions and realized scenarios. This serves as the ground truth
    generating function.

    This implementation assumes a standard linear two-stage structure:
    min q^T Y
    s.t. W Y <= h(xi) - T(xi) X
         Y >= 0
    """

    def __init__(self, q: np.ndarray, W: np.ndarray):
        """
        Initialize the evaluator with deterministic second-stage parameters.

        Args:
            q: Second-stage cost vector.
            W: Recourse matrix.
        """
        self.q = q
        self.W = W
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)  # Suppress Gurobi output for bulk evaluation
        self.env.start()

    def evaluate(self, X: np.ndarray, h_xi: np.ndarray, T_xi: np.ndarray) -> float:
        """
        Evaluate the second-stage cost for a single specific scenario.

        Args:
            X: First-stage decisions.
            h_xi: Realized right-hand side vector for this scenario.
            T_xi: Realized technology/transition matrix for this scenario.

        Returns:
            float: Optimal second-stage cost (or infinity if infeasible).
        """
        model = gp.Model("second_stage", env=self.env)
        num_y = len(self.q)

        # Define continuous recourse variables Y >= 0
        Y = model.addMVar(num_y, vtype=GRB.CONTINUOUS, lb=0.0, name="Y")

        # Add constraints: W Y <= h(xi) - T(xi) X
        rhs = h_xi - T_xi @ X
        model.addConstr(self.W @ Y <= rhs, name="recourse_constr")

        # Set Objective: Minimize q^T Y
        model.setObjective(self.q @ Y, GRB.MINIMIZE)

        model.optimize()

        if model.status == GRB.OPTIMAL:
            return float(model.ObjVal)
        else:
            return float("inf")

    def evaluate_scenarios(
        self, X: np.ndarray, h_scenarios: np.ndarray, T_scenarios: np.ndarray
    ) -> np.ndarray:
        """
        Evaluates the second-stage cost over an entire batch of scenarios.

        Args:
            X: First-stage decisions.
            h_scenarios: Array of realized right-hand side vectors.
            T_scenarios: Array of realized transition matrices.

        Returns:
            np.ndarray: Array of second-stage costs corresponding to the scenarios.
        """
        num_scenarios = len(h_scenarios)
        costs = np.zeros(num_scenarios)
        for i in range(num_scenarios):
            costs[i] = self.evaluate(X, h_scenarios[i], T_scenarios[i])
        return costs
