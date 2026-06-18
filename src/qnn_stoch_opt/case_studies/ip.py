import gurobipy as gp
import numpy as np
from gurobipy import GRB


class IPEvaluator:
    """
    Evaluator for the second stage of the Investment Problem (IP-I-H).
    First stage: x_i in [0, U] (continuous investment)
    Second stage: y_j in {0, 1} (binary recourse decisions)
    Objective: Maximization problem. Since our SAA/Optimizer minimizes by default,
    we can formulate this as minimizing the negative objective.
    """

    def __init__(self, q: np.ndarray, W: np.ndarray, T: np.ndarray):
        """
        Args:
            q: Array of shape (m,) for second stage objective coefficients.
            W: Recourse matrix of shape (k, m).
            T: Technology matrix of shape (k, n).
        """
        self.q = q
        self.W = W
        self.T = T
        self.m = W.shape[1]
        self.k = W.shape[0]

        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

    def evaluate(self, x: np.ndarray, h_scenario: np.ndarray) -> float:
        """
        Solve the second stage for a given first-stage decision x and an RHS scenario.
        Note: The problem is maximization of q^T y, which is equivalent to minimizing
        -q^T y.
        We return the maximization objective value directly.

        Args:
            x: Array of shape (n,) with continuous investment decisions.
            h_scenario: Array of shape (k,) with realized capacities/RHS.

        Returns:
            Optimal second-stage maximization value, or float('-inf') if infeasible.
        """
        model = gp.Model("ip_second_stage", env=self.env)

        # Second stage variables: y_j in {0, 1}
        y = model.addMVar(self.m, vtype=GRB.BINARY, name="y")

        # Objective: Maximize q^T y
        model.setObjective(self.q @ y, GRB.MAXIMIZE)

        # Constraints: W y <= h - T x
        rhs = h_scenario - self.T @ x
        model.addConstr(self.W @ y <= rhs)

        model.optimize()

        if model.status == GRB.OPTIMAL:
            return float(model.ObjVal)
        else:
            return float("-inf")

    def evaluate_scenarios(self, x: np.ndarray, h_scenarios: np.ndarray) -> np.ndarray:
        """
        Evaluates the second-stage maximization cost over an entire batch of scenarios.
        """
        num_scenarios = len(h_scenarios)
        costs = np.zeros(num_scenarios)
        for i in range(num_scenarios):
            costs[i] = self.evaluate(x, h_scenarios[i])
        return costs


def generate_ip_instance(
    n: int, m: int, k: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a synthetic instance of the IP problem.
    """
    rng = np.random.default_rng(seed)

    # First stage costs
    c = rng.uniform(1.0, 3.0, size=n)

    # Second stage costs
    q = rng.uniform(5.0, 15.0, size=m)

    # Recourse and technology matrices
    W = rng.uniform(0.5, 2.5, size=(k, m))
    T = rng.uniform(0.1, 1.0, size=(k, n))

    return c, q, W, T


def generate_ip_h_scenarios(k: int, num_scenarios: int, seed: int = 42) -> np.ndarray:
    """
    Generates scenarios for the RHS vector h.
    """
    rng = np.random.default_rng(seed)
    # Scenarios ~ Normal(10, 2)
    h_scenarios = rng.normal(loc=10.0, scale=2.0, size=(num_scenarios, k))
    return h_scenarios
