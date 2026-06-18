import gurobipy as gp
import numpy as np
from gurobipy import GRB


class CFLPEvaluator:
    """
    Evaluator for the second stage of the Capacitated Facility Location Problem (CFLP).
    First stage: x_i in {0, 1} (open facility i)
    Second stage: y_ij in {0, 1} (assign customer j to facility i)
    """

    def __init__(
        self,
        num_facilities: int,
        num_customers: int,
        capacities: np.ndarray,
        assignment_costs: np.ndarray,
    ):
        """
        Args:
            num_facilities: Number of facilities (n).
            num_customers: Number of customers (m).
            capacities: Array of shape (n,) with the capacity of each facility.
            assignment_costs: Array of shape (n, m) with the cost of assigning
                customer j to facility i.
        """
        self.n = num_facilities
        self.m = num_customers
        self.capacities = capacities
        self.costs = assignment_costs

        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

    def evaluate(self, x: np.ndarray, demand_scenario: np.ndarray) -> float:
        """
        Solve the second stage for a given first-stage decision x and a customer
        demand scenario.

        Args:
            x: Array of shape (n,) with binary decisions on facility opening.
            demand_scenario: Array of shape (m,) with customer demands.

        Returns:
            Optimal second-stage cost, or float('inf') if infeasible.
        """
        model = gp.Model("cflp_second_stage", env=self.env)

        # Second stage variables: y_ij in {0, 1}
        # Flattened array of length n*m
        y = model.addMVar((self.n, self.m), vtype=GRB.BINARY, name="y")

        # Objective: minimize sum_i sum_j c_ij * y_ij
        model.setObjective(
            gp.quicksum(
                self.costs[i, j] * y[i, j].item()
                for i in range(self.n)
                for j in range(self.m)
            ),
            GRB.MINIMIZE,
        )

        # Constraints
        # 1. Capacity constraints: sum_j d_j * y_ij <= C_i * x_i for all i
        for i in range(self.n):
            model.addConstr(
                gp.quicksum(demand_scenario[j] * y[i, j].item() for j in range(self.m))
                <= self.capacities[i] * x[i]
            )

        # 2. Demand satisfaction: sum_i y_ij == 1 for all j
        for j in range(self.m):
            model.addConstr(gp.quicksum(y[i, j].item() for i in range(self.n)) == 1)

        # 3. Can only assign to open facilities: y_ij <= x_i for all i, j
        for i in range(self.n):
            for j in range(self.m):
                model.addConstr(y[i, j].item() <= x[i])

        model.optimize()

        if model.status == GRB.OPTIMAL:
            return float(model.ObjVal)
        else:
            return float("inf")

    def evaluate_scenarios(
        self, x: np.ndarray, demand_scenarios: np.ndarray
    ) -> np.ndarray:
        """
        Evaluates the second-stage cost over an entire batch of scenarios.
        """
        num_scenarios = len(demand_scenarios)
        costs = np.zeros(num_scenarios)
        for i in range(num_scenarios):
            costs[i] = self.evaluate(x, demand_scenarios[i])
        return costs


def generate_cflp_instance(
    num_facilities: int, num_customers: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a synthetic instance of the CFLP problem.
    """
    rng = np.random.default_rng(seed)

    # First stage costs
    f_costs = rng.uniform(100, 500, size=num_facilities)

    # Second stage assignment costs
    assignment_costs = rng.uniform(10, 50, size=(num_facilities, num_customers))

    # Capacities
    capacities = rng.uniform(100, 200, size=num_facilities)

    return f_costs, assignment_costs, capacities


def generate_cflp_demand_scenarios(
    num_customers: int, num_scenarios: int, seed: int = 42
) -> np.ndarray:
    """
    Generates scenarios for customer demands.
    """
    rng = np.random.default_rng(seed)
    # Demand scenarios ~ Uniform(10, 30)
    demands = rng.uniform(10, 30, size=(num_scenarios, num_customers))
    return demands
