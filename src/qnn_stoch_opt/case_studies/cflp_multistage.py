"""
Three-stage Capacitated Facility Location Problem (CFLP) with endogenous
demand. Extends :mod:`qnn_stoch_opt.case_studies.cflp` to a second recourse
stage:

Stage 0 (here-and-now): binary facility opening decisions x_0 in {0,1}^n.
    Cost: f_0^T x_0.
Stage 1 (after observing demand wave 1): binary capacity expansion decisions
    x_1 in {0,1}^n. Cost: f_1^T x_1 + assignment cost for demand wave 1.
    Random parameter: demand_1 ~ Uniform(50, 150)^m.
Stage 2 (after observing demand wave 2): assignment decisions only.
    Cost: assignment cost for demand wave 2.
    Random parameter: demand_2 ~ Uniform(60 + 5*sum(x_1), 180 + 5*sum(x_1))^m
    -- the endogenous coupling: expansion signals quality and increases
    future demand.
"""

from __future__ import annotations

from typing import Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def generate_3stage_cflp_instance(
    n: int, m: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic 3-stage CFLP instance.

    Returns:
        f_0_costs: Facility opening costs, shape (n,), ~U(100,500).
        f_1_costs: Capacity expansion costs, shape (n,), ~U(50,200).
        assignment_costs: Per-unit assignment costs, shape (n,m), ~U(10,50),
            shared across both demand waves.
        base_capacities: Capacity before expansion, shape (n,), ~U(150,300).
        expanded_capacities: Capacity after expansion (50% boost), shape (n,).

    Capacities are scaled relative to demand_1 ~ U(50,150)^m and demand_2 ~
    U(60,180)^m (before the endogenous shift) so that bin-packing feasibility
    is common rather than a knife-edge event: aggregate base capacity is
    comfortably above typical demand_1, and aggregate expanded capacity stays
    above typical demand_2 even after the endogenous shift increases it.
    """
    rng = np.random.default_rng(seed)

    f_0_costs = rng.uniform(100, 500, size=n)
    f_1_costs = rng.uniform(50, 200, size=n)
    assignment_costs = rng.uniform(10, 50, size=(n, m))
    base_capacities = rng.uniform(150, 300, size=n)
    expanded_capacities = base_capacities * 1.5

    return f_0_costs, f_1_costs, assignment_costs, base_capacities, expanded_capacities


class ThreeStageCFLPEvaluator:
    """Evaluator for the recourse stages of the 3-stage CFLP.

    Effective capacity at facility i given (x_0, x_1):
        cap_i = x_0_i * (base_i + x_1_i * (expanded_i - base_i))

    This is used for both recourse-stage assignment LPs. Note that the
    *surrogate* trained on stage-2 data only conditions on x_1 (per the
    backward-induction data-generation procedure below); x_0 is sampled with
    a high open-rate during stage-2 data generation so its effect there stays
    small, consistent with the modeling assumption that x_0's contribution
    is captured by the stage-1 surrogate.
    """

    def __init__(
        self,
        num_facilities: int,
        num_customers: int,
        f_0_costs: np.ndarray,
        f_1_costs: np.ndarray,
        assignment_costs: np.ndarray,
        base_capacities: np.ndarray,
        expanded_capacities: np.ndarray,
    ):
        self.n = num_facilities
        self.m = num_customers
        self.f_0 = f_0_costs
        self.f_1 = f_1_costs
        self.costs = assignment_costs
        self.base_capacities = base_capacities
        self.expanded_capacities = expanded_capacities

        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

    def _effective_capacity(self, x_0: np.ndarray, x_1: np.ndarray) -> np.ndarray:
        capacity_gain = self.expanded_capacities - self.base_capacities
        return x_0 * (self.base_capacities + x_1 * capacity_gain)

    def _solve_assignment(
        self, x_0: np.ndarray, demand: np.ndarray, capacity: np.ndarray
    ) -> float:
        model = gp.Model("cflp_assignment", env=self.env)
        y = model.addMVar((self.n, self.m), vtype=GRB.BINARY, name="y")

        model.setObjective(
            gp.quicksum(
                self.costs[i, j] * y[i, j].item()
                for i in range(self.n)
                for j in range(self.m)
            ),
            GRB.MINIMIZE,
        )

        for i in range(self.n):
            model.addConstr(
                gp.quicksum(demand[j] * y[i, j].item() for j in range(self.m))
                <= capacity[i]
            )
        for j in range(self.m):
            model.addConstr(gp.quicksum(y[i, j].item() for i in range(self.n)) == 1)
        for i in range(self.n):
            for j in range(self.m):
                model.addConstr(y[i, j].item() <= x_0[i])

        model.optimize()

        if model.status == GRB.OPTIMAL:
            return float(model.ObjVal)
        return float("inf")

    def evaluate_stage1(
        self, x_0: np.ndarray, demand_1: np.ndarray, x_1: np.ndarray
    ) -> float:
        """Stage-1 recourse cost only (assignment of demand_1), not f_1^T x_1."""
        capacity = self._effective_capacity(x_0, x_1)
        return self._solve_assignment(x_0, demand_1, capacity)

    def evaluate_stage2(
        self, x_0: np.ndarray, x_1: np.ndarray, demand_2: np.ndarray
    ) -> float:
        """Stage-2 recourse cost (assignment of demand_2)."""
        capacity = self._effective_capacity(x_0, x_1)
        return self._solve_assignment(x_0, demand_2, capacity)

    def sample_demand_1(self, m: int, seed: int) -> np.ndarray:
        """Sample a single demand_1 scenario ~ Uniform(50, 150)^m."""
        rng = np.random.default_rng(seed)
        return rng.uniform(50, 150, size=m)

    def sample_demand_2(self, x_1: np.ndarray, m: int, seed: int) -> np.ndarray:
        """Sample a single demand_2 scenario, endogenously shifted by x_1.

        ~ Uniform(60 + 5*sum(x_1), 180 + 5*sum(x_1))^m
        """
        rng = np.random.default_rng(seed)
        shift = 5.0 * float(np.sum(x_1))
        return rng.uniform(60 + shift, 180 + shift, size=m)


def _random_open_decision(
    rng: np.random.Generator, n: int, p_open: float
) -> np.ndarray:
    """Sample a binary decision vector with at least one entry set to 1."""
    while True:
        x = (rng.random(n) < p_open).astype(np.float32)
        if x.sum() > 0:
            return x


def generate_backward_training_data(
    evaluator: ThreeStageCFLPEvaluator,
    n_stage0_samples: int,
    n_stage1_samples: int,
    n_facilities: int,
    n_customers: int,
    seed: int,
    n_x1_candidates: int = 5,
    n_mc_demand2: int = 5,
    infeasibility_penalty: float = 2000.0,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Generate training data for :class:`CascadeQNN` by backward induction.

    Stage 2 data (terminal surrogate, input = x_1): sample random (x_0, x_1)
    pairs and a random demand_2, solve evaluate_stage2, record (x_1, v_2)
    pairs (x_0 is needed to call the evaluator but is not part of the
    recorded input, per the modeling assumption documented on
    :class:`ThreeStageCFLPEvaluator`). Demand draws that turn out infeasible
    (capacity cannot serve every customer) are recorded with
    ``infeasibility_penalty`` rather than dropped, so the surrogate learns
    that thin capacity carries real downside risk -- without this, a stockout
    is simply absent from the training data rather than expensive, which
    erases any incentive to expand capacity.

    Stage 1 data (input = x_0): for each sampled x_0 and demand_1, score
    several candidate x_1 decisions by
    ``evaluate_stage1(x_0, demand_1, x_1) + mean_over_demand2_draws(
    evaluate_stage2(x_0, x_1, demand_2))`` (same infeasibility-penalty
    treatment) and keep the best as a Monte Carlo estimate of the optimal
    continuation value -- record (x_0, best_score) pairs.

    Returns:
        (stage1_data, stage2_data), each a tuple of (X, v) arrays.
    """
    rng = np.random.default_rng(seed)
    n, m = n_facilities, n_customers

    def _cost_or_penalty(cost: float) -> float:
        return infeasibility_penalty if cost == float("inf") else cost

    # Stage 2 (terminal) data. p_open/p_expand are biased high: this is a
    # capacitated problem, so most low-capacity (x_0, x_1) draws are simply
    # infeasible for realistic demand -- biasing the sampler keeps the
    # feasible-sample share high enough that the data isn't dominated by the
    # penalty value.
    X1_list, v2_list = [], []
    for _ in range(n_stage1_samples):
        x_0 = _random_open_decision(rng, n, p_open=0.9)
        x_1 = (rng.random(n) < 0.6).astype(np.float32)
        demand_2 = evaluator.sample_demand_2(
            x_1, m, seed=int(rng.integers(0, 1_000_000))
        )
        v_2 = evaluator.evaluate_stage2(x_0, x_1, demand_2)
        X1_list.append(x_1)
        v2_list.append(_cost_or_penalty(v_2))
    stage2_data = (
        np.array(X1_list, dtype=np.float32),
        np.array(v2_list, dtype=np.float32),
    )

    # Stage 1 data: best-of-K candidate x_1's against a Monte Carlo
    # continuation-value estimate.
    X0_list, v1_list = [], []
    for _ in range(n_stage0_samples):
        x_0 = _random_open_decision(rng, n, p_open=0.9)
        demand_1 = evaluator.sample_demand_1(m, seed=int(rng.integers(0, 1_000_000)))

        best_score = float("inf")
        for _ in range(n_x1_candidates):
            x_1_candidate = (rng.random(n) < 0.6).astype(np.float32)
            v_1 = _cost_or_penalty(
                evaluator.evaluate_stage1(x_0, demand_1, x_1_candidate)
            )

            mc_demand2_costs = []
            for _ in range(n_mc_demand2):
                demand_2 = evaluator.sample_demand_2(
                    x_1_candidate, m, seed=int(rng.integers(0, 1_000_000))
                )
                v_2 = evaluator.evaluate_stage2(x_0, x_1_candidate, demand_2)
                mc_demand2_costs.append(_cost_or_penalty(v_2))

            score = v_1 + float(np.mean(mc_demand2_costs))
            best_score = min(best_score, score)

        X0_list.append(x_0)
        v1_list.append(best_score)

    stage1_data = (
        np.array(X0_list, dtype=np.float32),
        np.array(v1_list, dtype=np.float32),
    )

    return stage1_data, stage2_data
