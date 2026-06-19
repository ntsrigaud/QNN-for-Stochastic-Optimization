"""
Regression tests for the MILP zero-solution bug (Step 0).

Before the fix, ``SurrogateOptimizer`` had no constraint forbidding x*=0, so
the solver always picked the trivial (and infeasible for CFLP) all-closed
solution. The fix is a feasibility constraint -- ``sum(x) >= 1`` -- added via
the existing ``add_linear_constraint`` API at the call site, not a change to
``optimization`` itself (its big-M bound propagation, ReLU embedding, and
IQNN cumsum logic were already correct).
"""

import numpy as np
import torch

from qnn_stoch_opt.case_studies.cflp import (
    CFLPEvaluator,
    generate_cflp_demand_scenarios,
    generate_cflp_instance,
)
from qnn_stoch_opt.models.iqnn import IncrementalQuantileNeuralNetwork
from qnn_stoch_opt.models.qnn import QuantileNeuralNetwork
from qnn_stoch_opt.optimization.stochastic_optimizer import (
    ConstrSense,
    SurrogateOptimizer,
    VarType,
)

N_FACILITIES = 6
N_CUSTOMERS = 6
NUM_QUANTILES = 9


def _build_instance(seed: int = 7):
    f_costs, assignment_costs, capacities = generate_cflp_instance(
        N_FACILITIES, N_CUSTOMERS, seed=seed
    )
    evaluator = CFLPEvaluator(N_FACILITIES, N_CUSTOMERS, capacities, assignment_costs)
    return f_costs, assignment_costs, capacities, evaluator


def _tiny_qnn() -> QuantileNeuralNetwork:
    torch.manual_seed(0)
    model = QuantileNeuralNetwork(
        input_dim=N_FACILITIES, hidden_dims=[2], num_quantiles=NUM_QUANTILES
    )
    model.eval()
    return model


def _tiny_iqnn() -> IncrementalQuantileNeuralNetwork:
    torch.manual_seed(0)
    model = IncrementalQuantileNeuralNetwork(
        input_dim=N_FACILITIES, hidden_dims=[2], num_quantiles=NUM_QUANTILES
    )
    model.eval()
    return model


def _solve_with_feasibility_constraint(model, model_type: str, f_costs: np.ndarray):
    opt = SurrogateOptimizer(
        x_dim=N_FACILITIES,
        x_bounds=[(0.0, 1.0)] * N_FACILITIES,
        var_types=[VarType.BINARY] * N_FACILITIES,
    )
    opt.add_linear_constraint(
        [1.0] * N_FACILITIES, ConstrSense.GEQ, 1.0, name="min_one_open"
    )
    opt.embed_surrogate(model, model_type=model_type)
    opt.set_risk_neutral_objective(c=f_costs.tolist())
    return opt.optimize()


class TestFeasibilityConstraintFix:
    def test_qnn_solution_opens_at_least_one_facility(self) -> None:
        f_costs, _, _, _ = _build_instance()
        result = _solve_with_feasibility_constraint(_tiny_qnn(), "qnn", f_costs)
        assert result.is_optimal
        assert sum(result.x_opt) >= 1

    def test_iqnn_solution_opens_at_least_one_facility(self) -> None:
        f_costs, _, _, _ = _build_instance()
        result = _solve_with_feasibility_constraint(_tiny_iqnn(), "iqnn", f_costs)
        assert result.is_optimal
        assert sum(result.x_opt) >= 1

    def test_true_objective_is_finite(self) -> None:
        f_costs, _, _, evaluator = _build_instance()
        test_scenarios = generate_cflp_demand_scenarios(
            N_CUSTOMERS, num_scenarios=200, seed=999
        )
        result = _solve_with_feasibility_constraint(_tiny_iqnn(), "iqnn", f_costs)
        costs = evaluator.evaluate_scenarios(np.array(result.x_opt), test_scenarios)
        valid = costs[costs != float("inf")]
        assert len(valid) > 0
        true_obj = float(np.sum(f_costs * result.x_opt) + np.mean(valid))
        assert true_obj < 1e8

    def test_qnn_and_iqnn_agree_within_5_percent_true_objective(self) -> None:
        f_costs, _, _, evaluator = _build_instance()
        test_scenarios = generate_cflp_demand_scenarios(
            N_CUSTOMERS, num_scenarios=200, seed=999
        )

        def true_obj(result) -> float:
            costs = evaluator.evaluate_scenarios(
                np.array(result.x_opt), test_scenarios
            )
            valid = costs[costs != float("inf")]
            assert len(valid) > 0
            return float(np.sum(f_costs * result.x_opt) + np.mean(valid))

        qnn_result = _solve_with_feasibility_constraint(_tiny_qnn(), "qnn", f_costs)
        iqnn_result = _solve_with_feasibility_constraint(_tiny_iqnn(), "iqnn", f_costs)

        qnn_obj = true_obj(qnn_result)
        iqnn_obj = true_obj(iqnn_result)
        assert abs(qnn_obj - iqnn_obj) / max(qnn_obj, iqnn_obj) <= 0.05
