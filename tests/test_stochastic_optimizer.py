"""
Tests for Phase 5: Optimization Problem Formulation.

Covers:
  - VarType / ConstrSense enums
  - SolveResult dataclass
  - SurrogateOptimizer constructor validation
  - Integer and binary first-stage variable creation
  - add_linear_constraint / add_constraints_matrix APIs
  - embed_surrogate with delta (Δ-tolerance QNN correction, Algorithm 2)
  - set_risk_neutral_objective / set_risk_averse_objective / set_mean_risk_objective
  - set_time_limit / set_mip_gap / set_threads solver parameter controls
  - optimize() returning SolveResult for feasible and infeasible models
  - Error paths (missing surrogate, bad inputs)
  - Public API exports from optimization package and root package
"""

import numpy as np
import pytest
import torch
from gurobipy import GRB

from qnn_stoch_opt.models.iqnn import IncrementalQuantileNeuralNetwork
from qnn_stoch_opt.models.qnn import QuantileNeuralNetwork
from qnn_stoch_opt.optimization.stochastic_optimizer import (
    ConstrSense,
    SolveResult,
    SurrogateOptimizer,
    VarType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_qnn(input_dim: int = 2, num_q: int = 5) -> QuantileNeuralNetwork:
    torch.manual_seed(42)
    m = QuantileNeuralNetwork(input_dim=input_dim, hidden_dims=[4], num_quantiles=num_q)
    m.eval()
    return m


def _tiny_iqnn(input_dim: int = 2, num_q: int = 5) -> IncrementalQuantileNeuralNetwork:
    torch.manual_seed(42)
    m = IncrementalQuantileNeuralNetwork(
        input_dim=input_dim, hidden_dims=[4], num_quantiles=num_q
    )
    m.eval()
    return m


def _cont_opt(x_dim: int = 2) -> SurrogateOptimizer:
    """Return a simple continuous-variable optimizer."""
    return SurrogateOptimizer(
        x_dim=x_dim,
        x_bounds=[(-1.0, 1.0)] * x_dim,
    )


# ---------------------------------------------------------------------------
# VarType / ConstrSense enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_var_type_values(self) -> None:
        assert VarType.CONTINUOUS.value == "C"
        assert VarType.INTEGER.value == "I"
        assert VarType.BINARY.value == "B"

    def test_constr_sense_values(self) -> None:
        assert ConstrSense.LEQ.value == "<="
        assert ConstrSense.EQ.value == "=="
        assert ConstrSense.GEQ.value == ">="


# ---------------------------------------------------------------------------
# SolveResult dataclass
# ---------------------------------------------------------------------------


class TestSolveResult:
    def test_is_optimal_true(self) -> None:
        r = SolveResult(status=GRB.OPTIMAL, x_opt=[1.0], obj_val=1.0)
        assert r.is_optimal

    def test_is_optimal_false_for_infeasible(self) -> None:
        r = SolveResult(status=GRB.INFEASIBLE)
        assert not r.is_optimal

    def test_defaults(self) -> None:
        r = SolveResult(status=GRB.OPTIMAL)
        assert r.x_opt == []
        assert r.obj_val is None
        assert r.solve_time_s == 0.0
        assert r.mip_gap == 0.0
        assert r.quantile_vals == []


# ---------------------------------------------------------------------------
# SurrogateOptimizer — constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_mismatched_x_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="x_bounds length"):
            SurrogateOptimizer(x_dim=3, x_bounds=[(0.0, 1.0)])

    def test_mismatched_var_types_raises(self) -> None:
        with pytest.raises(ValueError, match="var_types length"):
            SurrogateOptimizer(
                x_dim=2,
                x_bounds=[(0.0, 1.0), (0.0, 1.0)],
                var_types=[VarType.BINARY],
            )

    def test_default_var_types_are_continuous(self) -> None:
        opt = SurrogateOptimizer(x_dim=2, x_bounds=[(0.0, 1.0)] * 2)
        opt.model.update()
        for v in opt.x_vars:
            assert v.VType == GRB.CONTINUOUS


# ---------------------------------------------------------------------------
# Integer and binary variable creation
# ---------------------------------------------------------------------------


class TestIntegerBinaryVariables:
    def test_binary_variables_created(self) -> None:
        opt = SurrogateOptimizer(
            x_dim=3,
            x_bounds=[(0.0, 1.0)] * 3,
            var_types=[VarType.BINARY] * 3,
        )
        opt.model.update()
        assert opt.model.NumBinVars == 3

    def test_integer_variables_created(self) -> None:
        opt = SurrogateOptimizer(
            x_dim=2,
            x_bounds=[(0.0, 5.0)] * 2,
            var_types=[VarType.INTEGER, VarType.INTEGER],
        )
        opt.model.update()
        assert opt.model.NumIntVars == 2

    def test_mixed_var_types(self) -> None:
        opt = SurrogateOptimizer(
            x_dim=3,
            x_bounds=[(0.0, 1.0), (0.0, 5.0), (-1.0, 1.0)],
            var_types=[VarType.BINARY, VarType.INTEGER, VarType.CONTINUOUS],
        )
        opt.model.update()
        assert opt.model.NumBinVars == 1
        assert opt.model.NumIntVars == 2  # Gurobi counts binary as a subset of integer

    def test_binary_optimizer_solves_correctly(self) -> None:
        """Binary first-stage optimizer with a simple budget constraint."""
        n = 4
        opt = SurrogateOptimizer(
            x_dim=n,
            x_bounds=[(0.0, 1.0)] * n,
            var_types=[VarType.BINARY] * n,
        )
        # Budget: at most 2 of 4 binaries can be 1
        opt.add_linear_constraint([1.0] * n, ConstrSense.LEQ, 2.0, name="budget")
        qnn = _tiny_qnn(input_dim=n, num_q=5)
        opt.embed_surrogate(qnn, model_type="qnn")
        opt.set_risk_neutral_objective(c=[0.0] * n)
        result = opt.optimize()

        assert result.is_optimal
        assert all(v in (0.0, 1.0) or abs(v - round(v)) < 1e-6 for v in result.x_opt)
        assert sum(result.x_opt) <= 2.0 + 1e-6


# ---------------------------------------------------------------------------
# add_linear_constraint
# ---------------------------------------------------------------------------


class TestAddLinearConstraint:
    def test_wrong_coeffs_length_raises(self) -> None:
        opt = _cont_opt(x_dim=3)
        with pytest.raises(ValueError, match="coeffs length"):
            opt.add_linear_constraint([1.0, 2.0], ConstrSense.LEQ, 5.0)

    def test_leq_constraint_added(self) -> None:
        opt = _cont_opt(x_dim=2)
        n_before = opt.model.NumConstrs
        opt.add_linear_constraint([1.0, 1.0], ConstrSense.LEQ, 1.5, name="sum_leq")
        opt.model.update()
        assert opt.model.NumConstrs == n_before + 1

    def test_eq_constraint_added(self) -> None:
        opt = _cont_opt(x_dim=2)
        opt.add_linear_constraint([1.0, -1.0], ConstrSense.EQ, 0.0, name="equal")
        opt.model.update()
        c = opt.model.getConstrByName("equal")
        assert c is not None
        assert c.Sense == GRB.EQUAL

    def test_geq_constraint_added(self) -> None:
        opt = _cont_opt(x_dim=2)
        opt.add_linear_constraint([1.0, 1.0], ConstrSense.GEQ, 0.5, name="sum_geq")
        opt.model.update()
        c = opt.model.getConstrByName("sum_geq")
        assert c is not None
        assert c.Sense == GRB.GREATER_EQUAL

    def test_constraint_enforced_at_optimum(self) -> None:
        """sum(x) <= 0.5 should be satisfied at optimum when maximizing sum(x)."""
        opt = SurrogateOptimizer(x_dim=2, x_bounds=[(0.0, 1.0)] * 2)
        opt.add_linear_constraint([1.0, 1.0], ConstrSense.LEQ, 0.5, name="cap")
        qnn = _tiny_qnn(input_dim=2, num_q=3)
        opt.embed_surrogate(qnn, model_type="qnn")
        opt.set_risk_neutral_objective(c=[-1.0, -1.0])  # maximize sum(x)
        result = opt.optimize()

        assert result.is_optimal
        assert sum(result.x_opt) <= 0.5 + 1e-6


# ---------------------------------------------------------------------------
# add_constraints_matrix
# ---------------------------------------------------------------------------


class TestAddConstraintsMatrix:
    def test_wrong_A_shape_raises(self) -> None:
        opt = _cont_opt(x_dim=3)
        A = np.ones((2, 4))  # wrong column count
        b = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="A must have shape"):
            opt.add_constraints_matrix(A, ConstrSense.LEQ, b)

    def test_wrong_b_shape_raises(self) -> None:
        opt = _cont_opt(x_dim=3)
        A = np.ones((2, 3))
        b = np.array([1.0])  # wrong length
        with pytest.raises(ValueError, match="b must have shape"):
            opt.add_constraints_matrix(A, ConstrSense.LEQ, b)

    def test_matrix_constraints_count(self) -> None:
        opt = _cont_opt(x_dim=3)
        A = np.ones((4, 3))
        b = np.ones(4)
        constrs = opt.add_constraints_matrix(A, ConstrSense.LEQ, b)
        opt.model.update()
        assert len(constrs) == 4
        assert opt.model.NumConstrs == 4

    def test_named_matrix_constraints(self) -> None:
        opt = _cont_opt(x_dim=2)
        A = np.eye(2)
        b = np.array([0.8, 0.9])
        names = ["ub_x0", "ub_x1"]
        opt.add_constraints_matrix(A, ConstrSense.LEQ, b, names=names)
        opt.model.update()
        for name in names:
            assert opt.model.getConstrByName(name) is not None


# ---------------------------------------------------------------------------
# embed_surrogate — delta (Δ-tolerance) correction
# ---------------------------------------------------------------------------


class TestDeltaCorrection:
    def test_delta_zero_adds_no_ordering_constraints(self) -> None:
        opt = _cont_opt(x_dim=2)
        qnn = _tiny_qnn(num_q=5)
        opt.embed_surrogate(qnn, model_type="qnn", delta=0.0)
        opt.model.update()
        # No delta ordering constraints added
        named = [opt.model.getConstrByName(f"delta_ordering_{k}") for k in range(4)]
        assert all(c is None for c in named)

    def test_positive_delta_adds_ordering_constraints(self) -> None:
        opt = _cont_opt(x_dim=2)
        qnn = _tiny_qnn(num_q=5)
        opt.embed_surrogate(qnn, model_type="qnn", delta=10.0)
        opt.model.update()
        # Should add num_quantiles - 1 = 4 ordering constraints
        for k in range(4):
            c = opt.model.getConstrByName(f"delta_ordering_{k}")
            assert c is not None, f"delta_ordering_{k} constraint missing"

    def test_negative_delta_raises(self) -> None:
        opt = _cont_opt(x_dim=2)
        qnn = _tiny_qnn(num_q=3)
        with pytest.raises(ValueError, match="delta must be >= 0"):
            opt.embed_surrogate(qnn, model_type="qnn", delta=-1.0)

    def test_iqnn_delta_ignored(self) -> None:
        """delta is silently ignored for IQNN (already monotone by design)."""
        opt = _cont_opt(x_dim=2)
        iqnn = _tiny_iqnn(num_q=4)
        opt.embed_surrogate(iqnn, model_type="iqnn", delta=50.0)
        opt.model.update()
        # No delta ordering constraints for IQNN
        named = [opt.model.getConstrByName(f"delta_ordering_{k}") for k in range(3)]
        assert all(c is None for c in named)

    def test_unknown_model_type_raises(self) -> None:
        opt = _cont_opt(x_dim=2)
        qnn = _tiny_qnn(num_q=3)
        with pytest.raises(ValueError, match="Unknown model_type"):
            opt.embed_surrogate(qnn, model_type="bad_type")


# ---------------------------------------------------------------------------
# Objective formulations
# ---------------------------------------------------------------------------


class TestObjectiveFormulations:
    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.qnn = _tiny_qnn(input_dim=1, num_q=10)
        self.x_bounds = [(0.0, 1.0)]

    def _make_opt(self) -> SurrogateOptimizer:
        opt = SurrogateOptimizer(x_dim=1, x_bounds=self.x_bounds)
        opt.embed_surrogate(self.qnn, model_type="qnn")
        return opt

    def test_risk_neutral_returns_optimal(self) -> None:
        opt = self._make_opt()
        opt.set_risk_neutral_objective(c=[1.0])
        result = opt.optimize()
        assert result.is_optimal
        assert result.obj_val is not None
        assert len(result.x_opt) == 1

    def test_risk_averse_alpha_zero_matches_neutral(self) -> None:
        """CVaR with alpha=0 (tail = all quantiles) should equal risk-neutral."""
        opt_neutral = self._make_opt()
        opt_neutral.set_risk_neutral_objective(c=[0.0])
        result_neutral = opt_neutral.optimize()

        opt_cvar = self._make_opt()
        opt_cvar.set_risk_averse_objective(c=[0.0], alpha=0.0)
        result_cvar = opt_cvar.optimize()

        assert result_cvar.is_optimal
        assert result_cvar.obj_val is not None
        assert abs(result_cvar.obj_val - (result_neutral.obj_val or 0.0)) < 1e-5

    def test_risk_averse_high_alpha(self) -> None:
        opt = self._make_opt()
        opt.set_risk_averse_objective(c=[0.0], alpha=0.9)
        result = opt.optimize()
        assert result.is_optimal

    def test_risk_averse_alpha_too_high_raises(self) -> None:
        opt = self._make_opt()
        with pytest.raises(ValueError, match="too high"):
            opt.set_risk_averse_objective(c=[0.0], alpha=1.0)

    def test_mean_risk_lam_zero_matches_neutral(self) -> None:
        opt_neutral = self._make_opt()
        opt_neutral.set_risk_neutral_objective(c=[0.0])
        result_neutral = opt_neutral.optimize()

        opt_mr = self._make_opt()
        opt_mr.set_mean_risk_objective(c=[0.0], alpha=0.8, lam=0.0)
        result_mr = opt_mr.optimize()

        assert result_mr.is_optimal
        assert abs((result_mr.obj_val or 0.0) - (result_neutral.obj_val or 0.0)) < 1e-5

    def test_mean_risk_nonzero_lam_valid(self) -> None:
        opt = self._make_opt()
        opt.set_mean_risk_objective(c=[0.0], alpha=0.5, lam=5.0)
        result = opt.optimize()
        assert result.is_optimal
        assert result.obj_val is not None

    def test_mean_risk_negative_lam_raises(self) -> None:
        opt = self._make_opt()
        with pytest.raises(ValueError, match="lam must be >= 0"):
            opt.set_mean_risk_objective(c=[0.0], alpha=0.5, lam=-1.0)

    def test_mean_risk_alpha_too_high_raises(self) -> None:
        opt = self._make_opt()
        with pytest.raises(ValueError, match="too high"):
            opt.set_mean_risk_objective(c=[0.0], alpha=1.0, lam=1.0)

    def test_objective_before_embed_raises(self) -> None:
        opt = SurrogateOptimizer(x_dim=1, x_bounds=[(0.0, 1.0)])
        with pytest.raises(RuntimeError, match="No surrogate model embedded"):
            opt.set_risk_neutral_objective(c=[1.0])

    def test_cost_vector_wrong_length_raises(self) -> None:
        opt = self._make_opt()
        with pytest.raises(ValueError, match="Cost vector"):
            opt.set_risk_neutral_objective(c=[1.0, 2.0])


# ---------------------------------------------------------------------------
# Solver parameter controls
# ---------------------------------------------------------------------------


class TestSolverParameters:
    def test_set_time_limit_does_not_raise(self) -> None:
        opt = _cont_opt()
        opt.set_time_limit(10.0)
        assert opt.model.Params.TimeLimit == 10.0

    def test_set_mip_gap_does_not_raise(self) -> None:
        opt = _cont_opt()
        opt.set_mip_gap(0.01)
        assert abs(opt.model.Params.MIPGap - 0.01) < 1e-9

    def test_set_threads_does_not_raise(self) -> None:
        opt = _cont_opt()
        opt.set_threads(4)
        assert opt.model.Params.Threads == 4


# ---------------------------------------------------------------------------
# optimize() returns SolveResult
# ---------------------------------------------------------------------------


class TestOptimizeReturnsResult:
    def test_solve_result_has_timing(self) -> None:
        opt = _cont_opt(x_dim=2)
        qnn = _tiny_qnn(num_q=5)
        opt.embed_surrogate(qnn, model_type="qnn")
        opt.set_risk_neutral_objective(c=[0.0, 0.0])
        result = opt.optimize()
        assert result.solve_time_s >= 0.0

    def test_solve_result_quantile_vals_populated(self) -> None:
        opt = _cont_opt(x_dim=2)
        qnn = _tiny_qnn(num_q=5)
        opt.embed_surrogate(qnn, model_type="qnn")
        opt.set_risk_neutral_objective(c=[0.0, 0.0])
        result = opt.optimize()
        assert len(result.quantile_vals) == 5

    def test_infeasible_returns_non_optimal_result(self) -> None:
        opt = SurrogateOptimizer(x_dim=1, x_bounds=[(0.0, 1.0)])
        qnn = _tiny_qnn(input_dim=1, num_q=3)
        opt.embed_surrogate(qnn, model_type="qnn")
        opt.set_risk_neutral_objective(c=[0.0])
        # Make infeasible: x >= 2 but x <= 1
        opt.add_linear_constraint([1.0], ConstrSense.GEQ, 2.0, name="infeasible")
        result = opt.optimize()
        assert not result.is_optimal
        assert result.obj_val is None
        assert result.x_opt == []

    def test_iqnn_solve_result(self) -> None:
        opt = _cont_opt(x_dim=2)
        iqnn = _tiny_iqnn(num_q=6)
        opt.embed_surrogate(iqnn, model_type="iqnn")
        opt.set_risk_neutral_objective(c=[0.0, 0.0])
        result = opt.optimize()
        assert result.is_optimal
        assert len(result.quantile_vals) == 6


# ---------------------------------------------------------------------------
# End-to-end: binary CFLP-style problem
# ---------------------------------------------------------------------------


class TestCFLPStyleProblem:
    """
    Simulates a tiny Capacitated Facility Location Problem (CFLP) structure:
    binary first-stage open/close facility decisions with a budget constraint,
    and a QNN surrogate for second-stage cost.
    """

    def test_cflp_binary_qnn_solves(self) -> None:
        n_facilities = 4
        opt = SurrogateOptimizer(
            x_dim=n_facilities,
            x_bounds=[(0.0, 1.0)] * n_facilities,
            var_types=[VarType.BINARY] * n_facilities,
        )
        # Budget: open at most 2 facilities
        opt.add_linear_constraint(
            [1.0] * n_facilities, ConstrSense.LEQ, 2.0, name="budget"
        )
        qnn = _tiny_qnn(input_dim=n_facilities, num_q=5)
        opt.embed_surrogate(qnn, model_type="qnn")
        # Fixed operating cost per facility + expected second-stage
        opt.set_risk_neutral_objective(c=[1.0] * n_facilities)
        result = opt.optimize()

        assert result.is_optimal
        assert result.obj_val is not None
        total_open = sum(round(v) for v in result.x_opt)
        assert total_open <= 2

    def test_cflp_binary_iqnn_solves(self) -> None:
        n_facilities = 4
        opt = SurrogateOptimizer(
            x_dim=n_facilities,
            x_bounds=[(0.0, 1.0)] * n_facilities,
            var_types=[VarType.BINARY] * n_facilities,
        )
        opt.add_linear_constraint(
            [1.0] * n_facilities, ConstrSense.LEQ, 2.0, name="budget"
        )
        iqnn = _tiny_iqnn(input_dim=n_facilities, num_q=5)
        opt.embed_surrogate(iqnn, model_type="iqnn")
        opt.set_mean_risk_objective(c=[1.0] * n_facilities, alpha=0.8, lam=2.0)
        result = opt.optimize()

        assert result.is_optimal
        assert sum(round(v) for v in result.x_opt) <= 2

    def test_cflp_iqnn_quantiles_monotone_at_optimum(self) -> None:
        """IQNN quantiles at the MILP optimum must be non-decreasing."""
        n_facilities = 3
        opt = SurrogateOptimizer(
            x_dim=n_facilities,
            x_bounds=[(0.0, 1.0)] * n_facilities,
            var_types=[VarType.BINARY] * n_facilities,
        )
        opt.add_linear_constraint([1.0] * n_facilities, ConstrSense.LEQ, 2.0)
        iqnn = _tiny_iqnn(input_dim=n_facilities, num_q=6)
        opt.embed_surrogate(iqnn, model_type="iqnn")
        opt.set_risk_neutral_objective(c=[0.0] * n_facilities)
        result = opt.optimize()

        assert result.is_optimal
        q_vals = result.quantile_vals
        for i in range(len(q_vals) - 1):
            assert q_vals[i] <= q_vals[i + 1] + 1e-6, (
                f"Monotonicity violated: q[{i}]={q_vals[i]} "
                f"> q[{i + 1}]={q_vals[i + 1]}"
            )


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------


def test_optimization_package_exports_phase5() -> None:
    """All Phase 5 types are accessible via the optimization package."""
    from qnn_stoch_opt.optimization import (  # noqa: F401
        ConstrSense,
        QNNtoMILP,
        SolveResult,
        SurrogateOptimizer,
        VarType,
    )


def test_root_package_exports_phase5() -> None:
    """All Phase 5 types are accessible via the root package."""
    import qnn_stoch_opt as pkg

    for name in ("VarType", "ConstrSense", "SolveResult"):
        assert hasattr(pkg, name), f"qnn_stoch_opt missing export: {name!r}"


def test_root_package_all_exports_intact() -> None:
    """The existing Phase 1–4 exports are still present after Phase 5."""
    import qnn_stoch_opt as pkg

    expected = [
        "StochasticOptimizationDataset",
        "create_dataloaders",
        "SecondStageEvaluator",
        "generate_normal_scenarios",
        "generate_uniform_scenarios",
        "QuantileNeuralNetwork",
        "IncrementalQuantileNeuralNetwork",
        "pinball_loss",
        "train_model",
        "QNNtoMILP",
        "SurrogateOptimizer",
        "VarType",
        "ConstrSense",
        "SolveResult",
    ]
    for name in expected:
        assert hasattr(pkg, name), f"qnn_stoch_opt missing export: {name!r}"
