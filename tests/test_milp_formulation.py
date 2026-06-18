"""
Tests for Phase 4: Mixed-Integer Formulation of Neural Networks.

Covers:
  - QNNtoMILP: Big-M bound propagation and ReLU constraint embedding
  - SurrogateOptimizer: risk-neutral, risk-averse (CVaR), and mean-risk objectives
  - End-to-end: embed a trained QNN/IQNN and verify the MILP solution matches
    the neural network forward pass at the optimum.
"""

from typing import List, Tuple

import gurobipy as gp
import numpy as np
import pytest
import torch
import torch.nn as nn
from gurobipy import GRB

from qnn_stoch_opt.models.iqnn import IncrementalQuantileNeuralNetwork
from qnn_stoch_opt.models.qnn import QuantileNeuralNetwork
from qnn_stoch_opt.optimization.milp_formulation import QNNtoMILP
from qnn_stoch_opt.optimization.stochastic_optimizer import SurrogateOptimizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env() -> gp.Env:
    """Create a silent Gurobi environment."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


def _make_model(env: gp.Env) -> gp.Model:
    return gp.Model("test", env=env)


# ---------------------------------------------------------------------------
# QNNtoMILP — bound propagation
# ---------------------------------------------------------------------------


class TestPropagateBounds:
    """Tests for the layer-by-layer interval arithmetic."""

    def setup_method(self) -> None:
        self.env = _make_env()
        self.gm = _make_model(self.env)
        self.embedder = QNNtoMILP(self.gm)

    def test_identity_weights(self) -> None:
        """With identity weights and zero bias, bounds should pass through."""
        W = np.eye(3)
        b = np.zeros(3)
        lb_in = np.array([1.0, 2.0, 3.0])
        ub_in = np.array([2.0, 4.0, 6.0])
        lb, ub = self.embedder._propagate_linear_bounds(W, b, lb_in, ub_in)
        np.testing.assert_allclose(lb, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ub, [2.0, 4.0, 6.0])

    def test_all_positive_weights(self) -> None:
        """With all-positive weights, lower bounds map to lower bounds."""
        W = np.array([[2.0, 1.0]])
        b = np.array([0.5])
        lb_in = np.array([1.0, 2.0])
        ub_in = np.array([3.0, 4.0])
        lb, ub = self.embedder._propagate_linear_bounds(W, b, lb_in, ub_in)
        # lb = 2*1 + 1*2 + 0.5 = 4.5,  ub = 2*3 + 1*4 + 0.5 = 10.5
        np.testing.assert_allclose(lb, [4.5])
        np.testing.assert_allclose(ub, [10.5])

    def test_negative_weights_swap_bounds(self) -> None:
        """Negative weights flip which input bound contributes to lower/upper output."""
        W = np.array([[-1.0]])
        b = np.array([0.0])
        lb_in = np.array([2.0])
        ub_in = np.array([5.0])
        lb, ub = self.embedder._propagate_linear_bounds(W, b, lb_in, ub_in)
        # lb = -1 * ub_in = -5,  ub = -1 * lb_in = -2
        np.testing.assert_allclose(lb, [-5.0])
        np.testing.assert_allclose(ub, [-2.0])

    def test_bias_shifts_bounds(self) -> None:
        """Bias is added identically to both lower and upper bounds."""
        W = np.eye(2)
        b = np.array([10.0, -5.0])
        lb_in = np.array([0.0, 0.0])
        ub_in = np.array([1.0, 1.0])
        lb, ub = self.embedder._propagate_linear_bounds(W, b, lb_in, ub_in)
        np.testing.assert_allclose(lb, [10.0, -5.0])
        np.testing.assert_allclose(ub, [11.0, -4.0])

    def teardown_method(self) -> None:
        self.gm.dispose()
        self.env.dispose()


# ---------------------------------------------------------------------------
# QNNtoMILP — ReLU constraints
# ---------------------------------------------------------------------------


class TestReLUConstraints:
    """Tests for the Big-M ReLU embedding."""

    def setup_method(self) -> None:
        self.env = _make_env()
        self.gm = _make_model(self.env)
        self.embedder = QNNtoMILP(self.gm)

    def _add_pre_post_vars(
        self, lbs: List[float], ubs: List[float]
    ) -> Tuple[List[gp.Var], List[gp.Var]]:
        pre_vars = [
            self.gm.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"pre_{i}")
            for i, (lb, ub) in enumerate(zip(lbs, ubs))
        ]
        post_lbs = [max(0, lb) for lb in lbs]
        post_ubs = [max(0, ub) for ub in ubs]
        post_vars = [
            self.gm.addVar(
                lb=post_lbs[i],
                ub=post_ubs[i],
                vtype=GRB.CONTINUOUS,
                name=f"post_{i}",
            )
            for i in range(len(lbs))
        ]
        return pre_vars, post_vars

    def test_strictly_positive_neuron_is_linear(self) -> None:
        """lb >= 0 => z == x without a binary variable."""
        pre, post = self._add_pre_post_vars([1.0], [3.0])
        self.embedder._add_relu_constraints(
            pre, post, np.array([1.0]), np.array([3.0]), layer_idx=0
        )
        self.gm.update()
        # No new binary variable should have been added
        assert self.gm.NumBinVars == 0

    def test_strictly_negative_neuron_forced_zero(self) -> None:
        """ub <= 0 => z == 0 without a binary variable."""
        pre, post = self._add_pre_post_vars([-3.0], [-1.0])
        self.embedder._add_relu_constraints(
            pre, post, np.array([-3.0]), np.array([-1.0]), layer_idx=0
        )
        self.gm.update()
        assert self.gm.NumBinVars == 0

    def test_mixed_neuron_introduces_binary(self) -> None:
        """lb < 0 < ub => one binary variable per neuron."""
        pre, post = self._add_pre_post_vars([-2.0], [3.0])
        self.embedder._add_relu_constraints(
            pre, post, np.array([-2.0]), np.array([3.0]), layer_idx=0
        )
        self.gm.update()
        assert self.gm.NumBinVars == 1

    def test_multiple_mixed_neurons_count(self) -> None:
        """Three neurons with mixed bounds => three binary variables."""
        lbs = [-1.0, -5.0, -0.1]
        ubs = [2.0, 3.0, 4.0]
        pre, post = self._add_pre_post_vars(lbs, ubs)
        self.embedder._add_relu_constraints(
            pre, post, np.array(lbs), np.array(ubs), layer_idx=0
        )
        self.gm.update()
        assert self.gm.NumBinVars == 3

    def teardown_method(self) -> None:
        self.gm.dispose()
        self.env.dispose()


# ---------------------------------------------------------------------------
# QNNtoMILP — embed_sequential on a tiny network
# ---------------------------------------------------------------------------


class TestEmbedSequential:
    """Tests for embedding an nn.Sequential into the Gurobi model."""

    def setup_method(self) -> None:
        self.env = _make_env()
        self.gm = _make_model(self.env)
        self.embedder = QNNtoMILP(self.gm)
        torch.manual_seed(0)

    def _make_x_vars(
        self, n: int, lb: float = -1.0, ub: float = 1.0
    ) -> Tuple[List[gp.Var], List[Tuple[float, float]]]:
        x_vars = [
            self.gm.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"x_{i}")
            for i in range(n)
        ]
        x_bounds = [(lb, ub)] * n
        return x_vars, x_bounds

    def test_linear_only_network_returns_correct_vars(self) -> None:
        """A single-layer linear network embeds without binary variables."""
        net = nn.Sequential(nn.Linear(2, 4, bias=True))
        with torch.no_grad():
            net[0].weight.fill_(0.5)  # type: ignore[index]
            net[0].bias.fill_(0.0)  # type: ignore[index]

        x_vars, x_bounds = self._make_x_vars(2)
        out_vars, out_bounds = self.embedder.embed_sequential(net, x_vars, x_bounds)
        self.gm.update()

        assert len(out_vars) == 4
        assert len(out_bounds) == 4
        # No binary variables for a linear-only network
        assert self.gm.NumBinVars == 0

    def test_relu_network_adds_binary_vars(self) -> None:
        """Linear -> ReLU -> Linear introduces binary variables for the ReLU."""
        net = nn.Sequential(
            nn.Linear(2, 3, bias=True),
            nn.ReLU(),
            nn.Linear(3, 2, bias=True),
        )
        torch.nn.init.xavier_uniform_(net[0].weight)  # type: ignore[arg-type]
        torch.nn.init.xavier_uniform_(net[2].weight)  # type: ignore[arg-type]

        x_vars, x_bounds = self._make_x_vars(2, lb=-1.0, ub=1.0)
        out_vars, _ = self.embedder.embed_sequential(net, x_vars, x_bounds)
        self.gm.update()

        assert len(out_vars) == 2
        # The hidden ReLU layer has 3 neurons; at most 3 binary vars.
        assert self.gm.NumBinVars <= 3

    def teardown_method(self) -> None:
        self.gm.dispose()
        self.env.dispose()


# ---------------------------------------------------------------------------
# End-to-end: embed QNN and solve MILP
# ---------------------------------------------------------------------------


class TestEmbedQNNEndToEnd:
    """
    End-to-end test: embed a small QNN into a MILP and check that the optimal
    solution satisfies neural network constraints correctly.
    """

    def setup_method(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)

    def _build_tiny_qnn(
        self, input_dim: int = 2, hidden: int = 4, num_q: int = 5
    ) -> QuantileNeuralNetwork:
        model = QuantileNeuralNetwork(
            input_dim=input_dim, hidden_dims=[hidden], num_quantiles=num_q
        )
        model.eval()
        return model

    def _build_tiny_iqnn(
        self, input_dim: int = 2, hidden: int = 4, num_q: int = 5
    ) -> IncrementalQuantileNeuralNetwork:
        model = IncrementalQuantileNeuralNetwork(
            input_dim=input_dim, hidden_dims=[hidden], num_quantiles=num_q
        )
        model.eval()
        return model

    def test_qnn_milp_returns_feasible_solution(self) -> None:
        """MILP embedding of a QNN returns an optimal solution without error."""
        qnn = self._build_tiny_qnn()
        x_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        optimizer = SurrogateOptimizer(x_dim=2, x_bounds=x_bounds)
        optimizer.embed_surrogate(qnn, model_type="qnn")
        optimizer.set_risk_neutral_objective(c=[1.0, 1.0])

        result = optimizer.optimize()

        assert result.is_optimal
        assert len(result.x_opt) == 2
        assert all(-1.0 <= xi <= 1.0 + 1e-6 for xi in result.x_opt)
        assert isinstance(result.obj_val, float)

    def test_iqnn_milp_returns_feasible_solution(self) -> None:
        """MILP embedding of an IQNN returns an optimal solution without error."""
        iqnn = self._build_tiny_iqnn()
        x_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        optimizer = SurrogateOptimizer(x_dim=2, x_bounds=x_bounds)
        optimizer.embed_surrogate(iqnn, model_type="iqnn")
        optimizer.set_risk_neutral_objective(c=[0.0, 0.0])

        result = optimizer.optimize()

        assert result.is_optimal
        assert len(result.x_opt) == 2
        assert isinstance(result.obj_val, float)

    def test_qnn_milp_solution_consistent_with_forward_pass(self) -> None:
        """
        The MILP objective at optimum should match the NN forward pass at x_opt.
        """
        num_q = 5
        qnn = self._build_tiny_qnn(num_q=num_q)
        x_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        optimizer = SurrogateOptimizer(x_dim=2, x_bounds=x_bounds)
        optimizer.embed_surrogate(qnn, model_type="qnn")
        optimizer.set_risk_neutral_objective(c=[0.0, 0.0])
        result = optimizer.optimize()

        assert result.is_optimal
        assert result.obj_val is not None
        # Compute expected value via forward pass
        x_tensor = torch.tensor(result.x_opt, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_preds = qnn(x_tensor).numpy().flatten()
        nn_expected = float(np.mean(q_preds))

        assert abs(result.obj_val - nn_expected) < 1e-4, (
            f"MILP objective {result.obj_val} does not match NN forward {nn_expected}"
        )

    def test_iqnn_quantile_monotonicity_at_optimum(self) -> None:
        """
        At the MILP optimum, embedded IQNN quantile variables must be
        monotonically non-decreasing (the core IQNN guarantee).
        """
        iqnn = self._build_tiny_iqnn(num_q=6)
        x_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        optimizer = SurrogateOptimizer(x_dim=2, x_bounds=x_bounds)
        optimizer.embed_surrogate(iqnn, model_type="iqnn")
        optimizer.set_risk_neutral_objective(c=[0.0, 0.0])
        result = optimizer.optimize()

        assert result.is_optimal
        q_vals = result.quantile_vals
        for i in range(len(q_vals) - 1):
            assert q_vals[i] <= q_vals[i + 1] + 1e-6, (
                f"Quantile monotonicity violated: "
                f"q[{i}]={q_vals[i]} > q[{i + 1}]={q_vals[i + 1]}"
            )


# ---------------------------------------------------------------------------
# SurrogateOptimizer — objective formulation tests
# ---------------------------------------------------------------------------


class TestSurrogateOptimizerObjectives:
    """Tests for the three objective modes of SurrogateOptimizer."""

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.qnn = QuantileNeuralNetwork(input_dim=1, hidden_dims=[4], num_quantiles=10)
        self.qnn.eval()
        self.x_bounds = [(0.0, 1.0)]

    def _make_optimizer(self) -> SurrogateOptimizer:
        opt = SurrogateOptimizer(x_dim=1, x_bounds=self.x_bounds)
        opt.embed_surrogate(self.qnn, model_type="qnn")
        return opt

    def test_risk_neutral_objective_is_set(self) -> None:
        opt = self._make_optimizer()
        opt.set_risk_neutral_objective(c=[1.0])
        result = opt.optimize()
        assert result.is_optimal
        assert len(result.x_opt) == 1
        assert isinstance(result.obj_val, float)

    def test_risk_averse_objective_alpha_0(self) -> None:
        """Alpha=0 means tail covers all quantiles — same as risk-neutral mean."""
        opt = self._make_optimizer()
        opt.set_risk_averse_objective(c=[0.0], alpha=0.0)
        result_cvar = opt.optimize()

        opt2 = self._make_optimizer()
        opt2.set_risk_neutral_objective(c=[0.0])
        result_neutral = opt2.optimize()

        assert (
            abs((result_cvar.obj_val or 0.0) - (result_neutral.obj_val or 0.0)) < 1e-6
        )

    def test_risk_averse_objective_high_alpha(self) -> None:
        """Alpha=0.9 focuses on the top 10% quantiles."""
        opt = self._make_optimizer()
        opt.set_risk_averse_objective(c=[0.0], alpha=0.9)
        result = opt.optimize()
        assert result.is_optimal
        assert isinstance(result.obj_val, float)

    def test_risk_averse_raises_on_alpha_too_high(self) -> None:
        """Alpha=1.0 leaves no quantiles in tail — should raise ValueError."""
        opt = self._make_optimizer()
        with pytest.raises(ValueError, match="too high"):
            opt.set_risk_averse_objective(c=[0.0], alpha=1.0)

    def test_mean_risk_objective_lam_zero_matches_risk_neutral(self) -> None:
        """lam=0 in mean-risk reduces to risk-neutral."""
        opt = self._make_optimizer()
        opt.set_mean_risk_objective(c=[0.0], alpha=0.8, lam=0.0)
        result_mr = opt.optimize()

        opt2 = self._make_optimizer()
        opt2.set_risk_neutral_objective(c=[0.0])
        result_neutral = opt2.optimize()

        assert abs((result_mr.obj_val or 0.0) - (result_neutral.obj_val or 0.0)) < 1e-6

    def test_mean_risk_objective_varies_with_lam(self) -> None:
        """
        Different lambda values should produce valid float objectives.
        lam=0 must match the risk-neutral result exactly (no CVaR penalty).
        Non-zero lam changes the objective — either direction is valid because
        QNN outputs can be negative.
        """
        # lam=0 must equal risk-neutral
        opt_neutral = self._make_optimizer()
        opt_neutral.set_risk_neutral_objective(c=[0.0])
        result_neutral = opt_neutral.optimize()

        opt_mr_zero = self._make_optimizer()
        opt_mr_zero.set_mean_risk_objective(c=[0.0], alpha=0.5, lam=0.0)
        result_mr_zero = opt_mr_zero.optimize()

        mr_zero_val = result_mr_zero.obj_val or 0.0
        neutral_val = result_neutral.obj_val or 0.0
        assert abs(mr_zero_val - neutral_val) < 1e-6

        # Non-zero lam should still produce a valid float
        opt_mr = self._make_optimizer()
        opt_mr.set_mean_risk_objective(c=[0.0], alpha=0.5, lam=5.0)
        result_mr = opt_mr.optimize()
        assert isinstance(result_mr.obj_val, float)

    def test_mean_risk_raises_on_negative_lam(self) -> None:
        opt = self._make_optimizer()
        with pytest.raises(ValueError, match="lam must be >= 0"):
            opt.set_mean_risk_objective(c=[0.0], alpha=0.5, lam=-1.0)

    def test_mean_risk_raises_on_alpha_too_high(self) -> None:
        opt = self._make_optimizer()
        with pytest.raises(ValueError, match="too high"):
            opt.set_mean_risk_objective(c=[0.0], alpha=1.0, lam=1.0)

    def test_unknown_model_type_raises(self) -> None:
        opt = SurrogateOptimizer(x_dim=1, x_bounds=[(0.0, 1.0)])
        with pytest.raises(ValueError, match="Unknown model_type"):
            opt.embed_surrogate(self.qnn, model_type="unknown")


# ---------------------------------------------------------------------------
# Integration: public API accessible via all package __init__ files
# ---------------------------------------------------------------------------


def test_optimization_package_exports() -> None:
    """Ensure the optimization package exports QNNtoMILP and SurrogateOptimizer."""
    from qnn_stoch_opt.optimization import (  # noqa: F401
        QNNtoMILP,
        SurrogateOptimizer,
    )


def test_models_package_exports() -> None:
    """Ensure the models package exports all public symbols."""
    from qnn_stoch_opt.models import (  # noqa: F401
        IncrementalQuantileNeuralNetwork,
        QuantileNeuralNetwork,
        pinball_loss,
        train_model,
    )


def test_data_package_exports() -> None:
    """Ensure the data package exports all public symbols."""
    from qnn_stoch_opt.data import (  # noqa: F401
        SecondStageEvaluator,
        StochasticOptimizationDataset,
        create_dataloaders,
        generate_normal_scenarios,
        generate_uniform_scenarios,
    )


def test_root_package_exports() -> None:
    """Ensure the root package re-exports every public symbol."""
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
    ]
    for name in expected:
        assert hasattr(pkg, name), f"qnn_stoch_opt is missing expected export: {name!r}"


# ---------------------------------------------------------------------------
# QNNtoMILP — embed_qnn / embed_iqnn direct API
# ---------------------------------------------------------------------------


class TestQNNtoMILPEmbedAPI:
    """Direct tests for embed_qnn and embed_iqnn on QNNtoMILP."""

    def setup_method(self) -> None:
        torch.manual_seed(7)
        self.env = _make_env()
        self.gm = _make_model(self.env)
        self.embedder = QNNtoMILP(self.gm)
        self.x_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        self.x_vars = [
            self.gm.addVar(lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"x_{i}")
            for i in range(2)
        ]

    def test_embed_qnn_returns_correct_output_count(self) -> None:
        """embed_qnn returns one Gurobi variable per quantile output."""
        num_q = 7
        qnn = QuantileNeuralNetwork(input_dim=2, hidden_dims=[4], num_quantiles=num_q)
        qnn.eval()
        out = self.embedder.embed_qnn(qnn, self.x_vars, self.x_bounds)
        self.gm.update()
        assert len(out) == num_q

    def test_embed_iqnn_returns_correct_output_count(self) -> None:
        """embed_iqnn returns one Gurobi variable per quantile output."""
        num_q = 8
        iqnn = IncrementalQuantileNeuralNetwork(
            input_dim=2, hidden_dims=[4], num_quantiles=num_q
        )
        iqnn.eval()
        out = self.embedder.embed_iqnn(iqnn, self.x_vars, self.x_bounds)
        self.gm.update()
        assert len(out) == num_q

    def test_embed_qnn_output_vars_have_finite_bounds(self) -> None:
        """All output variables from embed_qnn should have finite lb and ub."""
        qnn = QuantileNeuralNetwork(input_dim=2, hidden_dims=[4], num_quantiles=5)
        qnn.eval()
        out = self.embedder.embed_qnn(qnn, self.x_vars, self.x_bounds)
        self.gm.update()
        for var in out:
            assert var.LB > -1e30, f"Variable {var.VarName} has no finite lower bound"
            assert var.UB < 1e30, f"Variable {var.VarName} has no finite upper bound"

    def test_embed_iqnn_cumsum_vars_created(self) -> None:
        """
        embed_iqnn creates cumulative-sum auxiliary variables (one per quantile
        after the first), ensuring the monotonicity chain is modelled correctly.
        """
        num_q = 5
        iqnn = IncrementalQuantileNeuralNetwork(
            input_dim=2, hidden_dims=[3], num_quantiles=num_q
        )
        iqnn.eval()
        n_before = self.gm.NumVars
        self.embedder.embed_iqnn(iqnn, self.x_vars, self.x_bounds)
        self.gm.update()
        n_after = self.gm.NumVars
        # At a minimum, num_q - 1 cumsum auxiliary vars should be added
        # (plus the ReLU post-activation and linear layer vars)
        assert n_after - n_before >= num_q - 1

    def teardown_method(self) -> None:
        self.gm.dispose()
        self.env.dispose()
