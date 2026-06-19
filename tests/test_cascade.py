"""
Tests for the 3-stage cascade extension: CascadeQNN, MultiStageSurrogateOptimizer,
ThreeStageCFLPEvaluator, and ConditionalDemandModel.
"""

import numpy as np
import torch

from qnn_stoch_opt.case_studies.cflp_multistage import (
    ThreeStageCFLPEvaluator,
    generate_3stage_cflp_instance,
)
from qnn_stoch_opt.endogenous import ConditionalDemandModel
from qnn_stoch_opt.models.cascade import CascadeQNN
from qnn_stoch_opt.models.qnn import QuantileNeuralNetwork
from qnn_stoch_opt.optimization.multistage_optimizer import MultiStageSurrogateOptimizer
from qnn_stoch_opt.optimization.stochastic_optimizer import ConstrSense, VarType

N = 5
M = 5
NUM_Q = 7
QUANTILES = torch.linspace(0.1, 0.9, NUM_Q)


def _three_stage_evaluator(seed: int = 1) -> ThreeStageCFLPEvaluator:
    f_0, f_1, costs, base_cap, exp_cap = generate_3stage_cflp_instance(N, M, seed=seed)
    return ThreeStageCFLPEvaluator(N, M, f_0, f_1, costs, base_cap, exp_cap)


def _random_xy(seed: int, n_samples: int, dim: int):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_samples, dim)).astype(np.float32)
    v = rng.uniform(50, 200, size=n_samples).astype(np.float32)
    return X, v


class TestCascadeQNNTraining:
    def test_train_backward_returns_finite_losses(self) -> None:
        cascade = CascadeQNN(
            n_stages=3, input_dims=[N, N], hidden_dims=[8], num_quantiles=NUM_Q,
            model_type="iqnn",
        )
        stage0_data = _random_xy(0, 40, N)
        stage1_data = _random_xy(1, 40, N)
        losses = cascade.train_backward(
            [stage0_data, stage1_data], QUANTILES, epochs=5, lr=1e-2, patience=3
        )
        assert len(losses) == 2
        assert all(np.isfinite(loss) for loss in losses)

    def test_two_stage_cascade_matches_single_stage_qnn_structure(self) -> None:
        # n_stages=2 -> a single surrogate, equivalent to training one QNN directly.
        cascade = CascadeQNN(
            n_stages=2, input_dims=[N], hidden_dims=[8], num_quantiles=NUM_Q,
            model_type="qnn",
        )
        assert len(cascade.surrogates) == 1
        assert isinstance(cascade.surrogates[0], QuantileNeuralNetwork)

        X, v = _random_xy(2, 40, N)
        losses = cascade.train_backward(
            [(X, v)], QUANTILES, epochs=5, lr=1e-2, patience=3
        )
        assert len(losses) == 1
        assert np.isfinite(losses[0])

        surrogate = cascade.surrogates[0]
        surrogate.eval()
        with torch.no_grad():
            preds = surrogate(torch.tensor(X[:3]))
        assert preds.shape == (3, NUM_Q)


class TestMultiStageSurrogateOptimizer:
    def test_returns_feasible_binary_solution(self) -> None:
        cascade = CascadeQNN(
            n_stages=3, input_dims=[N, N], hidden_dims=[8], num_quantiles=NUM_Q,
            model_type="iqnn",
        )
        stage0_data = _random_xy(10, 40, N)
        stage1_data = _random_xy(11, 40, N)
        cascade.train_backward(
            [stage0_data, stage1_data], QUANTILES, epochs=5, lr=1e-2, patience=3
        )

        opt = MultiStageSurrogateOptimizer(
            stage_dims=[N, N],
            var_types=[[VarType.BINARY] * N, [VarType.BINARY] * N],
        )
        opt.add_stage_constraint(0, [1.0] * N, ConstrSense.GEQ, 1.0, name="min_open")
        cascade.embed_all_stages(opt)

        f_0, f_1, _, _, _ = generate_3stage_cflp_instance(N, M, seed=3)
        opt.set_objective(stage_costs=[f_0.tolist(), f_1.tolist()])
        result = opt.optimize()

        assert result.is_optimal
        assert sum(result.x_opt[0]) >= 1


class TestThreeStageCFLPEvaluator:
    def test_evaluate_stage2_returns_finite_cost(self) -> None:
        # Generously sized capacities relative to demand so the assignment
        # LP is feasible regardless of which subset of facilities is open --
        # the point of this test is "does evaluate_stage2 run and return a
        # finite cost", not bin-packing feasibility of a random instance.
        _, _, costs, _, _ = generate_3stage_cflp_instance(N, M, seed=2)
        f_0 = np.full(N, 100.0)
        f_1 = np.full(N, 50.0)
        base_cap = np.full(N, 1000.0)
        exp_cap = np.full(N, 1500.0)
        evaluator = ThreeStageCFLPEvaluator(N, M, f_0, f_1, costs, base_cap, exp_cap)

        rng = np.random.default_rng(5)
        x_0 = np.ones(N, dtype=np.float32)
        x_1 = (rng.random(N) < 0.5).astype(np.float32)
        demand_2 = evaluator.sample_demand_2(x_1, M, seed=6)
        cost = evaluator.evaluate_stage2(x_0, x_1, demand_2)
        assert np.isfinite(cost)


class TestConditionalDemandModel:
    def test_higher_sum_x1_yields_higher_mean_demand(self) -> None:
        evaluator = _three_stage_evaluator()
        rng = np.random.default_rng(8)

        X1_train = rng.integers(0, 2, size=(200, N)).astype(np.float32)
        demand2_train = np.array(
            [
                evaluator.sample_demand_2(x1, M, seed=int(i))
                for i, x1 in enumerate(X1_train)
            ]
        )

        model = ConditionalDemandModel()
        model.fit(X1_train, demand2_train)

        low_x1 = np.zeros(N, dtype=np.float32)
        high_x1 = np.ones(N, dtype=np.float32)

        low_samples = model.sample(low_x1, n_samples=500, seed=1)
        high_samples = model.sample(high_x1, n_samples=500, seed=2)

        assert high_samples.mean() > low_samples.mean()
