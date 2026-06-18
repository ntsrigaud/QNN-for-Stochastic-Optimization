import numpy as np
from qnn_stoch_opt.data.saa_solver import SecondStageEvaluator


def test_second_stage_evaluator_single() -> None:
    # Problem: min Y s.t. -Y <= h_xi - T_xi * X, Y >= 0
    # With X=3, h_xi=-2, T_xi=1: -Y <= -2 - 3 = -5  => Y >= 5
    # The optimal cost should be 5.
    q = np.array([1.0])
    W = np.array([[-1.0]])

    evaluator = SecondStageEvaluator(q=q, W=W)

    X = np.array([3.0])
    h_xi = np.array([-2.0])
    T_xi = np.array([[1.0]])

    cost = evaluator.evaluate(X, h_xi, T_xi)
    np.testing.assert_allclose(cost, 5.0)


def test_second_stage_evaluator_scenarios() -> None:
    q = np.array([1.0])
    W = np.array([[-1.0]])
    evaluator = SecondStageEvaluator(q=q, W=W)

    X = np.array([3.0])
    h_scenarios = np.array([[-2.0], [-1.0], [0.0]])
    T_scenarios = np.array([[[1.0]], [[1.0]], [[1.0]]])

    # Expected Costs:
    # Scenario 1 (-2.0): Y >= 2 + 3 = 5
    # Scenario 2 (-1.0): Y >= 1 + 3 = 4
    # Scenario 3 ( 0.0): Y >= 0 + 3 = 3
    costs = evaluator.evaluate_scenarios(X, h_scenarios, T_scenarios)

    assert costs.shape == (3,)
    np.testing.assert_allclose(costs, np.array([5.0, 4.0, 3.0]))
