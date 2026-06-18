import numpy as np
from qnn_stoch_opt.data.scenario_generation import (
    generate_normal_scenarios,
    generate_uniform_scenarios,
)


def test_generate_uniform_scenarios() -> None:
    low = np.array([0.0, -1.0])
    high = np.array([1.0, 1.0])
    scenarios = generate_uniform_scenarios(low, high, num_scenarios=50)

    assert scenarios.shape == (50, 2)
    assert np.all(scenarios[:, 0] >= 0.0)
    assert np.all(scenarios[:, 0] <= 1.0)
    assert np.all(scenarios[:, 1] >= -1.0)
    assert np.all(scenarios[:, 1] <= 1.0)


def test_generate_normal_scenarios() -> None:
    mean = np.array([5.0, 10.0])
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    scenarios = generate_normal_scenarios(mean, cov, num_scenarios=500)

    assert scenarios.shape == (500, 2)
    # The mean of a large sample should be close to the true mean
    sample_mean = np.mean(scenarios, axis=0)
    np.testing.assert_allclose(sample_mean, mean, rtol=0.1, atol=0.5)
