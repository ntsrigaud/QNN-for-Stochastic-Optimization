import numpy as np
import scipy.stats as stats


def generate_normal_scenarios(
    mean: np.ndarray, cov: np.ndarray, num_scenarios: int, seed: int = 42
) -> np.ndarray:
    """
    Generate multivariate normal scenarios for random parameters.

    Args:
        mean: Mean vector of the distribution.
        cov: Covariance matrix of the distribution.
        num_scenarios: Number of scenarios to sample.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Sampled scenarios of shape (num_scenarios, len(mean)).
    """
    np.random.seed(seed)
    return stats.multivariate_normal.rvs(mean=mean, cov=cov, size=num_scenarios)


def generate_uniform_scenarios(
    low: np.ndarray, high: np.ndarray, num_scenarios: int, seed: int = 42
) -> np.ndarray:
    """
    Generate independent uniform scenarios for random parameters.

    Args:
        low: Lower bounds for each dimension.
        high: Upper bounds for each dimension.
        num_scenarios: Number of scenarios to sample.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Sampled scenarios of shape (num_scenarios, len(low)).
    """
    np.random.seed(seed)
    return stats.uniform.rvs(loc=low, scale=high - low, size=(num_scenarios, len(low)))
