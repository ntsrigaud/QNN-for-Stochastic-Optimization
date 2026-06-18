import argparse
import time
from typing import Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from qnn_stoch_opt.case_studies import (
    CFLPEvaluator,
    generate_cflp_demand_scenarios,
    generate_cflp_instance,
)
from qnn_stoch_opt.models.iqnn import IQNN
from qnn_stoch_opt.models.qnn import QuantileNeuralNetwork
from qnn_stoch_opt.models.trainer import train_model
from qnn_stoch_opt.optimization import SurrogateOptimizer, VarType


def create_dataset(
    n: int, m: int, num_samples: int, evaluator: CFLPEvaluator, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training dataset.

    Args:
        n: Number of facilities.
        m: Number of customers.
        num_samples: Number of samples to generate.
        evaluator: CFLPEvaluator instance to compute objective.
        rng: Random number generator.

    Returns:
        Tuple of feature matrix X and target vector y as float32 ndarrays.
    """
    # Duplicate untyped create_dataset removed
    X_data: list[np.ndarray] = []
    y_data: list[float] = []
    for _ in range(num_samples):
        while True:
            x_i = rng.integers(0, 2, size=n)
            if x_i.sum() > 0:
                break
        scenario_i = generate_cflp_demand_scenarios(
            m, 1, seed=int(rng.integers(0, 1_000_000))
        )[0]
        v_i = evaluator.evaluate(x_i, scenario_i)
        if v_i != float("inf"):
            X_data.append(x_i)
            y_data.append(v_i)
    return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.float32)


def run_benchmark_risk_averse(
    num_train: int = 10000, alpha: float = 0.9, lambda_weight: float = 1.0
) -> None:
    print(
        f"--- Benchmark: Risk-Averse (CVaR, a={alpha}, l={lambda_weight}) "
        "CFLP-10-10 ---"
    )
    n, m = 10, 10
    rng = np.random.default_rng(42)
    f_costs, assignment_costs, capacities = generate_cflp_instance(n, m, seed=42)
    evaluator = CFLPEvaluator(n, m, capacities, assignment_costs)

    print("Generating training data...")
    X_data, y_data = create_dataset(n, m, num_train, evaluator, rng)

    quantiles = torch.linspace(0.01, 0.99, 50)

    val_size = int(len(X_data) * 0.2)
    train_size = len(X_data) - val_size

    train_dataset = TensorDataset(
        torch.tensor(X_data[:train_size]),
        torch.tensor(y_data[:train_size]).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_data[-val_size:]), torch.tensor(y_data[-val_size:]).unsqueeze(1)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print("Training QNN...")
    qnn = QuantileNeuralNetwork(input_dim=n, hidden_dims=[64, 64], num_quantiles=50)
    train_model(
        qnn, train_loader, val_loader, quantiles, epochs=100, lr=1e-3, patience=10
    )

    print("Training IQNN...")
    iqnn = IQNN(input_dim=n, hidden_dims=[64, 64], num_quantiles=50)
    train_model(
        iqnn, train_loader, val_loader, quantiles, epochs=100, lr=1e-3, patience=10
    )

    x_bounds = [(0.0, 1.0)] * n
    x_vtypes = [VarType.BINARY] * n

    def run_surrogate(
        model: Any, model_type: str, delta: float = 100.0
    ) -> Tuple[Any, float]:
        opt = SurrogateOptimizer(x_dim=n, x_bounds=x_bounds, x_vtypes=x_vtypes)
        opt.embed_surrogate(model, model_type=model_type, delta_crossing=delta)
        # Note: We configure the CVaR objective here
        opt.set_cvar_objective(c=f_costs, alpha=alpha, lambda_weight=lambda_weight)

        start_t = time.time()
        res = opt.optimize()
        opt_time = time.time() - start_t

        return res, opt_time

    print("\nSolving Surrogate - QNN...")
    res_qnn, t_qnn = run_surrogate(qnn, "qnn")
    print(
        f"QNN Solution Time: {t_qnn:.2f}s | Obj: {res_qnn.obj_val:.2f} "
        f"| Gap: {res_qnn.mip_gap:.2%} | Opt: {res_qnn.x_opt}"
    )

    print("\nSolving Surrogate - IQNN...")
    res_iqnn, t_iqnn = run_surrogate(iqnn, "iqnn")
    print(
        f"IQNN Solution Time: {t_iqnn:.2f}s | Obj: {res_iqnn.obj_val:.2f} "
        f"| Gap: {res_iqnn.mip_gap:.2%} | Opt: {res_iqnn.x_opt}"
    )

    # True evaluation: For risk averse, the true objective is Mean(Costs) +
    # Lambda * CVaR(Costs)
    test_scenarios = generate_cflp_demand_scenarios(m, num_scenarios=500, seed=999)

    def evaluate_true_obj(x_opt: Any) -> float:
        if not x_opt or sum(x_opt) == 0:
            return float("inf")
        costs = evaluator.evaluate_scenarios(np.array(x_opt), test_scenarios)
        valid_costs = costs[costs != float("inf")]
        if len(valid_costs) == 0:
            return float("inf")

        first_stage = np.sum(f_costs * x_opt)

        # Calculate CVaR at alpha over the scenarios
        var_alpha = np.quantile(valid_costs, alpha)
        tail = valid_costs[valid_costs >= var_alpha]
        cvar = np.mean(tail) if len(tail) > 0 else var_alpha

        # Combined objective: Expectation + Lambda * CVaR
        # Actually in the paper, risk averse obj is sometimes (1-lambda)*E + lambda*CVaR
        # Let's assume standard E + lambda * CVaR
        return float(first_stage + np.mean(valid_costs) + lambda_weight * cvar)

    true_qnn = evaluate_true_obj(res_qnn.x_opt)
    true_iqnn = evaluate_true_obj(res_iqnn.x_opt)

    print("\n--- Final Evaluation (500 Scenarios) ---")
    print(f"QNN True Obj: {true_qnn:.2f}")
    print(f"IQNN True Obj: {true_iqnn:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-samples", type=int, default=10000, help="Number of training samples"
    )
    parser.add_argument("--alpha", type=float, default=0.9, help="CVaR alpha quantile")
    parser.add_argument(
        "--lambda-weight", type=float, default=1.0, help="Weight for CVaR"
    )
    args = parser.parse_args()
    run_benchmark_risk_averse(args.train_samples, args.alpha, args.lambda_weight)
