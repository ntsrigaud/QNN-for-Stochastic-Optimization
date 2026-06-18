import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from qnn_stoch_opt.case_studies import (
    CFLPEvaluator,
    generate_cflp_demand_scenarios,
    generate_cflp_instance,
)
from qnn_stoch_opt.models.qnn import QuantileNeuralNetwork
from qnn_stoch_opt.models.trainer import train_model


def run_dataset_impact_experiment(max_samples: int = 10000) -> None:
    print(f"Running Dataset Impact Experiment up to {max_samples} samples...")
    # 1. Problem Setup: CFLP-10-10
    n, m = 10, 10
    f_costs, assignment_costs, capacities = generate_cflp_instance(n, m, seed=42)
    evaluator = CFLPEvaluator(n, m, capacities, assignment_costs)

    # 2. Data Generation (Algorithm 1: single scenario per X_i)
    rng = np.random.default_rng(42)

    print("Generating dataset...")
    X_data = []
    y_data = []

    start_gen = time.time()
    for _ in range(max_samples):
        # Random feasible first-stage decision (just random binary vector,
        # ensuring at least 1 facility is open to avoid guaranteed infeasibility)
        while True:
            x_i = rng.integers(0, 2, size=n)
            if x_i.sum() > 0:
                break

        scenario_i = generate_cflp_demand_scenarios(
            m, 1, seed=rng.integers(0, 1000000)
        )[0]
        v_i = evaluator.evaluate(x_i, scenario_i)

        if v_i != float("inf"):
            X_data.append(x_i)
            y_data.append(v_i)

    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)
    print(f"Generated {len(X_data)} feasible samples in {time.time() - start_gen:.2f}s")

    quantiles = torch.linspace(0.01, 0.99, 50)

    # Validation set uses 20% of the max available
    val_size = int(len(X_data) * 0.2)

    X_val = torch.tensor(X_data[-val_size:])
    y_val = torch.tensor(y_data[-val_size:]).unsqueeze(1)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    train_sizes = [500, 1000, 5000, len(X_data) - val_size]

    for size in train_sizes:
        if size > len(X_data) - val_size:
            continue

        print(f"\n--- Training with {size} samples ---")
        X_train = torch.tensor(X_data[:size])
        y_train = torch.tensor(y_data[:size]).unsqueeze(1)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = QuantileNeuralNetwork(
            input_dim=n, hidden_dims=[64, 64], num_quantiles=50
        )

        start_train = time.time()
        _, best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            quantiles=quantiles,
            epochs=200,
            lr=1e-3,
            patience=10,
        )
        print(f"Training Time: {time.time() - start_train:.2f}s")
        print(f"Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", type=int, default=5000, help="Max training samples to generate"
    )
    args = parser.parse_args()
    run_dataset_impact_experiment(args.samples)
