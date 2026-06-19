# QNN for Stochastic Optimization

[![Tests & Linting](https://github.com/ntsrigaud/QNN-for-Stochastic-Optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/ntsrigaud/QNN-for-Stochastic-Optimization/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)

An implementation of the Quantile Neural Network (QNN) framework for two-stage and multi-stage stochastic optimization under uncertainty, based on the paper:
> **A Quantile Neural Network Framework for Two-stage Stochastic Optimization**  
> *Antonio Alcántara, Carlos Ruiz, and Calvin Tsay* (arXiv:2403.11707)

Instead of relying on Sample Average Approximation (SAA) which replicates recourse problems across hundreds of scenarios and scales poorly, this framework trains a **Quantile Neural Network** to approximate the conditional recourse cost distribution. The trained model is embedded directly into the first-stage Mixed-Integer Linear Program (MILP) using a tight Big-M formulation, enabling risk-neutral (expected value) or risk-averse (Conditional Value-at-Risk, CVaR) optimization in **under 1 second**.

---

## 🌟 Key Features

- **Quantile Neural Networks (QNN & IQNN):** Models the full recourse distribution. Features the **Incremental QNN (IQNN)** which mathematically guarantees monotonic quantiles, eliminating the "quantile crossing" problem by design.
- **Cascade QNN for Multi-Stage Problems:** Supports multi-stage (e.g., three-stage) stochastic optimization with **endogenous demand/uncertainty**, where early-stage decisions impact future parameter distributions.
- **Exact MILP Embeddings:** Embeds trained neural network layers as Gurobi constraints using binary variables for ReLU activations. Includes **interval-arithmetic-based bound tightening** to compute tight Big-M bounds and reduce the solver relaxation gap.
- **Risk-Neutral & Risk-Averse (CVaR) Optimization:** Formulates expected value minimization, Conditional Value-at-Risk (CVaR) tail minimization, or weighted mean-risk combinations.
- **Rich Case Studies:** Ready-to-run configurations for the Capacitated Facility Location Problem (CFLP), Multi-Stage CFLP with endogenous demand, and the Investment Problem (IP).

---

## 📁 Repository Structure

```text
├── docs/                      # Original paper summary
├── notebooks/                 # Jupyter notebook demonstrating the complete pipeline
├── scripts/                   # Executable scripts for benchmarking and analysis
└── src/qnn_stoch_opt/         # Main package source
    ├── case_studies/          # Formulation details for CFLP, Multistage-CFLP, and IP
    ├── data/                  # Scenario generation and SAA dataset builders
    ├── models/                # PyTorch architectures (QNN, IQNN, Cascade, Trainer)
    ├── optimization/          # MILP formulation, Big-M embedding, and Gurobi solvers
    └── utils/                 # Metrics (quantile crossing rates, pinball loss)
```

---

## ⚙️ Installation

### Conda (Recommended)
You can create a conda environment with all required packages (including PyTorch and Gurobi):
```bash
conda env create -f environment.yml
conda activate qnn
```

### Pip
Or install locally in your active virtual environment:
```bash
pip install -e .
# To install development dependencies (mypy, ruff, pytest):
pip install -e .[dev]
```

*Note: A Gurobi license is required to run the optimization models. Ensure the environment variable `GRB_LICENSE_FILE` is configured correctly or Gurobi is set up locally.*

---

## 🚀 Quickstart

### 1. Interactive Notebook
Check out the Jupyter notebook for a hands-on walk-through:
```bash
jupyter notebook notebooks/qnn_stochastic_opt.ipynb
```

### 2. Python API Example
Below is a simple snippet showing how to define, embed, and optimize a risk-averse surrogate model:

```python
import numpy as np
import torch
from qnn_stoch_opt.models.iqnn import IncrementalQuantileNeuralNetwork
from qnn_stoch_opt.optimization import SurrogateOptimizer, VarType, ConstrSense

# 1. Define a trained surrogate model
input_dim = 10
num_quantiles = 50
iqnn = IncrementalQuantileNeuralNetwork(input_dim=input_dim, hidden_dims=[64, 64], num_quantiles=num_quantiles)

# 2. Build the first-stage surrogate optimizer
opt = SurrogateOptimizer(
    x_dim=input_dim,
    x_bounds=[(0.0, 1.0)] * input_dim,
    var_types=[VarType.BINARY] * input_dim
)

# 3. Add first-stage constraints (e.g., must select at least one facility)
opt.add_linear_constraint(
    coeffs=[1.0] * input_dim,
    sense=ConstrSense.GEQ,
    rhs=1.0,
    name="at_least_one"
)

# 4. Embed surrogate and configure CVaR (alpha=0.9, lambda=1.0)
opt.embed_surrogate(iqnn, model_type="iqnn")
first_stage_cost = np.random.uniform(10, 50, size=input_dim)
opt.set_mean_risk_objective(c=first_stage_cost, alpha=0.9, lambda_weight=1.0)

# 5. Solve the MILP model
result = opt.optimize()
if result.is_optimal:
    print(f"Optimal decision: {result.x_opt}")
    print(f"Surrogate Obj: {result.obj_val:.4f}")
```

### 3. Running Benchmarks
Run the included benchmark scripts to reproduce paper comparisons against SAA:
```bash
# Run risk-neutral benchmark
python scripts/benchmark_risk_neutral.py --train-samples 10000

# Run risk-averse (CVaR) benchmark
python scripts/benchmark_risk_averse.py --train-samples 10000 --alpha 0.9

# Run dataset size impact analysis
python scripts/run_dataset_impact.py --samples 5000
```

---

## 🛠️ Verification & Linting

You can run quality checks and test suites easily via the `Makefile`:

```bash
# Run all code formatting, linting, typechecks, and tests
make check

# Auto-format and resolve lint errors
make format
```
