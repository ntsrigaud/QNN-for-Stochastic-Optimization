"""
Phase 5 – Optimization Problem Formulation
==========================================

Provides :class:`SurrogateOptimizer`, which assembles the complete first-stage
MILP problem by:

1. Defining continuous **and integer** first-stage decision variables.
2. Accepting arbitrary linear first-stage constraints (A x ≤ b, A x = b,
   A x ≥ b).
3. Embedding a trained QNN or IQNN surrogate model as MILP constraints.
4. Formulating risk-neutral, risk-averse (CVaR), or mean-risk objectives.
5. Optionally applying the tolerance-based Δ correction from Algorithm 2 of
   the paper (Section 3.3) to manage quantile-crossing in QNN outputs.
6. Exposing solver parameter controls (time limit, MIP gap, verbosity) and
   returning a rich :class:`SolveResult` dataclass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
import numpy as np
import torch.nn as nn
from gurobipy import GRB

from qnn_stoch_opt.optimization.milp_formulation import QNNtoMILP

# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


class VarType(str, Enum):
    """First-stage variable type."""

    CONTINUOUS = "C"
    INTEGER = "I"
    BINARY = "B"


class ConstrSense(str, Enum):
    """Sense of a linear constraint."""

    LEQ = "<="
    EQ = "=="
    GEQ = ">="


@dataclass
class SolveResult:
    """Structured result returned by :meth:`SurrogateOptimizer.optimize`.

    Attributes:
        status: Gurobi status code (GRB.OPTIMAL = 2, etc.).
        x_opt: Optimal first-stage decisions (empty list when infeasible).
        obj_val: Optimal objective value (``None`` when infeasible).
        solve_time_s: Wall-clock time spent inside the Gurobi solver (seconds).
        mip_gap: Relative MIP optimality gap (0 for LP / continuous models).
        quantile_vals: Predicted quantile values at optimum (empty when
            infeasible).
    """

    status: int
    x_opt: List[float] = field(default_factory=list)
    obj_val: Optional[float] = None
    solve_time_s: float = 0.0
    mip_gap: float = 0.0
    quantile_vals: List[float] = field(default_factory=list)

    @property
    def is_optimal(self) -> bool:
        """``True`` when the solver found a proven optimal solution."""
        return bool(self.status == GRB.OPTIMAL)


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------


class SurrogateOptimizer:
    """Formulates and solves the full first-stage MILP surrogate problem.

    This class implements the complete optimization problem formulation
    described in Section 3.3 of the paper:

    .. code-block:: text

        min  c^T x + objective_mode(quantile_vars)
        s.t. first-stage linear constraints
             MILP encoding of QNN/IQNN surrogate
             variable bounds and integrality

    Supports three objective modes (set before calling :meth:`optimize`):

    * **Risk-neutral**:  ``min c^T x + E[V(x,ξ)]``
    * **Risk-averse**:   ``min c^T x + CVaR_α[V(x,ξ)]``
    * **Mean-risk**:     ``min c^T x + E[V(x,ξ)] + λ·CVaR_α[V(x,ξ)]``

    Example usage::

        opt = SurrogateOptimizer(
            x_dim=5,
            x_bounds=[(0, 1)] * 5,
            var_types=[VarType.BINARY] * 5,
        )
        # add a capacity constraint: sum(x) <= 3
        opt.add_linear_constraint(
            coeffs=[1.0] * 5, sense=ConstrSense.LEQ, rhs=3.0, name="cap"
        )
        opt.embed_surrogate(trained_qnn, model_type="qnn")
        opt.set_risk_neutral_objective(c=[1.0] * 5)
        result = opt.optimize()
    """

    def __init__(
        self,
        x_dim: int,
        x_bounds: List[Tuple[float, float]],
        var_types: Optional[List[VarType]] = None,
        output_flag: int = 0,
    ):
        """
        Args:
            x_dim: Number of first-stage decision variables.
            x_bounds: List of ``(lb, ub)`` pairs, one per variable.
            var_types: Variable types (continuous by default).  Pass a list of
                :class:`VarType` values — one per variable — to declare integer
                or binary variables.
            output_flag: Gurobi ``OutputFlag`` parameter (0 = silent).
        """
        if len(x_bounds) != x_dim:
            raise ValueError(
                f"x_bounds length {len(x_bounds)} must equal x_dim={x_dim}."
            )
        if var_types is not None and len(var_types) != x_dim:
            raise ValueError(
                f"var_types length {len(var_types)} must equal x_dim={x_dim}."
            )

        self.x_dim = x_dim
        self.x_bounds = x_bounds
        self._var_types = var_types or [VarType.CONTINUOUS] * x_dim

        # Build the Gurobi environment and model
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", output_flag)
        self.env.start()
        self.model = gp.Model("surrogate_optimizer", env=self.env)

        # Create first-stage variables
        self.x_vars: List[gp.Var] = []
        for i in range(x_dim):
            lb, ub = x_bounds[i]
            vt = self._gurobi_vtype(self._var_types[i])
            var = self.model.addVar(lb=lb, ub=ub, vtype=vt, name=f"x_{i}")
            self.x_vars.append(var)

        self.embedder = QNNtoMILP(self.model)
        self.quantiles_vars: List[gp.Var] = []
        self._objective_set: bool = False

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gurobi_vtype(vt: VarType) -> str:
        mapping: Dict[VarType, str] = {
            VarType.CONTINUOUS: GRB.CONTINUOUS,
            VarType.INTEGER: GRB.INTEGER,
            VarType.BINARY: GRB.BINARY,
        }
        return mapping[vt]

    # ------------------------------------------------------------------
    # First-stage constraint API
    # ------------------------------------------------------------------

    def add_linear_constraint(
        self,
        coeffs: Sequence[float],
        sense: ConstrSense,
        rhs: float,
        name: str = "",
    ) -> gp.Constr:
        """Add a single linear constraint on the first-stage variables.

        Encodes ``coeffs · x  {<=, ==, >=}  rhs``.

        Args:
            coeffs: Coefficient vector of length ``x_dim``.
            sense: Constraint sense (:class:`ConstrSense`).
            rhs: Right-hand side scalar.
            name: Optional constraint name for Gurobi.

        Returns:
            gp.Constr: The Gurobi constraint object.

        Raises:
            ValueError: If ``len(coeffs) != x_dim``.
        """
        if len(coeffs) != self.x_dim:
            raise ValueError(
                f"coeffs length {len(coeffs)} must equal x_dim={self.x_dim}."
            )
        expr = gp.LinExpr()
        for i, c in enumerate(coeffs):
            expr.add(self.x_vars[i], float(c))

        if sense == ConstrSense.LEQ:
            return self.model.addConstr(expr <= rhs, name=name)
        elif sense == ConstrSense.EQ:
            return self.model.addConstr(expr == rhs, name=name)
        else:  # GEQ
            return self.model.addConstr(expr >= rhs, name=name)

    def add_constraints_matrix(
        self,
        A: np.ndarray,
        sense: ConstrSense,
        b: np.ndarray,
        names: Optional[List[str]] = None,
    ) -> List[gp.Constr]:
        """Add multiple linear constraints from a matrix A and vector b.

        Encodes ``A x  {<=, ==, >=}  b`` row by row.

        Args:
            A: Constraint matrix of shape ``(m, x_dim)``.
            sense: Constraint sense applied to every row.
            b: Right-hand side vector of length ``m``.
            names: Optional list of ``m`` constraint names.

        Returns:
            List[gp.Constr]: The added Gurobi constraint objects.

        Raises:
            ValueError: If ``A`` has incompatible dimensions.
        """
        if A.ndim != 2 or A.shape[1] != self.x_dim:
            raise ValueError(
                f"A must have shape (m, x_dim={self.x_dim}), got {A.shape}."
            )
        if b.ndim != 1 or len(b) != A.shape[0]:
            raise ValueError(f"b must have shape ({A.shape[0]},), got {b.shape}.")
        constrs = []
        for i in range(A.shape[0]):
            cname = names[i] if names is not None else f"constr_{i}"
            c = self.add_linear_constraint(
                A[i].tolist(), sense, float(b[i]), name=cname
            )
            constrs.append(c)
        return constrs

    # ------------------------------------------------------------------
    # Surrogate embedding
    # ------------------------------------------------------------------

    def embed_surrogate(
        self,
        torch_model: nn.Module,
        model_type: str = "qnn",
        delta: float = 0.0,
    ) -> None:
        """Embed the surrogate neural network into the Gurobi model.

        Args:
            torch_model: The trained PyTorch model (QNN or IQNN).
            model_type: Either ``"qnn"`` or ``"iqnn"``.
            delta: Tolerance parameter Δ ≥ 0 for QNN quantile-crossing
                correction (Algorithm 2, Section 3.3).  Ignored for IQNN.
                A positive Δ relaxes the quantile ordering by adding
                ``q_{k} + Δ ≤ q_{k+1}`` constraints, effectively filtering
                out crossings at the MILP level.

        Raises:
            ValueError: If ``model_type`` is unknown or ``delta < 0``.
        """
        if model_type not in ("qnn", "iqnn"):
            raise ValueError(
                f"Unknown model_type: {model_type!r}. Must be 'qnn' or 'iqnn'."
            )
        if delta < 0:
            raise ValueError(f"delta must be >= 0, got {delta}.")

        if model_type == "qnn":
            self.quantiles_vars = self.embedder.embed_qnn(
                torch_model, self.x_vars, self.x_bounds
            )
            if delta > 0.0:
                self._apply_delta_correction(delta)
        else:  # iqnn
            self.quantiles_vars = self.embedder.embed_iqnn(
                torch_model, self.x_vars, self.x_bounds
            )

    def _apply_delta_correction(self, delta: float) -> None:
        """Add ordering constraints q_{k} + delta ≤ q_{k+1} for QNN outputs.

        This implements the tolerance-based Δ correction from Algorithm 2 of
        the paper (Section 3.3) which prevents quantile crossing at the
        optimisation level.

        Args:
            delta: Tolerance Δ ≥ 0.
        """
        for k in range(len(self.quantiles_vars) - 1):
            self.model.addConstr(
                self.quantiles_vars[k] + delta <= self.quantiles_vars[k + 1],
                name=f"delta_ordering_{k}",
            )

    # ------------------------------------------------------------------
    # Objective formulations
    # ------------------------------------------------------------------

    def set_risk_neutral_objective(self, c: List[float]) -> None:
        """Set a risk-neutral objective: ``min c^T x + E[V(x, ξ)]``.

        The expected second-stage cost E[V] is approximated as the mean of
        all predicted quantiles (Section 3.3, risk-neutral formulation).

        Args:
            c: Linear cost coefficients for the first-stage variables.
        """
        self._check_quantiles_embedded()
        num_quantiles = len(self.quantiles_vars)
        expected_second_stage = gp.LinExpr()
        for q_var in self.quantiles_vars:
            expected_second_stage.add(q_var, 1.0 / num_quantiles)

        first_stage_cost = self._build_first_stage_cost(c)
        self.model.setObjective(first_stage_cost + expected_second_stage, GRB.MINIMIZE)
        self._objective_set = True

    def set_risk_averse_objective(self, c: List[float], alpha: float) -> None:
        """Set a purely risk-averse CVaR objective.

        Minimises ``c^T x + CVaR_α[V(x, ξ)]``.  CVaR at level α is
        approximated as the mean of the predicted quantiles at or above
        the α-tail (Section 3.3, risk-averse formulation).

        Args:
            c: Linear cost coefficients for the first-stage variables.
            alpha: Confidence level in [0, 1).  Higher values focus on
                a smaller, worse-case tail.

        Raises:
            ValueError: If ``alpha`` leaves no quantiles in the tail.
        """
        self._check_quantiles_embedded()
        cvar_expr = self._build_cvar_expr(alpha)
        first_stage_cost = self._build_first_stage_cost(c)
        self.model.setObjective(first_stage_cost + cvar_expr, GRB.MINIMIZE)
        self._objective_set = True

    def set_mean_risk_objective(self, c: List[float], alpha: float, lam: float) -> None:
        """Set a mean-risk objective combining expected cost and CVaR.

        Minimises ``c^T x + E[V(x, ξ)] + λ·CVaR_α[V(x, ξ)]`` — the
        mean-risk formulation from Section 2.1 of the paper.

        Args:
            c: Linear cost coefficients for the first-stage variables.
            alpha: CVaR confidence level in [0, 1).
            lam: Non-negative risk-aversion weight λ ≥ 0.  When ``lam=0``,
                reduces to risk-neutral; as ``lam`` increases, the solution
                becomes more risk-averse.

        Raises:
            ValueError: If ``lam < 0`` or if ``alpha`` leaves no tail quantiles.
        """
        self._check_quantiles_embedded()
        if lam < 0:
            raise ValueError(f"Risk-aversion weight lam must be >= 0, got {lam}.")

        num_quantiles = len(self.quantiles_vars)
        expected_second_stage = gp.LinExpr()
        for q_var in self.quantiles_vars:
            expected_second_stage.add(q_var, 1.0 / num_quantiles)

        cvar_expr = self._build_cvar_expr(alpha, weight=lam)
        first_stage_cost = self._build_first_stage_cost(c)
        self.model.setObjective(
            first_stage_cost + expected_second_stage + cvar_expr, GRB.MINIMIZE
        )
        self._objective_set = True

    # ------------------------------------------------------------------
    # Solver configuration
    # ------------------------------------------------------------------

    def set_time_limit(self, seconds: float) -> None:
        """Set a wall-clock time limit for the solver.

        Args:
            seconds: Maximum solver time in seconds.
        """
        self.model.setParam("TimeLimit", seconds)

    def set_mip_gap(self, gap: float) -> None:
        """Set the relative MIP optimality gap tolerance.

        The solver stops when the gap between the incumbent and the lower
        bound falls below ``gap``.

        Args:
            gap: Relative MIP gap in [0, 1).
        """
        self.model.setParam("MIPGap", gap)

    def set_threads(self, threads: int) -> None:
        """Set the number of threads used by Gurobi.

        Args:
            threads: Number of solver threads (0 = auto).
        """
        self.model.setParam("Threads", threads)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def optimize(self) -> SolveResult:
        """Solve the embedded MILP problem.

        Returns:
            :class:`SolveResult` with optimal decisions, objective, timing,
            MIP gap, and quantile values at the optimum.

        Note:
            Unlike a plain ``RuntimeError`` on infeasibility, this method
            always returns a :class:`SolveResult` — callers should check
            :attr:`SolveResult.is_optimal`.  An infeasible or time-limited
            result will have ``obj_val=None`` and an empty ``x_opt``.
        """
        t0 = time.perf_counter()
        self.model.optimize()
        wall_time = time.perf_counter() - t0

        status = self.model.status

        if status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and self.model.SolCount > 0:
            x_opt = [float(v.X) for v in self.x_vars]
            obj_val = float(self.model.ObjVal)
            q_vals = [float(v.X) for v in self.quantiles_vars]
            mip_gap = float(self.model.MIPGap) if self.model.IsMIP else 0.0
            return SolveResult(
                status=status,
                x_opt=x_opt,
                obj_val=obj_val,
                solve_time_s=wall_time,
                mip_gap=mip_gap,
                quantile_vals=q_vals,
            )

        return SolveResult(status=status, solve_time_s=wall_time)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_first_stage_cost(self, c: List[float]) -> gp.LinExpr:
        """Build c^T x as a Gurobi linear expression."""
        if len(c) != self.x_dim:
            raise ValueError(
                f"Cost vector c has length {len(c)}, expected {self.x_dim}."
            )
        expr = gp.LinExpr()
        for i, var in enumerate(self.x_vars):
            expr.add(var, float(c[i]))
        return expr

    def _build_cvar_expr(self, alpha: float, weight: float = 1.0) -> gp.LinExpr:
        """Build the CVaR tail mean expression.

        Args:
            alpha: Confidence level in [0, 1).
            weight: Scalar multiplier (λ for mean-risk; 1 for pure CVaR).

        Raises:
            ValueError: If ``alpha`` leaves no quantiles in the tail.
        """
        num_quantiles = len(self.quantiles_vars)
        tail_start_idx = int(alpha * num_quantiles)
        tail_vars = self.quantiles_vars[tail_start_idx:]

        if not tail_vars:
            raise ValueError(
                f"Alpha={alpha} is too high: no quantiles remain in the CVaR tail "
                f"(num_quantiles={num_quantiles})."
            )

        expr = gp.LinExpr()
        coeff = weight / len(tail_vars)
        for q_var in tail_vars:
            expr.add(q_var, coeff)
        return expr

    def _check_quantiles_embedded(self) -> None:
        """Raise if no surrogate has been embedded yet."""
        if not self.quantiles_vars:
            raise RuntimeError(
                "No surrogate model embedded. Call embed_surrogate() first."
            )
