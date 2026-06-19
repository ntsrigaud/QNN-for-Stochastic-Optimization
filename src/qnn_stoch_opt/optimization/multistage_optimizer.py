"""
Multi-stage MILP surrogate optimizer.

Builds and solves a single MILP that embeds one QNN/IQNN surrogate per
stage transition, as produced by :class:`qnn_stoch_opt.models.cascade.CascadeQNN`.
Generalizes :class:`qnn_stoch_opt.optimization.stochastic_optimizer.SurrogateOptimizer`
from a single first-stage decision to a sequence of per-stage decisions, all
solved jointly in one ``gp.Model``.

Known modeling approximation: each stage's surrogate is trained (via
backward induction) to anticipate a good decision in the *next* stage, so
its quantiles already partially capture the continuation value. Embedding
every stage's surrogate quantiles into the same joint objective therefore
has some overlap between consecutive stages' contributions, rather than
being a perfectly non-double-counting Bellman decomposition. This is the
price of a fixed-size, tractable joint MILP instead of exact multi-stage
dynamic programming -- true-objective evaluation against SAA baselines is
how this approximation should be judged in practice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
import torch.nn as nn
from gurobipy import GRB

from qnn_stoch_opt.optimization.milp_formulation import QNNtoMILP
from qnn_stoch_opt.optimization.stochastic_optimizer import ConstrSense, VarType


@dataclass
class MultiStageSolveResult:
    """Structured result returned by :meth:`MultiStageSurrogateOptimizer.optimize`."""

    status: int
    x_opt: List[List[float]] = field(default_factory=list)
    obj_val: Optional[float] = None
    solve_time_s: float = 0.0
    mip_gap: float = 0.0
    quantile_vals: List[List[float]] = field(default_factory=list)

    @property
    def is_optimal(self) -> bool:
        return bool(self.status == GRB.OPTIMAL)


class MultiStageSurrogateOptimizer:
    """Formulates and solves a joint MILP over all stages' decision variables.

    Example usage::

        opt = MultiStageSurrogateOptimizer(
            stage_dims=[n, n], var_types=[[VarType.BINARY] * n] * 2
        )
        opt.add_stage_constraint(0, [1.0] * n, ConstrSense.GEQ, 1.0, name="min_open")
        cascade.embed_all_stages(opt)
        opt.set_objective(stage_costs=[f_0.tolist(), f_1.tolist()])
        result = opt.optimize()
    """

    def __init__(
        self,
        stage_dims: List[int],
        var_types: Optional[List[List[VarType]]] = None,
        output_flag: int = 0,
    ):
        """
        Args:
            stage_dims: Dimension of each stage's own decision vector.
            var_types: Per-stage list of :class:`VarType`. Continuous by
                default.
            output_flag: Gurobi ``OutputFlag`` parameter (0 = silent).
        """
        self.stage_dims = stage_dims
        self.n_stages_decisions = len(stage_dims)
        self._var_types = var_types or [
            [VarType.CONTINUOUS] * dim for dim in stage_dims
        ]
        if len(self._var_types) != len(stage_dims):
            raise ValueError(
                f"var_types must have {len(stage_dims)} entries, "
                f"got {len(self._var_types)}."
            )

        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", output_flag)
        self.env.start()
        self.model = gp.Model("multistage_surrogate_optimizer", env=self.env)

        self.x_vars: List[List[gp.Var]] = []
        self.x_bounds: List[List[Tuple[float, float]]] = []
        for t, dim in enumerate(stage_dims):
            bounds = [(0.0, 1.0) if vt == VarType.BINARY else (0.0, GRB.INFINITY)
                      for vt in self._var_types[t]]
            stage_vars = []
            for i in range(dim):
                lb, ub = bounds[i]
                vt = self._gurobi_vtype(self._var_types[t][i])
                stage_vars.append(
                    self.model.addVar(lb=lb, ub=ub, vtype=vt, name=f"x{t}_{i}")
                )
            self.x_vars.append(stage_vars)
            self.x_bounds.append(bounds)

        self.embedder = QNNtoMILP(self.model)
        self.quantile_vars: List[List[gp.Var]] = [[] for _ in stage_dims]
        self._objective_set = False

    @staticmethod
    def _gurobi_vtype(vt: VarType) -> str:
        mapping: Dict[VarType, str] = {
            VarType.CONTINUOUS: GRB.CONTINUOUS,
            VarType.INTEGER: GRB.INTEGER,
            VarType.BINARY: GRB.BINARY,
        }
        return mapping[vt]

    def add_stage_constraint(
        self,
        stage_idx: int,
        coeffs: Sequence[float],
        sense: ConstrSense,
        rhs: float,
        name: str = "",
    ) -> gp.Constr:
        """Add a linear constraint scoped to one stage's decision vector."""
        stage_vars = self.x_vars[stage_idx]
        if len(coeffs) != len(stage_vars):
            raise ValueError(
                f"coeffs length {len(coeffs)} must equal stage_dims[{stage_idx}]"
                f"={len(stage_vars)}."
            )
        expr = gp.LinExpr()
        for i, c in enumerate(coeffs):
            expr.add(stage_vars[i], float(c))

        if sense == ConstrSense.LEQ:
            return self.model.addConstr(expr <= rhs, name=name)
        elif sense == ConstrSense.EQ:
            return self.model.addConstr(expr == rhs, name=name)
        else:  # GEQ
            return self.model.addConstr(expr >= rhs, name=name)

    def embed_stage_surrogate(
        self, stage_idx: int, torch_model: nn.Module, model_type: str = "iqnn"
    ) -> None:
        """Embed one stage's surrogate against that stage's decision variables."""
        if model_type not in ("qnn", "iqnn"):
            raise ValueError(
                f"Unknown model_type: {model_type!r}. Must be 'qnn' or 'iqnn'."
            )

        x_vars = self.x_vars[stage_idx]
        x_bounds = self.x_bounds[stage_idx]
        if model_type == "qnn":
            self.quantile_vars[stage_idx] = self.embedder.embed_qnn(
                torch_model, x_vars, x_bounds
            )
        else:
            self.quantile_vars[stage_idx] = self.embedder.embed_iqnn(
                torch_model, x_vars, x_bounds
            )

    def _check_all_stages_embedded(self) -> None:
        if any(not qv for qv in self.quantile_vars):
            raise RuntimeError(
                "Not all stages have an embedded surrogate. Call "
                "embed_stage_surrogate() (or CascadeQNN.embed_all_stages()) for "
                "every stage first."
            )

    def _stage_quantile_expr(
        self, stage_idx: int, mode: str, alpha: float, lam: float
    ) -> gp.LinExpr:
        q_vars = self.quantile_vars[stage_idx]
        num_q = len(q_vars)
        mean_expr = gp.LinExpr()
        for q in q_vars:
            mean_expr.add(q, 1.0 / num_q)

        if mode == "risk_neutral":
            return mean_expr

        tail_start = int(alpha * num_q)
        tail_vars = q_vars[tail_start:]
        if not tail_vars:
            raise ValueError(
                f"alpha={alpha} leaves no quantiles in the CVaR tail "
                f"(num_quantiles={num_q})."
            )
        cvar_expr = gp.LinExpr()
        coeff = lam / len(tail_vars)
        for q in tail_vars:
            cvar_expr.add(q, coeff)

        if mode == "risk_averse":
            return cvar_expr
        return mean_expr + cvar_expr  # mean_risk

    def set_objective(
        self,
        stage_costs: List[List[float]],
        mode: str = "risk_neutral",
        alpha: float = 0.9,
        lam: float = 1.0,
    ) -> None:
        """Set the joint objective: sum over stages of (own linear cost +
        objective_mode(that stage's surrogate quantiles)).

        Args:
            stage_costs: ``stage_costs[t]`` is the linear cost vector for
                stage t's own decision variables.
            mode: ``"risk_neutral"``, ``"risk_averse"``, or ``"mean_risk"``,
                applied independently to each stage's quantiles.
            alpha: CVaR confidence level (used when mode != "risk_neutral").
            lam: CVaR weight (used when mode == "mean_risk").
        """
        self._check_all_stages_embedded()
        if mode not in ("risk_neutral", "risk_averse", "mean_risk"):
            raise ValueError(f"Unknown mode: {mode!r}.")
        if len(stage_costs) != self.n_stages_decisions:
            raise ValueError(
                f"stage_costs must have {self.n_stages_decisions} entries, "
                f"got {len(stage_costs)}."
            )

        total = gp.LinExpr()
        for t, costs in enumerate(stage_costs):
            stage_vars = self.x_vars[t]
            if len(costs) != len(stage_vars):
                raise ValueError(
                    f"stage_costs[{t}] length {len(costs)} must equal "
                    f"stage_dims[{t}]={len(stage_vars)}."
                )
            for i, c in enumerate(costs):
                total.add(stage_vars[i], float(c))
            total += self._stage_quantile_expr(t, mode, alpha, lam)

        self.model.setObjective(total, GRB.MINIMIZE)
        self._objective_set = True

    def optimize(self) -> MultiStageSolveResult:
        """Solve the embedded multi-stage MILP."""
        t0 = time.perf_counter()
        self.model.optimize()
        wall_time = time.perf_counter() - t0

        status = self.model.status

        if status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and self.model.SolCount > 0:
            x_opt = [[float(v.X) for v in stage_vars] for stage_vars in self.x_vars]
            q_vals = [[float(v.X) for v in qv] for qv in self.quantile_vars]
            obj_val = float(self.model.ObjVal)
            mip_gap = float(self.model.MIPGap) if self.model.IsMIP else 0.0
            return MultiStageSolveResult(
                status=status,
                x_opt=x_opt,
                obj_val=obj_val,
                solve_time_s=wall_time,
                mip_gap=mip_gap,
                quantile_vals=q_vals,
            )

        return MultiStageSolveResult(status=status, solve_time_s=wall_time)
