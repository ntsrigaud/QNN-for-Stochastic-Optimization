from typing import List, Tuple

import gurobipy as gp
import numpy as np
import torch.nn as nn
from gurobipy import GRB


class QNNtoMILP:
    """
    Translates trained PyTorch QNN/IQNN surrogate models into exact Mixed-Integer
    Linear Programming (MILP) constraints within Gurobi.
    """

    def __init__(self, gurobi_model: gp.Model):
        self.model = gurobi_model

    def _propagate_linear_bounds(
        self,
        weights: np.ndarray,
        biases: np.ndarray,
        lower_in: np.ndarray,
        upper_in: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute interval bounds for a linear layer: y = Wx + b
        """
        W_plus = np.maximum(weights, 0)
        W_minus = np.minimum(weights, 0)

        lower_out = W_plus @ lower_in + W_minus @ upper_in + biases
        upper_out = W_plus @ upper_in + W_minus @ lower_in + biases
        return lower_out, upper_out

    def _add_relu_constraints(
        self,
        pre_vars: List[gp.Var],
        post_vars: List[gp.Var],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        layer_idx: int,
    ) -> None:
        """
        Embeds Big-M constraints exactly modeling z = max(0, x).
        """
        for i, (x, z, lb, ub) in enumerate(
            zip(pre_vars, post_vars, lower_bounds, upper_bounds)
        ):
            # If strictly positive, it's just z = x
            if lb >= 0:
                self.model.addConstr(z == x, name=f"relu_pos_l{layer_idx}_{i}")
            # If strictly negative, it's just z = 0
            elif ub <= 0:
                self.model.addConstr(z == 0, name=f"relu_neg_l{layer_idx}_{i}")
            # Otherwise, use Big-M formulation
            else:
                sigma = self.model.addVar(
                    vtype=GRB.BINARY, name=f"sigma_l{layer_idx}_{i}"
                )
                # z >= x
                self.model.addConstr(z >= x, name=f"relu_1_l{layer_idx}_{i}")
                # z >= 0
                self.model.addConstr(z >= 0, name=f"relu_2_l{layer_idx}_{i}")
                # z <= x - M-(1 - sigma)  where M- is lb
                self.model.addConstr(
                    z <= x - lb * (1 - sigma), name=f"relu_3_l{layer_idx}_{i}"
                )
                # z <= M+ sigma where M+ is ub
                self.model.addConstr(z <= ub * sigma, name=f"relu_4_l{layer_idx}_{i}")

    def embed_sequential(
        self,
        torch_sequential: nn.Sequential,
        x_vars: List[gp.Var],
        x_bounds: List[Tuple[float, float]],
    ) -> Tuple[List[gp.Var], List[Tuple[float, float]]]:
        """
        Embeds a standard nn.Sequential (Linear -> ReLU -> Linear) network
        into the model.
        """
        current_vars = x_vars
        current_lb = np.array([b[0] for b in x_bounds])
        current_ub = np.array([b[1] for b in x_bounds])

        layer_idx = 0
        for layer in torch_sequential:
            if isinstance(layer, nn.Linear):
                W = layer.weight.detach().numpy()
                b = layer.bias.detach().numpy()
                num_out = W.shape[0]

                # Propagate bounds
                next_lb, next_ub = self._propagate_linear_bounds(
                    W, b, current_lb, current_ub
                )

                # Create Gurobi variables for the pre-activation output
                next_vars = []
                for i in range(num_out):
                    var = self.model.addVar(
                        lb=next_lb[i],
                        ub=next_ub[i],
                        vtype=GRB.CONTINUOUS,
                        name=f"linear_l{layer_idx}_{i}",
                    )
                    next_vars.append(var)

                    # Add linear constraint: y = Wx + b
                    expr = gp.LinExpr(b[i])
                    for j in range(len(current_vars)):
                        expr.add(current_vars[j], W[i, j])
                    self.model.addConstr(
                        var == expr, name=f"lin_constr_l{layer_idx}_{i}"
                    )

                current_vars = next_vars
                current_lb = next_lb
                current_ub = next_ub

            elif isinstance(layer, nn.ReLU):
                num_out = len(current_vars)
                post_vars = []

                post_lb = np.maximum(current_lb, 0)
                post_ub = np.maximum(current_ub, 0)

                for i in range(num_out):
                    var = self.model.addVar(
                        lb=post_lb[i],
                        ub=post_ub[i],
                        vtype=GRB.CONTINUOUS,
                        name=f"relu_out_l{layer_idx}_{i}",
                    )
                    post_vars.append(var)

                self._add_relu_constraints(
                    current_vars, post_vars, current_lb, current_ub, layer_idx
                )

                current_vars = post_vars
                current_lb = post_lb
                current_ub = post_ub

            layer_idx += 1

        current_bounds = list(zip(current_lb, current_ub))
        return current_vars, current_bounds

    def embed_qnn(
        self, qnn: nn.Module, x_vars: List[gp.Var], x_bounds: List[Tuple[float, float]]
    ) -> List[gp.Var]:
        """
        Embeds a standard QuantileNeuralNetwork.
        """
        assert isinstance(qnn.network, nn.Sequential), (
            f"Expected qnn.network to be nn.Sequential, got {type(qnn.network)}"
        )
        out_vars, _ = self.embed_sequential(qnn.network, x_vars, x_bounds)
        return out_vars

    def embed_iqnn(
        self, iqnn: nn.Module, x_vars: List[gp.Var], x_bounds: List[Tuple[float, float]]
    ) -> List[gp.Var]:
        """
        Embeds an IncrementalQuantileNeuralNetwork, accounting for the final
        ReLU increment and cumulative sum mechanism.
        """
        assert isinstance(
            iqnn.hidden_layers, nn.Sequential
        ), f"""Expected iqnn.hidden_layers to be nn.Sequential, 
                got {type(iqnn.hidden_layers)}"""
        h_vars, h_bounds = self.embed_sequential(iqnn.hidden_layers, x_vars, x_bounds)

        # Process output layer (Linear)
        assert isinstance(iqnn.output_layer, nn.Linear), (
            f"Expected iqnn.output_layer to be nn.Linear, got {type(iqnn.output_layer)}"
        )
        W = iqnn.output_layer.weight.detach().numpy()
        b = iqnn.output_layer.bias.detach().numpy()
        num_out = W.shape[0]

        h_lb = np.array([bnd[0] for bnd in h_bounds])
        h_ub = np.array([bnd[1] for bnd in h_bounds])
        out_lb, out_ub = self._propagate_linear_bounds(W, b, h_lb, h_ub)

        out_vars = []
        for i in range(num_out):
            var = self.model.addVar(
                lb=out_lb[i], ub=out_ub[i], vtype=GRB.CONTINUOUS, name=f"iqnn_out_{i}"
            )
            out_vars.append(var)
            expr = gp.LinExpr(b[i])
            for j in range(len(h_vars)):
                expr.add(h_vars[j], W[i, j])
            self.model.addConstr(var == expr, name=f"iqnn_lin_constr_{i}")

        # The first variable is the base (unconstrained)
        base_var = out_vars[0]

        # The remaining are increments requiring ReLU
        inc_vars_pre = out_vars[1:]
        inc_lb = out_lb[1:]
        inc_ub = out_ub[1:]

        inc_vars_post = []
        for i in range(len(inc_vars_pre)):
            var = self.model.addVar(
                lb=max(0, inc_lb[i]),
                ub=max(0, inc_ub[i]),
                vtype=GRB.CONTINUOUS,
                name=f"iqnn_inc_relu_{i}",
            )
            inc_vars_post.append(var)

        self._add_relu_constraints(
            inc_vars_pre, inc_vars_post, inc_lb, inc_ub, layer_idx=999
        )

        # Apply cumulative sum
        final_quantiles = []
        current_sum = base_var
        final_quantiles.append(current_sum)

        for i in range(len(inc_vars_post)):
            cum_var = self.model.addVar(
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name=f"iqnn_cum_{i + 1}",
            )
            self.model.addConstr(
                cum_var == current_sum + inc_vars_post[i],
                name=f"iqnn_cumsum_constr_{i + 1}",
            )
            current_sum = cum_var
            final_quantiles.append(current_sum)

        return final_quantiles
