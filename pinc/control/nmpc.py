import numpy as np
import torch
from scipy.optimize import minimize


class NMPCController:
    """
    Nonlinear MPC controller following the Multiple-Shooting formulation
    of Eq. (12) in the paper, built on top of any "control interface"
    function f_hat_w(y, u) -> y_next that predicts one T-second-ahead
    state transition (Eq. 8). In this codebase that function is either:

      - model.step               (the trained PINC net), or
      - an RK4-based one-step predictor (the "ODE/RK" baseline),

    both exposing the same signature, so the exact same NMPC code can
    drive either one (as done for Table 1 / Fig. 10 in the paper).

    Decision variables: the control increments du[0..Nu-1] (Eq. 12c/12d).
    Cost (Eq. 12a):
        J = sum_{j=N1}^{N2} ||y[k+j] - y_ref[k+j]||^2_Q
          + sum_{i=0}^{Nu-1} ||du[k+i]||^2_R

    Constraints: box bounds on u (h in Eq. 12e, simplified to bounds).
    """

    def __init__(self, control_interface, control_dim,
                 N1, N2, Nu, Q, R,
                 u_min=None, u_max=None,
                 state_constraints=None):
        """
        state_constraints : optional list of (state_index, low, high)
            tuples, enforced at every step of the prediction horizon
            (Eq. 12e), e.g. [(2, 0.6, 5.5), (3, 0.6, 5.5)] to constrain
            h3, h4 of the four-tank system as in Sec. 4.2.2.
        """
        self.f = control_interface
        self.control_dim = control_dim
        self.N1 = N1
        self.N2 = N2
        self.Nu = Nu
        self.Q = Q
        self.R = R
        self.u_min = u_min
        self.u_max = u_max
        self.state_constraints = state_constraints or []

    def _rollout(self, y0, u_prev, du_flat):
        """Rolls the predictive model forward N2 steps given control
        increments, returning the predicted state trajectory."""
        du = du_flat.reshape(self.Nu, self.control_dim)

        u = u_prev.clone()
        y = y0.clone()

        ys = []
        for j in range(self.N2):
            i = min(j, self.Nu - 1)
            if j < self.Nu:
                u = u + du[i]
            y = self.f(y.unsqueeze(0), u.unsqueeze(0))[0]
            ys.append(y)

        return torch.stack(ys), u

    def _cost_and_grad(self, du_np, y0, u_prev, y_ref_horizon):
        du = torch.tensor(du_np, dtype=torch.float32, requires_grad=True)

        ys, _ = self._rollout(y0, u_prev, du)

        # tracking cost, only over the penalized window [N1, N2]
        track = 0.0
        for j in range(self.N1 - 1, self.N2):
            err = ys[j] - y_ref_horizon[j]
            track = track + err @ self.Q @ err

        du_mat = du.reshape(self.Nu, self.control_dim)
        reg = 0.0
        for i in range(self.Nu):
            reg = reg + du_mat[i] @ self.R @ du_mat[i]

        cost = track + reg
        cost.backward()

        grad = du.grad.detach().numpy().astype(np.float64)
        return cost.item(), grad

    def _state_constraint_funcs(self, y0, u_prev):
        """
        Builds a single vectorized scipy-style constraint dict (with an
        autograd-computed Jacobian) enforcing low <= y[k+j][idx] <= high
        for every step j and every configured (idx, low, high) box,
        implementing Eq. (12e) as one stacked vector of inequalities
        (two rows per bound: value - low >= 0 and high - value >= 0).

        All bounds are evaluated from a *single* rollout per call (rather
        than one rollout per bound), which matters a lot for runtime
        since the rollout involves N2 forward passes through the
        predictive model.
        """
        if not self.state_constraints:
            return []

        def all_vals(du):
            ys, _ = self._rollout(y0, u_prev, du)
            rows = []
            for (idx, low, high) in self.state_constraints:
                rows.append(ys[:, idx] - low)
                rows.append(high - ys[:, idx])
            return torch.cat(rows)

        def fun(du_np):
            du = torch.tensor(du_np, dtype=torch.float32)
            return all_vals(du).detach().numpy().astype(np.float64)

        def jac(du_np):
            du = torch.tensor(du_np, dtype=torch.float32)
            j = torch.autograd.functional.jacobian(all_vals, du, vectorize=True)
            return j.numpy().astype(np.float64)

        return [{"type": "ineq", "fun": fun, "jac": jac}]

    def solve(self, y_current, u_prev, y_ref_horizon):
        """
        y_current     : (state_dim,) true/measured current state, y_hat[k-1]
        u_prev        : (control_dim,) previously applied control, u[k-1]
        y_ref_horizon : (N2, state_dim) reference trajectory for steps 1..N2

        returns u_apply : (control_dim,) the first control action u[k]
                to be applied to the plant (receding horizon).
        """
        n_dec = self.Nu * self.control_dim
        du0 = np.zeros(n_dec)

        bounds = None
        if self.u_min is not None and self.u_max is not None:
            # bounds are on the *absolute* control action, approximated
            # here as bounds on cumulative increments around u_prev
            lo = (np.array(self.u_min) - u_prev.numpy())
            hi = (np.array(self.u_max) - u_prev.numpy())
            bounds = [(lo[d % self.control_dim], hi[d % self.control_dim]) for d in range(n_dec)]

        constraints = self._state_constraint_funcs(y_current, u_prev)

        res = minimize(
            self._cost_and_grad,
            du0,
            args=(y_current, u_prev, y_ref_horizon),
            jac=True,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 30, "ftol": 1e-6},
        )

        du_opt = torch.tensor(res.x, dtype=torch.float32).reshape(self.Nu, self.control_dim)
        u_apply = u_prev + du_opt[0]

        if self.u_min is not None:
            u_apply = torch.clamp(u_apply,
                                   torch.tensor(self.u_min, dtype=torch.float32),
                                   torch.tensor(self.u_max, dtype=torch.float32))

        return u_apply.detach()


def run_nmpc_simulation(control_interface, plant_step, y0, y_ref_full,
                         control_dim, N1, N2, Nu, Q, R,
                         u_min=None, u_max=None, state_constraints=None):
    """
    Implements Algorithm 2: for each timestep k, solve the NMPC problem
    using `control_interface` as predictive model, apply the resulting
    u[k] to `plant_step` (the real process, e.g. RK4 ground truth), and
    record the resulting closed-loop trajectory.

    y_ref_full : (C, state_dim) reference for the whole simulation; the
                 controller looks N2 steps ahead into this array (padded
                 at the end by repeating the last value).
    """
    controller = NMPCController(control_interface, control_dim, N1, N2, Nu, Q, R,
                                 u_min=u_min, u_max=u_max,
                                 state_constraints=state_constraints)

    C = y_ref_full.shape[0]
    y = y0.clone()
    u = torch.zeros(control_dim)

    y_hist = [y.clone()]
    u_hist = []

    padded_ref = torch.cat([y_ref_full, y_ref_full[-1:].repeat(N2, 1)], dim=0)

    for k in range(C):
        y_ref_horizon = padded_ref[k + 1: k + 1 + N2]
        u = controller.solve(y, u, y_ref_horizon)
        y = plant_step(y.unsqueeze(0), u.unsqueeze(0))[0]

        y_hist.append(y.clone())
        u_hist.append(u.clone())

    return torch.stack(y_hist), torch.stack(u_hist)
