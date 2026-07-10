"""
MPC controller built on top of a trained `PINCTransientPDE` net, following
Eq. (37) and Algorithm 2 of Sec. 4.5 of arXiv:2506.06188.

Unlike the ODE-PINC's `NMPCController` (pinc/control/nmpc.py), which rolls
a *state* y forward via f(y, u) -> y_next, the PDE-PINC's forward
simulation (Sec. 4.4.3, Algorithm 1) rolls the *previous control* forward:
the network never sees a fed-back predicted state at all, only the control
applied in the previous window (u0) and the one applied in the current
window (u). So the "receding horizon" here literally means: solve for the
best u1..uNc given the true previously-applied control u0, apply u1, then
next iteration's u0 is exactly that applied u1 (Algorithm 2) -- there is
no other state feedback into the model (the paper notes this is a known
limitation, addressed only optionally via error-correction filtering,
which is not implemented here).
"""
import numpy as np
import torch
from scipy.optimize import minimize
from tqdm import tqdm


class PDEMPCController:
    """
    Decision variables: the absolute (normalized) control sequence
    u1..uNc (Eq. 37 optimizes u directly, not increments -- the
    consecutive-difference term IS the smoothness penalty already).

    Cost:
        J = sum_{i=1}^{Np} (f(x_bar, u_{i-1}, u_i) - target)^2
          + lambda * sum_{i=1}^{Nc} (u_i - u_{i-1})^2

    Constraints:
        |f(x_bar, u_i, u_{i+1}) - f(x_bar, u_{i-1}, u_i)| <= dy_max
        u_i = u_Nc for i = Nc+1, ..., Np   (implemented by construction,
                                             not as an explicit constraint)
    """

    def __init__(self, transient_model, x_bar, target, Np, Nc,
                 lambda_smooth=0.1, dy_max=None, u_min=0.0, u_max=1.0,
                 maxiter=50):
        self.model = transient_model
        self.x_bar = x_bar
        self.target = target
        self.Np = Np
        self.Nc = Nc
        self.lambda_smooth = lambda_smooth
        self.dy_max = dy_max
        self.u_min = u_min
        self.u_max = u_max
        self.maxiter = maxiter

    def _predict_pressure_seq(self, u0, u_seq):
        """
        u0    : scalar tensor, control applied in the window before this
                horizon starts
        u_seq : (Np,) tensor of control decisions covering the whole
                prediction horizon (already extended so u_seq[Nc:] is
                held at u_seq[Nc-1], matching Eq. 12d's u-hold rule)

        returns P_seq : (Np,) predicted pressure at x_bar, one entry per
                        step of the prediction horizon
        """
        u_prev = torch.cat([u0.reshape(1), u_seq[:-1]])
        n = u_seq.shape[0]
        y = self.model.at_position(self.x_bar, u_prev.reshape(n, 1), u_seq.reshape(n, 1))
        return y[:, 0]  # pressure

    def _extend(self, u_dec):
        """u_dec: (Nc,) decision vars -> (Np,) full horizon (Eq. 12d hold)."""
        if self.Np <= self.Nc:
            return u_dec[: self.Np]
        hold = u_dec[-1].expand(self.Np - self.Nc)
        return torch.cat([u_dec, hold])

    def _cost_and_grad(self, u_np, u0):
        u_dec = torch.tensor(u_np, dtype=torch.float32, requires_grad=True)
        u_seq = self._extend(u_dec)

        P_seq = self._predict_pressure_seq(u0, u_seq)
        track = torch.sum((P_seq - self.target) ** 2)

        u_prev_dec = torch.cat([u0.reshape(1), u_dec[:-1]])
        smooth = torch.sum((u_dec - u_prev_dec) ** 2)

        cost = track + self.lambda_smooth * smooth
        cost.backward()

        grad = u_dec.grad.detach().numpy().astype(np.float64)
        return cost.item(), grad

    def _rate_constraint(self, u0, y0_true):
        """
        Eq. (37)'s two pairs of constraints:
          |f(x_bar,u0,u1) - y0|        <= dy_max   (anchored to the *true*
                                                      current measurement)
          |f(x_bar,ui,ui+1) - f(x_bar,ui-1,ui)| <= dy_max   for the rest

        y0_true must be the actual current measured/plant output (NOT
        re-derived from the model), otherwise the very first -- and
        typically largest -- transition slips through unconstrained
        (model.at_position(u0, u1) was being compared to itself).
        """
        if self.dy_max is None:
            return []

        def all_vals(u_dec):
            u_seq = self._extend(u_dec)
            P_seq = self._predict_pressure_seq(u0, u_seq)
            P_full = torch.cat([y0_true.reshape(1), P_seq])
            dP = P_full[1:] - P_full[:-1]
            return torch.cat([self.dy_max - dP, self.dy_max + dP])

        def fun(u_np):
            u_dec = torch.tensor(u_np, dtype=torch.float32)
            return all_vals(u_dec).detach().numpy().astype(np.float64)

        def jac(u_np):
            u_dec = torch.tensor(u_np, dtype=torch.float32)
            j = torch.autograd.functional.jacobian(all_vals, u_dec, vectorize=True)
            return j.numpy().astype(np.float64)

        return [{"type": "ineq", "fun": fun, "jac": jac}]

    def solve(self, u0: torch.Tensor, y0_true: torch.Tensor, u_init=None):
        """
        u0      : scalar tensor, the control applied in the previous
                  window (this + the model is all the predictive rollout
                  needs -- see module docstring).
        y0_true : scalar tensor, the actual current measured output
                  (pressure at x_bar), used only to anchor the first
                  rate constraint (Eq. 37's first two inequality lines).

        returns (u_apply, u_dec_opt): u_apply is the first control action
        u1 to apply to the plant; u_dec_opt is the full Nc-length decision
        vector, useful for warm-starting the next solve.
        """
        if u_init is not None:
            x0 = u_init.numpy().astype(np.float64)
        else:
            x0 = np.full(self.Nc, float(u0), dtype=np.float64)

        bounds = [(self.u_min, self.u_max)] * self.Nc
        constraints = self._rate_constraint(u0, y0_true)

        res = minimize(
            self._cost_and_grad, x0, args=(u0,), jac=True, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": self.maxiter, "ftol": 1e-6},
        )

        u_dec_opt = torch.tensor(res.x, dtype=torch.float32)
        self.last_success = res.success
        self.last_nit = res.nit
        return u_dec_opt[0].detach(), u_dec_opt.detach()


def run_pde_mpc_simulation(transient_model, plant_step, physics, x_bar, target,
                            u0_init, n_steps, Np, Nc, lambda_smooth=0.1,
                            dy_max=None, u_min=0.0, u_max=1.0, maxiter=50,
                            warm_start=True, desc="PDE-MPC", controller=None):
    """
    Implements Algorithm 2 (Sec. 4.5.1): at every step, solve the MPC
    problem given the last *applied* control u0, apply the resulting u1
    to the true plant (the exact reduced-ODE plant from
    `pinc.simulation.pipe_flow_plant`), and record the resulting
    trajectory of applied controls and the true pressure they produce.

    plant_step : control interface for the true plant, i.e.
                 `pinc.simulation.rk4.rk4_control_interface(plant, T=1.0, ...)`
                 -- takes (V_batch, u_batch) -> V_next_batch (normalized
                 velocity), one window (tref seconds) ahead.

    controller : optional, a pre-built controller exposing the same
                 `.solve(u0, y0_true, u_init=None) -> (u_apply, u_dec_opt)`
                 interface as `PDEMPCController` -- e.g. `CasadiPDEMPC`
                 or `CasadiPDERTIController` from `nmpc_pde_casadi.py`.
                 When given, `transient_model`, `Np`, `Nc`,
                 `lambda_smooth`, `dy_max`, `u_min`, `u_max`, `maxiter`
                 are ignored in favor of however `controller` was
                 already configured; only `plant_step`, `physics`,
                 `x_bar`, `target`, `u0_init`, `n_steps`, and `desc`
                 still apply. Mirrors `run_nmpc_simulation`'s
                 `controller=` argument on the ODE-PINC side.

    returns (V_hist, P_hist, u_hist): (n_steps+1,), (n_steps+1,), (n_steps,)
    """
    if controller is None:
        controller = PDEMPCController(transient_model, x_bar, target, Np, Nc,
                                       lambda_smooth=lambda_smooth, dy_max=dy_max,
                                       u_min=u_min, u_max=u_max, maxiter=maxiter)

    u0 = torch.tensor(u0_init, dtype=torch.float32)
    # true initial velocity consistent with steady state under u0_init
    V = _steady_velocity_for_control(physics, u0_init).reshape(1)

    u_hist, V_hist, P_hist = [], [V.clone()], [physics.pressure_profile(
        torch.tensor(x_bar), V[0], u0).clone()]

    y_true = P_hist[0]
    u_dec_init = None
    pbar = tqdm(range(n_steps), desc=desc, unit="step")
    for _ in pbar:
        u_apply, u_dec_opt = controller.solve(u0, y_true, u_init=u_dec_init)

        V = plant_step(V.reshape(1, 1), u_apply.reshape(1, 1))[0]
        P = physics.pressure_profile(torch.tensor(x_bar), V[0], u_apply)
        y_true = P

        u_hist.append(u_apply.clone())
        V_hist.append(V.clone())
        P_hist.append(P.clone())

        pbar.set_postfix(u=f"{u_apply.item():.3f}", P=f"{P.item():.3f}",
                          ok=controller.last_success)

        if warm_start:
            u_dec_init = torch.cat([u_dec_opt[1:], u_dec_opt[-1:]])
        u0 = u_apply

    return (torch.stack(V_hist)[:, 0], torch.stack(P_hist), torch.stack(u_hist))


def _steady_velocity_for_control(physics, u_const, bracket=(-5.0, 5.0)):
    """
    Finds V such that dV/dt = 0 under a constant control u_const (i.e. the
    true steady-state velocity), via root-finding on the plant ODE --
    used only to initialize closed-loop MPC simulations from a physically
    consistent starting point.
    """
    from scipy.optimize import brentq

    def f(v):
        return physics.plant_dVdt(torch.tensor(v), torch.tensor(float(u_const))).item()

    v_star = brentq(f, *bracket)
    return torch.tensor(v_star)
