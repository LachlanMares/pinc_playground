"""
Drop-in replacements for `PDEMPCController` (nmpc_pde.py), solving the
same problem (Eq. 37 of arXiv:2506.06188) with CasADi instead of scipy's
SLSQP. Each class exposes the same public surface:

    controller = SomePDECasadiController(transient_model, x_bar, target,
                                          Np, Nc, ...)
    u_apply, u_dec_opt = controller.solve(u0, y0_true, u_init=None)

so either can be substituted wherever `PDEMPCController` is used --
including inside `run_pde_mpc_simulation` (pass `controller=...`, see
the updated signature in `nmpc_pde.py`).

Only two architectures are provided here, not three like the ODE-PINC's
`nmpc_casadi.py`:

1. `CasadiPDEMPC` -- IPOPT, single shooting (the direct analogue of
   `CasadiSingleShootingNMPC`).

2. `CasadiPDERTIController` -- a single QP per timestep, linearizing the
   transient net's pressure prediction around a nominal control
   trajectory, in the same spirit as `CasadiRTIController`.

No multiple-shooting variant is provided, and this is a deliberate
choice rather than an omission: multiple shooting earns its keep when
decoupling a *nonlinear state* from the controls lets the solver move
each independently (Eq. 12b in the ODE-PINC case). Here there is no
nonlinear state at all -- the "state" fed forward between prediction
steps is just the previous stage's control itself (an identity map, see
`nmpc_pde.py`'s module docstring), which is already a free decision
variable in single-shooting form. Introducing separate shooting nodes
for it would add decision variables and equality constraints for a
recursion IPOPT already sees exactly as it is.
"""
import numpy as np
import torch
import casadi as ca

from pinc.control.casadi_backend_pde import pinc_transient_step_to_casadi


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return np.asarray(x)


class CasadiPDEMPC:
    """See module docstring, architecture (1)."""

    def __init__(self, transient_model, x_bar, target, Np, Nc,
                 lambda_smooth=0.1, dy_max=None, u_min=0.0, u_max=1.0,
                 ipopt_opts=None):
        self.x_bar = x_bar
        self.target = float(target)
        self.Np, self.Nc = Np, Nc
        self.lambda_smooth = lambda_smooth
        self.dy_max = dy_max
        self.u_min = u_min
        self.u_max = u_max

        step = pinc_transient_step_to_casadi(transient_model, x_bar)
        self._step_keepalive = step  # see casadi_backend.py's Callback note; harmless to keep here too

        opti = ca.Opti()
        u_dec = opti.variable(Nc)          # absolute controls u1..uNc (Eq. 37, not increments)
        u0_p = opti.parameter(1)           # control applied in the previous window
        y0_p = opti.parameter(1)           # true measured pressure at x_bar right now

        def u_at(j):
            """Control applied in prediction-horizon stage j (0-indexed),
            holding at u_dec[Nc-1] for j >= Nc (Eq. 12d's hold rule)."""
            return u_dec[j] if j < Nc else u_dec[Nc - 1]

        u_prev = u0_p
        P_seq = []
        for j in range(Np):
            u_cur = u_at(j)
            y = step(u_prev, u_cur)
            P_seq.append(y[0])
            opti.subject_to(opti.bounded(u_min, u_cur, u_max))
            u_prev = u_cur

        track = sum((P_seq[j] - self.target) ** 2 for j in range(Np))
        smooth = 0
        u_prev_dec = u0_p
        for i in range(Nc):
            smooth = smooth + (u_dec[i] - u_prev_dec) ** 2
            u_prev_dec = u_dec[i]
        opti.minimize(track + lambda_smooth * smooth)

        if dy_max is not None:
            # Eq. (37)'s rate constraints, anchored on the *true* current
            # measurement y0_p for the first stage -- same fix applied to
            # the scipy version in nmpc_pde.py (comparing a prediction to
            # itself would leave the largest, first transition
            # unconstrained).
            P_full = [y0_p] + P_seq
            for j in range(Np):
                dP = P_full[j + 1] - P_full[j]
                opti.subject_to(opti.bounded(-dy_max, dP, dy_max))

        opts = {"print_time": 0, "ipopt.print_level": 0, "ipopt.max_iter": 100,
                "ipopt.sb": "yes"}
        opts.update(ipopt_opts or {})
        opti.solver("ipopt", opts)

        self.opti, self.u_dec, self.u0_p, self.y0_p = opti, u_dec, u0_p, y0_p
        self._prev_lam_g = None

    def solve(self, u0, y0_true, u_init=None):
        opti = self.opti
        opti.set_value(self.u0_p, float(u0))
        opti.set_value(self.y0_p, float(y0_true))

        if u_init is not None:
            opti.set_initial(self.u_dec, _to_np(u_init))
        if self._prev_lam_g is not None:
            opti.set_initial(opti.lam_g, self._prev_lam_g)

        try:
            sol = opti.solve()
            self.last_success = True
        except RuntimeError:
            sol = opti.debug
            self.last_success = False

        try:
            self.last_nit = sol.stats().get("iter_count", -1)
        except RuntimeError:
            self.last_nit = -1
        self._prev_lam_g = sol.value(opti.lam_g)

        u_dec_opt = torch.tensor(np.atleast_1d(sol.value(self.u_dec)), dtype=torch.float32)
        u_apply = torch.clamp(u_dec_opt[0], self.u_min, self.u_max)
        return u_apply.detach(), u_dec_opt.detach()


class CasadiPDERTIController:
    """
    See module docstring, architecture (2): a single convex QP per
    timestep, linearizing the transient net's pressure prediction
    P = f_w(x_bar, 1, u_prev, u)[0] around a nominal control trajectory
    (the previous solve's shifted decisions, or u0 held constant if
    none is available yet), solved with CasADi's built-in `qrqp`.

    Like `CasadiRTIController` on the ODE-PINC side, this trades
    solution quality (one Gauss-Newton-like step rather than iterating
    to NLP convergence) for a fixed, small per-step cost -- appropriate
    when latency matters more than per-step optimality.
    """

    def __init__(self, transient_model, x_bar, target, Np, Nc,
                 lambda_smooth=0.1, dy_max=None, u_min=0.0, u_max=1.0):
        self.x_bar = x_bar
        self.target = float(target)
        self.Np, self.Nc = Np, Nc
        self.lambda_smooth = lambda_smooth
        self.dy_max = dy_max
        self.u_min = u_min
        self.u_max = u_max

        step = pinc_transient_step_to_casadi(transient_model, x_bar)
        self._step_keepalive = step

        u0_sym = ca.MX.sym("u0", 1)
        u_sym = ca.MX.sym("u", 1)
        y_sym = step(u0_sym, u_sym)
        P_sym = y_sym[0]
        dPdu0 = ca.jacobian(P_sym, u0_sym)
        dPdu = ca.jacobian(P_sym, u_sym)
        # one Function giving the affine pressure model (P, dP/du0, dP/du)
        # at a point, evaluated fresh at every stage of every timestep's
        # nominal trajectory below
        self._lin = ca.Function("lin", [u0_sym, u_sym], [P_sym, dPdu0, dPdu])

        self._prev_u_nom = None  # warm-start nominal control trajectory (Np,)

    def _nominal_trajectory(self, u0, u_init):
        if self._prev_u_nom is not None:
            return self._prev_u_nom

        if u_init is not None:
            u_dec = _to_np(u_init).reshape(-1)
        else:
            u_dec = np.full(self.Nc, float(u0))

        u_full = np.concatenate([u_dec, np.full(max(self.Np - self.Nc, 0), u_dec[-1])])
        return u_full[: self.Np]

    def solve(self, u0, y0_true, u_init=None):
        Np, Nc = self.Np, self.Nc
        u_nom = self._nominal_trajectory(u0, u_init)

        # Linearize at every stage j: P_j ~= c_j + a_j*u_prev_j + b_j*u_j,
        # around (u_nom[j-1] or u0, u_nom[j]).
        a, b, c = [], [], []
        u_prev_lin = float(u0)
        for j in range(Np):
            P_lin, dPdu0, dPdu = self._lin(u_prev_lin, u_nom[j])
            P_lin, dPdu0, dPdu = float(P_lin), float(dPdu0), float(dPdu)
            c.append(P_lin - dPdu0 * u_prev_lin - dPdu * u_nom[j])
            a.append(dPdu0)
            b.append(dPdu)
            u_prev_lin = u_nom[j]

        u_dec = ca.MX.sym("u_dec", Nc)

        def u_at(j):
            return u_dec[j] if j < Nc else u_dec[Nc - 1]

        u_prev_expr = ca.DM(float(u0))
        P_seq = []
        g_rows, lbg_rows, ubg_rows = [], [], []
        for j in range(Np):
            u_cur = u_at(j)
            P_j = a[j] * u_prev_expr + b[j] * u_cur + c[j]
            P_seq.append(P_j)
            g_rows.append(u_cur)
            lbg_rows.append(self.u_min)
            ubg_rows.append(self.u_max)
            u_prev_expr = u_cur

        if self.dy_max is not None:
            P_full = [ca.DM(float(y0_true))] + P_seq
            for j in range(Np):
                g_rows.append(P_full[j + 1] - P_full[j])
                lbg_rows.append(-self.dy_max)
                ubg_rows.append(self.dy_max)

        cost = sum((P_seq[j] - self.target) ** 2 for j in range(Np))
        smooth = 0
        u_prev_dec = float(u0)
        for i in range(Nc):
            smooth = smooth + (u_dec[i] - u_prev_dec) ** 2
            u_prev_dec = u_dec[i]
        cost = cost + self.lambda_smooth * smooth

        g = ca.vertcat(*g_rows)
        qp = {"x": u_dec, "f": cost, "g": g}
        solver = ca.qpsol("pde_rti_qp", "qrqp", qp, {
            "print_time": 0, "print_iter": False, "print_header": False,
            "print_info": False, "error_on_fail": False,
        })

        lbg = np.array(lbg_rows, dtype=float)
        ubg = np.array(ubg_rows, dtype=float)

        try:
            sol = solver(lbg=lbg, ubg=ubg)
            self.last_success = bool(solver.stats().get("success", False))
        except RuntimeError:
            self.last_success = False
            sol = None

        if not self.last_success or sol is None:
            # QP infeasible/failed: fall back to holding the nominal
            # trajectory as-is (not zeroing it out), so the next
            # timestep's linearization point isn't poisoned -- same
            # fallback behavior as CasadiRTIController on the ODE side.
            u_dec_opt = u_nom[:Nc].copy()
        else:
            u_dec_opt = np.array(sol["x"]).reshape(-1)

        self.last_nit = -1

        u_apply = float(np.clip(u_dec_opt[0], self.u_min, self.u_max))

        # shift the nominal trajectory forward for next timestep's
        # linearization point (RTI warm start)
        updated = u_nom.copy()
        updated[:Nc] = u_dec_opt
        if Nc < Np:
            updated[Nc:] = updated[Nc - 1]
        self._prev_u_nom = np.concatenate([updated[1:], updated[-1:]])

        return (torch.tensor(u_apply, dtype=torch.float32),
                torch.tensor(u_dec_opt, dtype=torch.float32))
