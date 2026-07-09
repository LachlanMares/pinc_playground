"""
Drop-in replacements for `NMPCController` (nmpc.py), all implementing
the same Multiple-Shooting-inspired NMPC of Eq. (12), but solved with
CasADi instead of scipy's SLSQP. Each class exposes the exact same
public surface as `NMPCController`:

    controller = SomeCasadiController(control_interface, control_dim,
                                       N1, N2, Nu, Q, R, ...)
    u_apply, du_opt = controller.solve(y_current, u_prev, y_ref_horizon,
                                        du_init=None)

so any of them can be substituted directly wherever `NMPCController` is
used -- including inside `run_nmpc_simulation` (pass `controller=...`,
see the updated signature in `nmpc.py`).

Three different MPC *architectures* are provided:

1. `CasadiSingleShootingNMPC`
   Same decision variables as the original (`NMPCController`): the
   control increment sequence only, states obtained by rolling the
   model forward inside the cost/constraint expressions (single
   shooting). The NLP is built symbolically *once* in `__init__` using
   CasADi's `Opti` stack and then re-solved every timestep by just
   updating parameter values and warm-starting -- solved with IPOPT.
   Simplest change from the original: same problem, better solver
   (exact sparse AD instead of scipy's numeric/autograd-bridged SLSQP,
   proper warm-starting of both primal and dual variables).

2. `CasadiMultipleShootingNMPC`
   True multiple shooting, matching Eq. (12a)-(12f) literally: the
   predicted state at every stage is its own decision variable, tied
   to the previous stage by an explicit equality constraint
   `Y[j] == f_hat_w(Y[j-1], U[j])` (Eq. 12b). This is more robust than
   single shooting for systems with slow/indirectly-coupled dynamics
   (e.g. the four-tank system, Sec. 4.2) since IPOPT can move the state
   trajectory and the controls independently rather than only ever
   seeing the compounded effect of the whole rollout, and it makes
   state constraints (Eq. 12e) trivial box constraints on decision
   variables instead of requiring a Jacobian of the entire rollout.

3. `CasadiRTIController`
   A linear-time-varying (LTV) Real-Time Iteration style controller
   (Diehl-style RTI / classical linear MPC): at each timestep, the
   network's control interface is linearized (via CasADi's exact AD)
   around a nominal trajectory, and a single strictly-convex QP is
   solved (with CasADi's built-in `qrqp` solver -- no external QP
   dependency needed) instead of a full nonlinear program. This trades
   optimality (one Gauss-Newton-like step per timestep, rather than
   iterating to NLP convergence) for speed and determinism, which is
   the standard trade-off made in real-time embedded NMPC.

All three use the exact symbolic PINC network reconstruction from
`casadi_backend.py` whenever `control_interface` is a `PINCModel.step`,
and fall back to a finite-difference `Callback` wrapper otherwise (e.g.
for the RK4 "ODE/RK" baseline), exactly mirroring how `NMPCController`
in `nmpc.py` can be pointed at either predictive model already.
"""
import numpy as np
import torch
import casadi as ca

from pinc.control.casadi_backend import make_control_interface_casadi


def _infer_state_dim(control_interface, state_dim=None):
    model = getattr(control_interface, "__self__", None)
    if state_dim is not None:
        return state_dim
    if model is not None and hasattr(model, "state_dim"):
        return model.state_dim
    raise ValueError("state_dim could not be inferred; pass it explicitly.")


def _to_np(mat_or_tensor):
    if isinstance(mat_or_tensor, torch.Tensor):
        return mat_or_tensor.detach().numpy()
    return np.asarray(mat_or_tensor)


class CasadiSingleShootingNMPC:
    """See module docstring, architecture (1)."""

    def __init__(self, control_interface, control_dim,
                 N1, N2, Nu, Q, R,
                 u_min=None, u_max=None,
                 state_constraints=None,
                 state_dim=None,
                 ipopt_opts=None):
        self.control_dim = control_dim
        self.state_dim = _infer_state_dim(control_interface, state_dim)
        self.N1, self.N2, self.Nu = N1, N2, Nu
        self.u_min = u_min
        self.u_max = u_max
        self.state_constraints = state_constraints or []

        step = make_control_interface_casadi(control_interface, self.state_dim, control_dim)
        # CasADi Callbacks (used for non-PINCModel control interfaces, e.g. the
        # RK4 baseline) need their Python wrapper object kept alive for as long
        # as any graph referencing them exists, or CasADi's C++ side loses the
        # Python-side eval() target ("Callback object has been deleted") --
        # keeping this reference on self is what prevents that.
        self._step_backend_keepalive = step

        Q_np, R_np = _to_np(Q), _to_np(R)

        opti = ca.Opti()
        du = opti.variable(Nu, control_dim)
        y0_p = opti.parameter(self.state_dim)
        u_prev_p = opti.parameter(control_dim)
        yref_p = opti.parameter(N2, self.state_dim)

        u = u_prev_p
        y = y0_p
        ys = []
        for j in range(N2):
            i = min(j, Nu - 1)
            if j < Nu:
                u = u + du[i, :].T
            y = step(y, u)
            ys.append(y)
            if u_min is not None:
                opti.subject_to(opti.bounded(ca.DM(u_min), u, ca.DM(u_max)))

        cost = 0
        for j in range(N1 - 1, N2):
            e = ys[j] - yref_p[j, :].T
            cost = cost + ca.mtimes([e.T, ca.DM(Q_np), e])
        for i in range(Nu):
            d = du[i, :].T
            cost = cost + ca.mtimes([d.T, ca.DM(R_np), d])
        opti.minimize(cost)

        for (idx, lo, hi) in self.state_constraints:
            for j in range(N2):
                opti.subject_to(opti.bounded(lo, ys[j][idx], hi))

        opts = {"print_time": 0, "ipopt.print_level": 0, "ipopt.max_iter": 100,
                "ipopt.sb": "yes"}
        opts.update(ipopt_opts or {})
        opti.solver("ipopt", opts)

        self.opti, self.du, self.y0_p, self.u_prev_p, self.yref_p = opti, du, y0_p, u_prev_p, yref_p
        self._prev_lam_g = None

    def solve(self, y_current, u_prev, y_ref_horizon, du_init=None):
        opti = self.opti
        opti.set_value(self.y0_p, _to_np(y_current))
        opti.set_value(self.u_prev_p, _to_np(u_prev))
        opti.set_value(self.yref_p, _to_np(y_ref_horizon))

        if du_init is not None:
            opti.set_initial(self.du, _to_np(du_init))
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

        du_opt = np.atleast_2d(sol.value(self.du))
        du_opt = torch.tensor(du_opt, dtype=torch.float32).reshape(self.Nu, self.control_dim)
        u_apply = u_prev + du_opt[0]
        if self.u_min is not None:
            u_apply = torch.clamp(u_apply,
                                   torch.tensor(self.u_min, dtype=torch.float32),
                                   torch.tensor(self.u_max, dtype=torch.float32))
        return u_apply.detach(), du_opt.detach()


class CasadiMultipleShootingNMPC:
    """See module docstring, architecture (2)."""

    def __init__(self, control_interface, control_dim,
                 N1, N2, Nu, Q, R,
                 u_min=None, u_max=None,
                 state_constraints=None,
                 state_dim=None,
                 ipopt_opts=None):
        self.control_dim = control_dim
        self.state_dim = _infer_state_dim(control_interface, state_dim)
        self.N1, self.N2, self.Nu = N1, N2, Nu
        self.u_min = u_min
        self.u_max = u_max
        self.state_constraints = state_constraints or []

        step = make_control_interface_casadi(control_interface, self.state_dim, control_dim)
        # CasADi Callbacks (used for non-PINCModel control interfaces, e.g. the
        # RK4 baseline) need their Python wrapper object kept alive for as long
        # as any graph referencing them exists, or CasADi's C++ side loses the
        # Python-side eval() target ("Callback object has been deleted") --
        # keeping this reference on self is what prevents that.
        self._step_backend_keepalive = step

        Q_np, R_np = _to_np(Q), _to_np(R)

        opti = ca.Opti()
        du = opti.variable(Nu, control_dim)
        Y = opti.variable(N2, self.state_dim)   # state nodes y[k+1..k+N2], Eq. (12b) decision vars
        y0_p = opti.parameter(self.state_dim)
        u_prev_p = opti.parameter(control_dim)
        yref_p = opti.parameter(N2, self.state_dim)

        u = u_prev_p
        y_prev_node = y0_p
        for j in range(N2):
            i = min(j, Nu - 1)
            if j < Nu:
                u = u + du[i, :].T
            y_pred = step(y_prev_node, u)
            opti.subject_to(Y[j, :].T == y_pred)  # Eq. (12b): explicit shooting-node equality
            y_prev_node = Y[j, :].T
            if u_min is not None:
                opti.subject_to(opti.bounded(ca.DM(u_min), u, ca.DM(u_max)))

        cost = 0
        for j in range(N1 - 1, N2):
            e = Y[j, :].T - yref_p[j, :].T
            cost = cost + ca.mtimes([e.T, ca.DM(Q_np), e])
        for i in range(Nu):
            d = du[i, :].T
            cost = cost + ca.mtimes([d.T, ca.DM(R_np), d])
        opti.minimize(cost)

        # Eq. (12e): direct box constraints on shooting-node variables,
        # no rollout Jacobian required (unlike the single-shooting /
        # scipy version), since Y is itself a decision variable.
        for (idx, lo, hi) in self.state_constraints:
            for j in range(N2):
                opti.subject_to(opti.bounded(lo, Y[j, idx], hi))

        opts = {"print_time": 0, "ipopt.print_level": 0, "ipopt.max_iter": 100,
                "ipopt.sb": "yes"}
        opts.update(ipopt_opts or {})
        opti.solver("ipopt", opts)

        self.opti, self.du, self.Y = opti, du, Y
        self.y0_p, self.u_prev_p, self.yref_p = y0_p, u_prev_p, yref_p
        self._prev_lam_g = None
        self._prev_Y = None  # warm start for the state nodes (not exposed via solve()'s signature)

    def solve(self, y_current, u_prev, y_ref_horizon, du_init=None):
        opti = self.opti
        opti.set_value(self.y0_p, _to_np(y_current))
        opti.set_value(self.u_prev_p, _to_np(u_prev))
        opti.set_value(self.yref_p, _to_np(y_ref_horizon))

        if du_init is not None:
            opti.set_initial(self.du, _to_np(du_init))
        if self._prev_Y is not None:
            # shift the previous state-node trajectory forward by one
            # stage as the warm start (drop node 0, repeat the last)
            opti.set_initial(self.Y, np.vstack([self._prev_Y[1:], self._prev_Y[-1:]]))
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
        self._prev_Y = np.atleast_2d(sol.value(self.Y))

        du_opt = np.atleast_2d(sol.value(self.du))
        du_opt = torch.tensor(du_opt, dtype=torch.float32).reshape(self.Nu, self.control_dim)
        u_apply = u_prev + du_opt[0]
        if self.u_min is not None:
            u_apply = torch.clamp(u_apply,
                                   torch.tensor(self.u_min, dtype=torch.float32),
                                   torch.tensor(self.u_max, dtype=torch.float32))
        return u_apply.detach(), du_opt.detach()


class CasadiRTIController:
    """
    See module docstring, architecture (3): a linear-time-varying Real-
    Time Iteration controller. Only one convex QP is solved per
    timestep (via CasADi's built-in `qrqp`), by linearizing the control
    interface's exact CasADi reconstruction around a nominal trajectory
    (the previous solve's applied-and-shifted trajectory, or an
    open-loop rollout at u_prev if none is available yet).

    This is *not* drop-in-equivalent in solution quality to the two NLP
    based classes above (it takes one Gauss-Newton-like step towards
    the optimum per timestep rather than iterating to convergence), but
    it is the architecture of choice when per-timestep latency matters
    more than each individual solve being fully optimal -- the standard
    trade-off in embedded/real-time NMPC.
    """

    def __init__(self, control_interface, control_dim,
                 N1, N2, Nu, Q, R,
                 u_min=None, u_max=None,
                 state_constraints=None,
                 state_dim=None):
        self.control_dim = control_dim
        self.state_dim = _infer_state_dim(control_interface, state_dim)
        self.N1, self.N2, self.Nu = N1, N2, Nu
        self.Q_np, self.R_np = _to_np(Q), _to_np(R)
        self.u_min = u_min
        self.u_max = u_max
        self.state_constraints = state_constraints or []

        step = make_control_interface_casadi(control_interface, self.state_dim, control_dim)
        # CasADi Callbacks (used for non-PINCModel control interfaces, e.g. the
        # RK4 baseline) need their Python wrapper object kept alive for as long
        # as any graph referencing them exists, or CasADi's C++ side loses the
        # Python-side eval() target ("Callback object has been deleted") --
        # keeping this reference on self is what prevents that.
        self._step_backend_keepalive = step

        y_sym = ca.MX.sym("y", self.state_dim)
        u_sym = ca.MX.sym("u", control_dim)
        y_next_sym = step(y_sym, u_sym)
        A_sym = ca.jacobian(y_next_sym, y_sym)
        B_sym = ca.jacobian(y_next_sym, u_sym)
        # one Function giving the affine model (A, B, y_next) at a point,
        # evaluated fresh at every stage of every timestep's nominal
        # trajectory below
        self._lin = ca.Function("lin", [y_sym, u_sym], [y_next_sym, A_sym, B_sym])

        self._prev_traj = None  # warm-start nominal (y[1..N2], u[0..Nu-1])

    def _nominal_trajectory(self, y0, u_prev, du_init):
        """Builds the trajectory to linearize around: the previous
        solve's shifted trajectory if available (classic RTI warm
        start), otherwise an open-loop rollout holding u_prev fixed."""
        if self._prev_traj is not None:
            return self._prev_traj

        y = y0.clone()
        u = u_prev.clone()
        ys, us = [], []
        for j in range(self.N2):
            i = min(j, self.Nu - 1)
            if j < self.Nu and du_init is not None:
                u = u + du_init[i]
            y_next = torch.tensor(np.array(self._lin(y.numpy(), u.numpy())[0]).reshape(-1),
                                   dtype=torch.float32)
            ys.append(y_next)
            us.append(u.clone())
            y = y_next
        return torch.stack(ys), torch.stack(us)

    def solve(self, y_current, u_prev, y_ref_horizon, du_init=None):
        N1, N2, Nu, nu, ny = self.N1, self.N2, self.Nu, self.control_dim, self.state_dim

        y_nom, u_nom = self._nominal_trajectory(y_current, u_prev, du_init)
        yref_np = _to_np(y_ref_horizon)

        # Linearize at every stage: y[j+1] ~= c_j + A_j*y[j] + B_j*u[j],
        # taken around the nominal trajectory (y_nom[j-1], u_nom[j]),
        # with y_nom[-1] standing in for the true current state.
        As, Bs, cs = [], [], []
        for j in range(N2):
            y_lin = y_current.numpy() if j == 0 else y_nom[j - 1].numpy()
            u_lin = u_nom[j].numpy()
            y_next, A, B = self._lin(y_lin, u_lin)
            A, B = np.array(A), np.array(B)
            y_next = np.array(y_next).reshape(-1)
            c = y_next - A @ y_lin - B @ u_lin
            As.append(A); Bs.append(B); cs.append(c)

        # Build the QP explicitly (decision vector x = [dU_flat; Y_flat])
        # and hand it to CasADi's built-in `qrqp` *conic* (QP) solver via
        # `qpsol`, rather than through the `Opti` convenience layer --
        # `Opti.solver(...)` expects a generic NLP-solver plugin name,
        # and the pip-distributed CasADi build only ships `qrqp` as a
        # low-level QP/conic plugin, not wrapped as one of those. Since
        # our problem *is* a QP (linear dynamics, quadratic cost, linear
        # bounds) this is in fact the more direct/appropriate interface
        # anyway, and it is what makes this genuinely a single-QP-per-
        # timestep controller rather than a disguised NLP solve.
        dU = ca.MX.sym("dU", Nu * nu)
        Y = ca.MX.sym("Y", N2 * ny)

        def dU_i(i):
            return dU[i * nu:(i + 1) * nu]

        def Y_j(j):
            return Y[j * ny:(j + 1) * ny]

        g_rows, lbg_rows, ubg_rows = [], [], []
        y_expr = ca.DM(y_current.numpy())
        for j in range(N2):
            i = min(j, Nu - 1)
            u_expr = ca.DM(u_nom[j].numpy()) + (dU_i(i) if j < Nu else 0)
            y_expr = ca.DM(As[j]) @ y_expr + ca.DM(Bs[j]) @ u_expr + ca.DM(cs[j])

            g_rows.append(Y_j(j) - y_expr)  # Eq. (12b) as an equality row
            lbg_rows.append(np.zeros(ny)); ubg_rows.append(np.zeros(ny))

            if self.u_min is not None:
                g_rows.append(u_expr)
                lbg_rows.append(np.asarray(self.u_min, dtype=float))
                ubg_rows.append(np.asarray(self.u_max, dtype=float))

        cost = 0
        for j in range(N1 - 1, N2):
            e = Y_j(j) - ca.DM(yref_np[j])
            cost = cost + ca.mtimes([e.T, ca.DM(self.Q_np), e])
        for i in range(Nu):
            d = dU_i(i)
            cost = cost + ca.mtimes([d.T, ca.DM(self.R_np), d])

        x = ca.vertcat(dU, Y)
        g = ca.vertcat(*g_rows)
        lbg = np.concatenate(lbg_rows) if g_rows else np.zeros(0)
        ubg = np.concatenate(ubg_rows) if g_rows else np.zeros(0)

        lbx = -np.inf * np.ones(Nu * nu + N2 * ny)
        ubx = np.inf * np.ones(Nu * nu + N2 * ny)
        for (idx, lo, hi) in self.state_constraints:
            for j in range(N2):
                pos = Nu * nu + j * ny + idx
                lbx[pos], ubx[pos] = lo, hi

        qp = {"x": x, "f": cost, "g": g}
        solver = ca.qpsol("rti_qp", "qrqp", qp, {
            "print_time": 0, "print_iter": False, "print_header": False, "print_info": False,
            "error_on_fail": False,
        })

        try:
            sol = solver(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            self.last_success = bool(solver.stats().get("success", False))
            self.last_nit = -1
        except RuntimeError:
            self.last_success = False
            self.last_nit = -1
            sol = None

        if not self.last_success or sol is None:
            # QP infeasible/failed (common if the chained linearization has
            # drifted into a region where the sqrt() orifice nonlinearity's
            # steep local gradient makes the affine model a poor fit --
            # see the four-tank caveat in the class docstring): fall back
            # to applying the nominal (zero-increment) trajectory as-is,
            # rather than a garbage control action -- and critically,
            # keep the *nominal* state trajectory (not an all-zero one)
            # so next timestep's linearization point isn't poisoned.
            sol = {"x": ca.DM(np.concatenate([np.zeros(Nu * nu), y_nom.numpy().reshape(-1)]))}

        x_opt = np.array(sol["x"]).reshape(-1)
        dU_opt = x_opt[:Nu * nu].reshape(Nu, nu)
        Y_opt = x_opt[Nu * nu:].reshape(N2, ny)

        u_apply = u_nom[0].numpy() + dU_opt[0]
        u_apply = torch.tensor(u_apply, dtype=torch.float32)
        if self.u_min is not None:
            u_apply = torch.clamp(u_apply,
                                   torch.tensor(self.u_min, dtype=torch.float32),
                                   torch.tensor(self.u_max, dtype=torch.float32))

        du_opt = torch.tensor(dU_opt, dtype=torch.float32)

        # shift the nominal trajectory forward for next timestep's
        # linearization point (RTI warm start)
        # Apply this solve's correction to the nominal trajectory (only
        # the first Nu stages get an actual decision variable; stages
        # beyond Nu are held at the same value, mirroring the u_expr
        # construction above), then shift the whole thing forward by
        # one stage as next timestep's linearization point.
        updated_u = u_nom.clone()
        updated_u[:Nu] = u_nom[:Nu] + du_opt
        if Nu < N2:
            updated_u[Nu:] = updated_u[Nu - 1]
        u_nom_new = torch.cat([updated_u[1:], updated_u[-1:]], dim=0)
        y_nom_new = torch.tensor(Y_opt, dtype=torch.float32)
        y_nom_new = torch.cat([y_nom_new[1:], y_nom_new[-1:]], dim=0)
        self._prev_traj = (y_nom_new, u_nom_new)

        return u_apply.detach(), du_opt.detach()