"""
PDE-PINC analogue of `pinc.control.casadi_backend`: bridges a trained
`PINCTransientPDE` net into CasADi so the MPC of Sec. 4.5 / Eq. (37) can
be handed to IPOPT (or a linearized QP) instead of scipy's SLSQP
(`pinc.control.nmpc_pde.PDEMPCController`).

The control interface being reconstructed here is different in kind from
the ODE-PINC's f(y, u) -> y_next (see `nmpc_pde.py`'s module docstring):
the PDE-PINC's transient net is evaluated at a *fixed* spatial position
x_bar and a *fixed* t=1 (one window), taking the previous window's
control u0 and the current window's control u as inputs and returning
the predicted (pressure, velocity) pair -- there is no "state" being
rolled forward through the network itself, only the control sequence.

    y = (P, V) = f_w(x_bar, 1, u0, u)

This reuses `mlp_forward_casadi` from `casadi_backend.py` verbatim (the
backbone is the same plain `pinc.nn.mlp.MLP`), so any fix or new
activation added there is picked up here automatically.
"""
import casadi as ca

from pinc.control.casadi_backend import mlp_forward_casadi


def pinc_transient_pressure_symbolic(model, x_bar, u0_sym, u_sym):
    """
    Builds the symbolic expression for

        y = f_w(x_bar, 1, u0, u) = (P, V)

    directly out of CasADi symbols `u0_sym`, `u_sym` (each scalar, i.e.
    (1, 1)), for a `PINCTransientPDE` at fixed measurement position
    `x_bar`. Returns the full (2, 1) output; callers pick out pressure
    (index 0) or velocity (index 1) as needed.
    """
    x_sym = ca.DM([x_bar])
    t_sym = ca.DM([1.0])
    net_in = ca.vertcat(x_sym, t_sym, u0_sym, u_sym)
    return mlp_forward_casadi(model.backbone, net_in)


def pinc_transient_step_to_casadi(model, x_bar, name="pinc_transient_step"):
    """
    Wraps `pinc_transient_pressure_symbolic` into a standalone
    `casadi.Function` y = f(u0, u), y = (P, V), for embedding into a
    larger NLP/QP (chaining it Np times for a prediction horizon, or
    differentiating it for an RTI-style linearization).
    """
    u0 = ca.MX.sym("u0", 1)
    u = ca.MX.sym("u", 1)
    y = pinc_transient_pressure_symbolic(model, x_bar, u0, u)
    return ca.Function(name, [u0, u], [y], ["u0", "u"], ["y"])
