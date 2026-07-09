"""
Bridges trained PINC nets (and arbitrary torch "control interface"
functions f_hat_w(y, u) -> y_next, Eq. 8) into CasADi, so that the
NMPC problem of Section 3.3.4 / Eq. (12) can be handed to a real NLP
solver (IPOPT) or QP solver instead of scipy's SLSQP.

Two ways of getting a control interface into CasADi are provided:

1. `pinc_step_to_casadi(model)` -- an *exact* symbolic reconstruction.
   A PINCModel's backbone is a plain MLP (Linear + activation, repeated,
   Linear at the end -- see `pinc/nn/mlp.py`). Its weights are just
   numbers once trained, so the same computation graph can be rebuilt
   directly out of CasADi symbols. This gives CasADi's own AD exact,
   sparse first- and second-order derivatives of the network with
   essentially zero overhead per call -- no finite differences, no
   callback round-trips to Python/torch. This is the recommended path
   for the PINC net itself, and is what both NMPC classes in
   `nmpc_casadi.py` use whenever they're handed a `PINCModel.step`.

2. `TorchStepCallback` -- a generic `casadi.Callback` wrapping *any*
   torch function `f(y, u) -> y_next` (batched, shape (1, dim)), with
   first derivatives supplied via `torch.autograd`. This is the
   fallback used for control interfaces that aren't a plain PINCModel
   (e.g. `rk4_control_interface`, if one wants to solve the ODE/RK
   baseline with the same CasADi-based NMPC code for a fair
   architecture-for-architecture comparison). It is much slower than
   (1) since every call/Jacobian crosses the Python/torch boundary and
   only exposes first derivatives (IPOPT falls back to a
   quasi-Newton/BFGS Hessian approximation automatically in that case).
"""
import numpy as np
import torch
import casadi as ca


_ACTIVATION_MAP = {
    "Tanh": ca.tanh,
    "ReLU": lambda z: ca.fmax(z, 0),
    "Sigmoid": lambda z: 1 / (1 + ca.exp(-z)),
    "Identity": lambda z: z,
}


def mlp_forward_casadi(mlp: torch.nn.Module, x_sym):
    """
    Rebuilds the forward pass of a `pinc.nn.mlp.MLP` symbolically, given
    a CasADi symbol `x_sym` (column vector, shape (in_dim, 1)).

    Walks `mlp.net` (an `nn.Sequential` of alternating Linear/activation
    layers, see `pinc/nn/mlp.py`), extracting each Linear's weight/bias
    as plain numpy arrays (`.detach().numpy()`) and applying the matching
    CasADi op for each activation. Returns a CasADi expression (MX or SX,
    matching the type of `x_sym`) for the network output, shape
    (out_dim, 1).

    Raises a clear error rather than silently mis-reconstructing the
    network if it encounters a layer type it doesn't know how to
    translate (e.g. a custom or unsupported activation).
    """
    z = x_sym
    for layer in mlp.net:
        cls_name = type(layer).__name__
        if isinstance(layer, torch.nn.Linear):
            W = layer.weight.detach().numpy()  # (out, in)
            b = layer.bias.detach().numpy()    # (out,)
            z = ca.mtimes(ca.DM(W), z) + ca.DM(b)
        elif cls_name in _ACTIVATION_MAP:
            z = _ACTIVATION_MAP[cls_name](z)
        else:
            raise NotImplementedError(
                f"mlp_forward_casadi: don't know how to translate layer "
                f"'{cls_name}' to CasADi. Supported activations: "
                f"{list(_ACTIVATION_MAP)}. Add it to _ACTIVATION_MAP if "
                f"it's elementwise."
            )
    return z


def pinc_step_symbolic(model, y_sym, u_sym):
    """
    Builds the symbolic expression for the PINC control interface

        y_next = f_hat_w(y, u) = f_w(T, y, u)                  (Eq. 8)

    directly out of CasADi symbols `y_sym` (state_dim, 1) and `u_sym`
    (control_dim, 1), for embedding into a larger NLP (e.g. chaining it
    N2 times for a rollout, or using it as an equality constraint in a
    multiple-shooting formulation). `model` must be a `PINCModel`.
    """
    t_sym = ca.DM([model.T])
    net_in = ca.vertcat(t_sym, y_sym, u_sym)
    return mlp_forward_casadi(model.backbone, net_in)


def pinc_step_to_casadi(model, name="pinc_step"):
    """
    Wraps `pinc_step_symbolic` into a standalone `casadi.Function`
    y_next = f(y, u), useful when you just want a black-box-looking but
    still fully-symbolic (exact-AD) callable, e.g. for quick testing or
    for building the RTI controller's Jacobian.
    """
    y = ca.MX.sym("y", model.state_dim)
    u = ca.MX.sym("u", model.control_dim)
    y_next = pinc_step_symbolic(model, y, u)
    return ca.Function(name, [y, u], [y_next], ["y", "u"], ["y_next"])


class TorchStepCallback(ca.Callback):
    """
    Generic CasADi Callback wrapping an arbitrary torch control
    interface `step(y, u) -> y_next` (batched signature, shape (1, dim),
    matching `PINCModel.step` / `rk4_control_interface`'s convention)
    as a black-box CasADi function, with the Jacobian supplied via
    `torch.autograd` rather than CasADi's own AD (since CasADi cannot
    see through arbitrary torch code).

    Use this for control interfaces that are *not* a plain PINCModel
    (most commonly the RK4 "ODE/RK" baseline), so that the same
    CasADi/IPOPT-based NMPC classes in `nmpc_casadi.py` can drive either
    predictive model, exactly mirroring how `NMPCController` in
    `nmpc.py` can already be pointed at either `model.step` or
    `rk4_control_interface(...)`.

    Derivatives are obtained via CasADi's own built-in finite-difference
    Callback differentiation (`enable_fd=True`), rather than a hand
    rolled reverse-mode Jacobian -- simpler and more robust than wiring
    a custom `get_jacobian`, at the cost of a few extra function
    evaluations per NLP iteration (each perturbing one input dimension).
    For the small state/control dimensions used here (<=6 combined) this
    overhead is negligible next to IPOPT's own linear-algebra cost, and
    this fallback path is only used for control interfaces that are not
    a plain `PINCModel` in the first place (see
    `make_control_interface_casadi`).
    """

    def __init__(self, step_fn, state_dim, control_dim, name="torch_step", opts=None):
        ca.Callback.__init__(self)
        self.step_fn = step_fn
        self.state_dim = state_dim
        self.control_dim = control_dim
        fd_opts = {"enable_fd": True, "fd_method": "central"}
        fd_opts.update(opts or {})
        self.construct(name, fd_opts)

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.state_dim if i == 0 else self.control_dim, 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.state_dim, 1)

    def eval(self, arg):
        y = torch.tensor(np.array(arg[0]).reshape(1, -1), dtype=torch.float32)
        u = torch.tensor(np.array(arg[1]).reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            y_next = self.step_fn(y, u)[0]
        return [ca.DM(y_next.numpy().reshape(-1, 1))]


def make_control_interface_casadi(control_interface, state_dim, control_dim):
    """
    Convenience dispatcher: if `control_interface` is the bound `.step`
    method of a `PINCModel`, returns the exact symbolic builder (fast
    path); otherwise wraps it generically with `TorchStepCallback`
    (slow-but-general fallback, e.g. for the RK4 baseline).

    Returns a callable `f(y_sym, u_sym) -> y_next_sym` usable directly
    inside a CasADi symbolic graph (both paths support this, since a
    `casadi.Function` called on MX/SX symbols also returns a symbolic
    expression).
    """
    model = getattr(control_interface, "__self__", None)
    if model is not None and type(model).__name__ == "PINCModel":
        fn = pinc_step_to_casadi(model)
        return lambda y_sym, u_sym: fn(y_sym, u_sym)

    callback = TorchStepCallback(control_interface, state_dim, control_dim)
    return lambda y_sym, u_sym: callback(y_sym, u_sym)
