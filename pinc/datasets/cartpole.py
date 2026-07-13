import numpy as np
import torch
from scipy.optimize import minimize

from pinc.datasets.pinc_dataset import PINCSampler
from pinc.simulation.rk4 import rk4_step

# Single source of truth for the cart-pole state domain -- used both by
# `make_cartpole_sampler` (as the actual sampling box) and by
# `cartpole_state_weights` (to derive per-channel loss weights from the
# same characteristic scales), so the two can't silently drift apart.
#
# NOTE: tightened after an initial training run showed the physics-
# residual loss plateauing well above the data/endpoint losses, with
# visible open-loop self-loop drift even in the u=0 free-response case
# -- the original +/-2*pi / +/-10 / +/-6 box was wide enough that the
# network's capacity/training budget was mostly being spent fitting a
# residual across huge swaths of state space the actual swing-up
# trajectory never visits, diluting effective sample density where it
# mattered. These ranges are still generous relative to the verified
# reference trajectory (theta in [~0.25, ~4.85] rad, |x| < 0.65m -- see
# `generate_swingup_reference`'s docstring/verification), but no longer
# span a near-full revolution of slack on every side.
#
# x, x_dot   : +/- 1.5m / +/- 4 m/s -- still headroom beyond the
#              reference's observed |x| < 0.65m (e.g. for NMPC tracking
#              error to push the plant somewhat off the reference),
#              just not the original +/-3 / +/-6.
# theta      : (-1.0, 2*pi + 0.5) rad -- covers hanging-start (pi)
#              through a full swing-up-and-slightly-past (~2*pi), plus
#              a bit of negative-side margin in case a correcting
#              controller overshoots past upright the other way. theta
#              is never wrapped anywhere in this codebase (see
#              `CartPole`'s docstring), and the sin/cos encoding in
#              `CartPolePINCModel` resolves the resulting periodicity
#              ambiguity by having the network predict a *delta* rather
#              than an absolute angle -- this range only needs to cover
#              states actually visited, not exactly one period.
# theta_dot  : +/- 8 rad/s -- swing-up passes through fairly fast
#              intermediate angular velocities (energy-pumping motion),
#              so this still needs real headroom, just less than the
#              original +/-10.
CARTPOLE_Y_RANGE = [(-1.5, 1.5), (-4.0, 4.0), (-1.0, 2 * np.pi + 0.5), (-8.0, 8.0)]
# u (force): +/- 15 N -- this is the actuator limit itself (matches
# `generate_swingup_reference`'s `u_max`), not a domain-coverage
# question, so it isn't part of the same "tighten to what's visited"
# reasoning as the state ranges above.
CARTPOLE_U_RANGE = [(-15.0, 15.0)]


def cartpole_state_weights():
    """
    Per-channel weights for `PINCLoss(..., state_weights=...)`, derived
    from `CARTPOLE_Y_RANGE`'s half-widths (1 / half-width, so a given
    *relative* error -- e.g. "off by 10% of this channel's typical
    range" -- costs the same in the loss regardless of which channel
    it's in).

    Why this matters here specifically: `theta` naturally ranges over
    several radians while `x_dot` naturally ranges over a few tenths of
    a m/s -- in an *unweighted* MSE across all four channels combined
    (the base `PINCLoss` behavior), theta's errors are numerically much
    larger and dominate the gradient, so the optimizer has little
    incentive to fix x/x_dot even when they're relatively very wrong.
    That's exactly the failure mode a training run showed: theta/
    theta_dot tracked a validation rollout reasonably well while x/x_dot
    drifted badly (and inconsistently with their own predicted
    derivative) in the self-loop diagnostics. Weighting by inverse
    range restores balance across channels.
    """
    half_widths = torch.tensor([(hi - lo) / 2.0 for lo, hi in CARTPOLE_Y_RANGE])
    return 1.0 / half_widths


def make_cartpole_sampler(physics, T, device="cpu"):
    """Sampling ranges for the cart-pole swing-up task -- see
    `CARTPOLE_Y_RANGE`/`CARTPOLE_U_RANGE` above for the reasoning behind
    each bound."""
    return PINCSampler(physics, T=T, y_range=CARTPOLE_Y_RANGE, u_range=CARTPOLE_U_RANGE, device=device)


def random_control_signal(n_steps, control_dim=1, u_range=(-15.0, 15.0), hold=1, seed=None):
    """Same piecewise-constant random-control generator as
    `pinc.datasets.vanderpol.random_control_signal`, used here for the
    long-range self-loop generalization diagnostic."""
    if seed is not None:
        torch.manual_seed(seed)

    n_holds = n_steps // hold + 1
    values = torch.empty(n_holds, control_dim).uniform_(*u_range)
    u = values.repeat_interleave(hold, dim=0)[:n_steps]
    return u


def _rollout_cost_and_grad(u_flat_np, physics, x0, dt, N, Qf, R, x_target, x_penalty):
    u = torch.tensor(u_flat_np, dtype=torch.float32, requires_grad=True)
    u_seq = u.reshape(N, physics.control_dim)

    x = x0.clone()
    xs = [x]
    for k in range(N):
        x = rk4_step(physics.dynamics, x, u_seq[k], dt)
        xs.append(x)
    xs = torch.stack(xs)  # (N+1, state_dim)

    err = xs[-1] - x_target
    cost = err @ Qf @ err
    cost = cost + R * torch.sum(u_seq ** 2)
    cost = cost + x_penalty * torch.sum(xs[:, 0] ** 2)

    cost.backward()
    grad = u.grad.detach().numpy().astype(np.float64)
    return cost.item(), grad


def generate_swingup_reference(physics, T, N=250, x0=None, x_target=None,
                                Qf=None, R=1e-2, x_penalty=1e-2,
                                u_max=15.0, maxiter=300, u_init=None):
    """
    Generates an open-loop reference swing-up trajectory by direct
    single-shooting trajectory optimization (decision variables:
    the whole control sequence u[0..N-1]; cost: quadratic terminal
    error to the upright equilibrium + control effort + a small
    running penalty on cart drift; solved with the same
    `scipy.optimize.minimize(method="SLSQP")` + autograd-gradient
    pattern already used by `pinc.control.nmpc.NMPCController`).

    This is *not* itself an NMPC controller -- it's a one-shot offline
    solve, used to build a `y_ref_full` trajectory that
    `pinc.control.nmpc.run_nmpc_simulation` (unmodified) can then track
    in closed loop, exactly like `train_vanderpol.py`'s
    `run_control_experiment` tracks a (much simpler, all-zero) setpoint.
    Reference-tracking was chosen deliberately over asking the
    receding-horizon NMPC to discover the swing-up maneuver from
    scratch at every step: SLSQP's local gradients don't reliably find
    a genuinely nonconvex energy-pumping motion in a from-zero,
    iteration-budget-limited solve at every timestep, whereas a single
    generous offline solve is well-suited to finding it once, before
    closed-loop tracking (with model-plant mismatch, in the PINC case)
    even enters the picture.

    x0, x_target : (4,) tensors; default to hanging-down-at-rest ->
                   upright-at-rest if not given.
    Qf           : (4,4) terminal-state cost weight; default emphasizes
                   the angle strongly (it's the coordinate that must
                   actually flip by pi, i.e. genuinely nonconvex),
                   moderately for cart position/velocities so the
                   solve doesn't need to *also* fight to recenter
                   perfectly (a downstream balance step, if desired,
                   handles fine centering).

    returns (u_opt, y_traj): u_opt (N, control_dim), y_traj (N+1, 4).
    """
    if x0 is None:
        x0 = torch.tensor([0.0, 0.0, np.pi, 0.0])
    if x_target is None:
        x_target = torch.tensor([0.0, 0.0, 0.0, 0.0])
    if Qf is None:
        Qf = torch.diag(torch.tensor([2.0, 2.0, 80.0, 30.0]))

    n_dec = N * physics.control_dim
    u0 = np.zeros(n_dec) if u_init is None else u_init.reshape(-1).numpy().astype(np.float64)
    bounds = [(-u_max, u_max)] * n_dec

    res = minimize(
        _rollout_cost_and_grad, u0,
        args=(physics, x0, T, N, Qf, R, x_target, x_penalty),
        jac=True, method="SLSQP", bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-8},
    )

    u_opt = torch.tensor(res.x, dtype=torch.float32).reshape(N, physics.control_dim)

    with torch.no_grad():
        x = x0.clone()
        traj = [x.clone()]
        for k in range(N):
            x = rk4_step(physics.dynamics, x, u_opt[k], T)
            traj.append(x.clone())
        y_traj = torch.stack(traj)

    print(f"Swing-up reference solve: success={res.success}, iters={res.nit}, "
          f"final cost={res.fun:.4f}, final state={y_traj[-1].tolist()}")

    return u_opt, y_traj