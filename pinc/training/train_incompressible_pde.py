"""
Reproduces (at small scale) the incompressible-flow experiments of Sec.
5.1 of Miyatake et al., "Physics-Informed Neural Networks for Control of
Single-Phase Flow Systems Governed by Partial Differential Equations"
(arXiv:2506.06188), extending this repo's ODE-PINC (arXiv:2104.02556)
implementation to PDEs:

  1) Train a PINC-SteadyState net over a family of controls (Sec. 4.3 /
     5.1.1) and compare its pressure/velocity profiles against the exact
     analytic steady state (Fig. 8 style).
  2) Train a PINC-Transient net (Sec. 4.4 / 5.1.2), using the frozen
     steady-state net to supply initial-condition targets (Eq. 34),
     and validate an open-loop forward simulation against the exact
     plant (Fig. 10 style).
  3) Closed-loop MPC control (Sec. 4.5 / 5.1.2) using the transient net
     as predictive model, tracking an (unfeasible, production-maximizing)
     low target pressure at a fixed "PDG" position, subject to a rate
     constraint (Fig. 11 style).

Run with:  python -m pinc.training.train_incompressible_pde
Resume a previously interrupted run with:
           python -m pinc.training.train_incompressible_pde --resume
Skip training and just re-run the plots/MPC demo from saved checkpoints:
           python -m pinc.training.train_incompressible_pde --load-only

Simplification vs. the paper (documented, see README): here the PINC
window length T (tref) is set equal to the MPC sampling period Ts, so one
network call (t=1) corresponds to exactly one control step -- the paper
allows tref >> Ts and queries the transient net at intermediate t in
(0,1) within MPC's own prediction horizon. This keeps the control-loop
plumbing simple and reuses the same "network step == one MPC step"
pattern as the ODE-PINC case in this repo.
"""
import argparse

import matplotlib.pyplot as plt
import torch

from pinc.physics.pde_incompressible_flow import IncompressibleFlowParams, IncompressibleFlowPhysics
from pinc.simulation.pipe_flow_plant import IncompressiblePipePlant
from pinc.simulation.rk4 import simulate, rk4_control_interface
from pinc.nn.mlp import MLP
from pinc.models.pinc_pde import PINCSteadyStatePDE, PINCTransientPDE
from pinc.datasets.pde_incompressible import SteadyStatePDESampler, TransientPDESampler
from pinc.losses.pinc_pde_loss import SteadyStatePDELoss, TransientPDELoss
from pinc.core.pde_trainer import PDETrainer
from pinc.control.nmpc_pde import run_pde_mpc_simulation, _steady_velocity_for_control
from pinc.utils.checkpoint import load_pinc_steady_state_pde_model, load_pinc_transient_pde_model


U_RANGE = (0.05, 0.95)  # normalized downstream pressure (control) range trained over
_VAL_US = [0.15, 0.35, 0.55, 0.75, 0.90]  # controls used for steady-state validation/MAPE


# ----------------------------------------------------------------------
# Steady-state stage
# ----------------------------------------------------------------------
def train_steady_state(physics, hidden=32, depth=3, k1_epochs=300, k2_iters=300,
                        n_collocation=1000, n_boundary=200, device="cpu",
                        checkpoint_path=None, resume=False, save_every=100):
    """
    checkpoint_path : if given, periodically saves training progress
                      here (model + ADAM state + history) so a killed/
                      interrupted run can be resumed -- see
                      `pinc.utils.checkpoint.save_checkpoint` and
                      `Trainer.fit` on the ODE-PINC side for the same
                      pattern.
    resume          : if True and a checkpoint already exists at
                      `checkpoint_path`, restores model/optimizer state
                      and continues training instead of starting over.
    """
    meta = {"hidden": hidden, "depth": depth}

    model = PINCSteadyStatePDE(MLP(in_dim=2, out_dim=2, hidden=hidden, depth=depth)).to(device)
    sampler = SteadyStatePDESampler(u_range=U_RANGE, device=device)
    loss_fn = SteadyStatePDELoss(physics)

    def sample_batches():
        return sampler.sample_collocation(n_collocation), sampler.sample_boundary(n_boundary)

    x_val = torch.linspace(0, 1, 25).unsqueeze(-1)

    def validate_fn(m):
        mape_p, mape_v = _steady_state_mape(m, physics, _VAL_US, x_val)
        return (mape_p + mape_v) / 2

    trainer = PDETrainer(model, sample_batches, loss_fn, lr=1e-3, device=device)
    history = trainer.fit(k1_epochs=k1_epochs, k2_iters=k2_iters, desc="[SteadyState]",
                           validate_fn=validate_fn, checkpoint_path=checkpoint_path,
                           meta=meta, save_every=save_every, resume=resume)
    return trainer.model, history


def plot_steady_state_profiles(model, physics, out_path="plots/pde/incompressible_pde_steady_state.png"):
    """Fig. 8 style: PINC-SS vs. exact analytic steady-state profile for
    several control values."""
    us = _VAL_US
    x = torch.linspace(0, 1, 25).unsqueeze(-1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for u_val in us:
        u = torch.full_like(x, u_val)
        with torch.no_grad():
            y = model(x, u)
        P_pinc, V_pinc = y[:, 0], y[:, 1]

        V_exact = _steady_velocity_for_control(physics, u_val)
        P_exact = physics.pressure_profile(x[:, 0], V_exact, torch.tensor(u_val))

        axes[0].plot(x[:, 0], P_exact, "--", color="gray")
        axes[0].plot(x[:, 0], P_pinc, "o", markersize=3, label=f"u={u_val}")
        axes[1].axhline(V_exact.item(), linestyle="--", color="gray")
        axes[1].plot(x[:, 0], V_pinc, "o", markersize=3, label=f"u={u_val}")

    axes[0].set_ylabel(r"$\tilde{P}$")
    axes[0].set_title("Steady-state pressure profile (dots: PINC-SS, dashed: exact)")
    axes[0].legend(fontsize=8, ncol=5)
    axes[1].set_ylabel(r"$\tilde{V}$")
    axes[1].set_xlabel(r"$\tilde{x}$")
    axes[1].set_title("Steady-state velocity profile")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    mape_p, mape_v = _steady_state_mape(model, physics, us, x)
    print(f"Steady-state MAPE -- pressure: {mape_p:.2f}%, velocity: {mape_v:.2f}%")


def _steady_state_mape(model, physics, us, x):
    errs_p, errs_v = [], []
    for u_val in us:
        u = torch.full_like(x, u_val)
        with torch.no_grad():
            y = model(x, u)
        P_pinc, V_pinc = y[:, 0], y[:, 1]
        V_exact = _steady_velocity_for_control(physics, u_val)
        P_exact = physics.pressure_profile(x[:, 0], V_exact, torch.tensor(u_val))
        errs_p.append(((P_pinc - P_exact).abs() / P_exact.abs().clamp(min=1e-6)).mean().item())
        errs_v.append(abs((V_pinc.mean().item() - V_exact.item()) / max(abs(V_exact.item()), 1e-6)))
    return 100 * sum(errs_p) / len(errs_p), 100 * sum(errs_v) / len(errs_v)


# ----------------------------------------------------------------------
# Transient stage
# ----------------------------------------------------------------------
def train_transient(physics, steady_state_model, hidden=32, depth=3,
                     k1_epochs=400, k2_iters=400,
                     n_collocation=4000, n_boundary=1000, n_ic=1000, device="cpu",
                     checkpoint_path=None, resume=False, save_every=100, x_bar=0.1):
    """
    checkpoint_path, resume, save_every : see `train_steady_state`.
    x_bar : measurement position used to compute the open-loop
            forward-simulation MSE tracked as this stage's validation
            metric (see `_open_loop_mse`).
    """
    meta = {"hidden": hidden, "depth": depth}

    model = PINCTransientPDE(MLP(in_dim=4, out_dim=2, hidden=hidden, depth=depth)).to(device)
    sampler = TransientPDESampler(u_range=U_RANGE, device=device)
    loss_fn = TransientPDELoss(physics, steady_state_model)

    def sample_batches():
        return (sampler.sample_collocation(n_collocation),
                sampler.sample_boundary(n_boundary),
                sampler.sample_initial_condition(n_ic))

    def validate_fn(m):
        return _open_loop_mse(m, physics, x_bar=x_bar)

    trainer = PDETrainer(model, sample_batches, loss_fn, lr=1e-3, device=device)
    history = trainer.fit(k1_epochs=k1_epochs, k2_iters=k2_iters, desc="[Transient]",
                           validate_fn=validate_fn, checkpoint_path=checkpoint_path,
                           meta=meta, save_every=save_every, resume=resume)
    return trainer.model, history


def plot_training_curves(history, title, out_path):
    plt.figure(figsize=(8, 5))
    for key in ("physics", "bc", "ic"):
        if key in history:
            plt.semilogy(history[key], label=key)
    plt.xlabel("Iteration (ADAM epochs then L-BFGS iters)")
    plt.ylabel("MSE (log scale)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _open_loop_forward_simulation(transient_model, physics, x_bar=0.1):
    """
    Shared by `plot_open_loop_forward_simulation` (plotting) and
    `_open_loop_mse` (validation metric): chains the transient net
    across several windows (Algorithm 1, no autoregressive feedback --
    each window's prediction depends only on (u0, u), see
    PINCTransientPDE docstring) and compares against the exact plant.

    returns (t_axis, u_windows, P_true, P_pinc)
    """
    u_windows = torch.tensor([0.7, 0.6, 0.5, 0.4, 0.3])
    n_windows = len(u_windows)

    plant = IncompressiblePipePlant(physics)
    V0 = _steady_velocity_for_control(physics, u_windows[0].item())
    V_true = simulate(plant, V0.unsqueeze(-1), u_windows.unsqueeze(-1), dt=1.0, substeps=20)[:, 0]
    P_true = torch.stack([
        physics.pressure_profile(torch.tensor(x_bar), V_true[i],
                                  u_windows[max(i - 1, 0)])
        for i in range(n_windows + 1)
    ])

    u0_seq = torch.cat([u_windows[:1], u_windows[:-1]])
    with torch.no_grad():
        y_pinc = transient_model.at_position(x_bar, u0_seq.unsqueeze(-1), u_windows.unsqueeze(-1))
    P_pinc = torch.cat([P_true[:1], y_pinc[:, 0]])

    t_axis = torch.arange(n_windows + 1)
    return t_axis, u_windows, P_true, P_pinc


def _open_loop_mse(transient_model, physics, x_bar=0.1):
    """Scalar validation metric (Eq. 13-style) used by `train_transient`
    to track the best transient net seen during training."""
    _, _, P_true, P_pinc = _open_loop_forward_simulation(transient_model, physics, x_bar=x_bar)
    return torch.mean((P_pinc - P_true) ** 2).item()


def plot_open_loop_forward_simulation(transient_model, physics, x_bar=0.1,
                                       out_path="plots/pde/incompressible_pde_transient_openloop.png"):
    """Fig. 10 style plot of `_open_loop_forward_simulation`'s result."""
    t_axis, u_windows, P_true, P_pinc = _open_loop_forward_simulation(transient_model, physics, x_bar=x_bar)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_axis, P_true, "k-", label="exact plant")
    ax.plot(t_axis, P_pinc, "o", color="tab:blue", label="PINC-Transient")
    ax.step(t_axis[:-1], u_windows, where="post", color="gray", alpha=0.5, label="control u")
    ax.set_xlabel("window index k")
    ax.set_ylabel(r"$\tilde{P}(\bar{x}, t)$")
    ax.set_title("Open-loop forward simulation (PDE-PINC), no feedback across windows")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    mse = torch.mean((P_pinc - P_true) ** 2).item()
    print(f"Open-loop transient forward-simulation MSE (pressure @ x_bar): {mse:.3e}")


# ----------------------------------------------------------------------
# MPC control stage
# ----------------------------------------------------------------------
def run_mpc_demo(transient_model, physics, x_bar=0.1, n_steps=30,
                  out_path="plots/pde/incompressible_pde_mpc_control.png"):
    """Fig. 11 style: drive the PDG pressure down towards an unattainable
    (production-maximizing) target, subject to a rate constraint."""
    plant = IncompressiblePipePlant(physics)
    plant_step = rk4_control_interface(plant, T=1.0, substeps=20)

    target = torch.tensor(0.0)  # unattainable low target -> maximize drawdown
    dy_max = 0.08  # normalized per-window rate limit

    V_hist, P_hist, u_hist = run_pde_mpc_simulation(
        transient_model, plant_step, physics, x_bar=x_bar, target=target,
        u0_init=0.9, n_steps=n_steps, Np=6, Nc=2, lambda_smooth=0.05,
        dy_max=dy_max, u_min=U_RANGE[0], u_max=U_RANGE[1], maxiter=50,
    )

    t_axis = torch.arange(n_steps + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t_axis, P_hist, "b-", label=r"$\tilde{P}(\bar{x})$ (PDG)")
    axes[0].step(t_axis[1:], u_hist, where="post", color="tab:red", label="manipulated u")
    axes[0].axhline(target.item(), color="k", linestyle=":", label="target")
    axes[0].set_ylabel("normalized pressure")
    axes[0].legend(fontsize=8)
    axes[0].set_title("MPC control of the PDG pressure (PDE-PINC as predictive model)")

    dP = P_hist[1:] - P_hist[:-1]
    axes[1].plot(t_axis[1:], dP, "b-")
    axes[1].axhline(dy_max, color="orange", linestyle="--", label="rate limit")
    axes[1].axhline(-dy_max, color="orange", linestyle="--")
    axes[1].set_ylabel(r"$\Delta \tilde{P}$ per window")
    axes[1].set_xlabel("window index k")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k1", type=int, default=25000, help="ADAM epochs (both stages)")
    parser.add_argument("--k2", type=int, default=2500, help="L-BFGS iters (both stages)")
    parser.add_argument("--checkpoint-steady-state", default="checkpoints/pde_steady_state.pt",
                         help="path to save/load the steady-state training checkpoint")
    parser.add_argument("--checkpoint-transient", default="checkpoints/pde_transient.pt",
                         help="path to save/load the transient training checkpoint")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                         help="resume training from the checkpoints above if they exist (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                         help="ignore any existing checkpoints and train from scratch")
    parser.add_argument("--no-checkpoint", action="store_true",
                         help="disable checkpointing entirely")
    parser.add_argument("--load-only", action="store_true",
                         help="skip training entirely and load both trained models from "
                              "--checkpoint-steady-state / --checkpoint-transient (e.g. for "
                              "re-running the plots/MPC demo only)")
    args = parser.parse_args()

    physics = IncompressibleFlowPhysics(IncompressibleFlowParams())

    if args.load_only:
        print(f"Loading trained models from '{args.checkpoint_steady_state}' and "
              f"'{args.checkpoint_transient}' (skipping training)...")
        ss_model, ss_payload = load_pinc_steady_state_pde_model(
            args.checkpoint_steady_state, map_location="cpu")
        tr_model, tr_payload = load_pinc_transient_pde_model(
            args.checkpoint_transient, map_location="cpu")
        ss_history = ss_payload["extra"].get("history")
        tr_history = tr_payload["extra"].get("history")
    else:
        ss_checkpoint = None if args.no_checkpoint else args.checkpoint_steady_state
        tr_checkpoint = None if args.no_checkpoint else args.checkpoint_transient

        print("Training PINC-SteadyState...")
        ss_model, ss_history = train_steady_state(
            physics, k1_epochs=args.k1, k2_iters=args.k2, device=args.device,
            checkpoint_path=ss_checkpoint, resume=args.resume)

        print("Training PINC-Transient (using frozen steady-state net for IC targets)...")
        ss_model_cpu = ss_model.to("cpu")
        tr_model, tr_history = train_transient(
            physics, ss_model_cpu, k1_epochs=args.k1, k2_iters=args.k2, device=args.device,
            checkpoint_path=tr_checkpoint, resume=args.resume)

    if ss_history is not None:
        plot_training_curves(ss_history, "PDE-PINC steady-state training",
                              "plots/pde/incompressible_pde_steady_state_training.png")
    plot_steady_state_profiles(ss_model.to("cpu"), physics)

    if tr_history is not None:
        plot_training_curves(tr_history, "PDE-PINC transient training",
                              "plots/pde/incompressible_pde_transient_training.png")

    tr_model = tr_model.to("cpu")
    plot_open_loop_forward_simulation(tr_model, physics)

    print("Running MPC demo...")
    run_mpc_demo(tr_model, physics)

    print("\nSaved figures: incompressible_pde_steady_state_training.png, "
          "incompressible_pde_steady_state.png, incompressible_pde_transient_training.png, "
          "incompressible_pde_transient_openloop.png, incompressible_pde_mpc_control.png")


if __name__ == "__main__":
    main()