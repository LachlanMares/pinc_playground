"""
Cart-pole swing-up with PINC-ODE-MPC.

Mirrors `pinc.training.train_vanderpol`'s structure end to end:

  1) Train a PINC net (sin/cos-encoded, see
     `pinc.models.pinc_cartpole.CartPolePINCModel`) for the cart-pole.
  2) Long-range self-loop prediction diagnostic on a random-switching
     control input.
  3) Generate an offline swing-up reference trajectory (direct
     trajectory optimization, see
     `pinc.datasets.cartpole.generate_swingup_reference`), then run
     closed-loop NMPC *tracking* that reference, comparing the trained
     PINC net as predictive model against the true ODE/RK4 baseline
     (both driving the same, unmodified `NMPCController`).
  4) Animate the PINC-driven closed-loop swing-up (cart + pole
     schematic, with the applied control force plotted alongside).

Run with:  python -m pinc.training.train_cartpole
Resume:    python -m pinc.training.train_cartpole --resume
GPU:       python -m pinc.training.train_cartpole --device cuda
"""
import argparse
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

from pinc.physics.cartpole import CartPole
from pinc.nn.mlp import MLP
from pinc.models.pinc_cartpole import CartPolePINCModel, load_cartpole_pinc_model
from pinc.core.trainer import Trainer
from pinc.losses.pinc_loss import PINCLoss
from pinc.datasets.cartpole import (make_cartpole_sampler, random_control_signal,
                                     generate_swingup_reference)
from pinc.evaluation.rollout import pinc_rollout, mse_gen
from pinc.simulation.rk4 import simulate, rk4_control_interface
from pinc.control.nmpc import run_nmpc_simulation
from pinc.visualization.cartpole_animation import animate_cartpole


def build_validation_trajectory(physics: CartPole, T, n_steps=40, seed=0):
    """
    A modest-amplitude random-control trajectory used to track
    generalization during training (Eq. 13's validate_fn). Unlike Van
    der Pol (globally bounded), the cart-pole is an unstable/
    underactuated system, so the validation input is kept mild (small
    control range, moderate horizon, starting near the downward rest
    position) -- large enough to be a meaningful generalization check,
    but not so aggressive it sends the "ground truth" RK4 trajectory
    somewhere so extreme the untrained-or-partially-trained network
    has no hope of tracking it early in training (which would make
    `validate_fn` uselessly noisy for early-epoch model selection).
    """
    torch.manual_seed(seed)
    y0 = torch.tensor([0.0, 0.0, np.pi - 0.2, 0.1])
    u_seq = random_control_signal(n_steps, control_dim=1, u_range=(-3.0, 3.0), seed=seed)
    y_true = simulate(physics, y0, u_seq, dt=T, substeps=20)
    return y0, u_seq, y_true


def build_easy_validation_trajectory(physics: CartPole, T, n_steps=20, u_const=0.0):
    """Simpler long-range diagnostic: constant (default zero) control
    starting from the same near-hanging-rest state, mirroring
    `train_vanderpol.py`'s "easy" free-response diagnostic."""
    y0 = torch.tensor([0.0, 0.0, np.pi - 0.2, 0.0])
    u_seq = torch.full((n_steps, physics.control_dim), u_const)
    y_true = simulate(physics, y0, u_seq, dt=T, substeps=20)
    return y0, u_seq, y_true


def train_pinc(physics: CartPole, T, k1_epochs=500, k2_iters=2000, hidden=64, depth=4,
               n_boundary=4000, n_collocation=100000,
               n_multistep=4000, multistep_k=3, lambda_phys=1.0,
               device="cpu", checkpoint_path=None, resume=False, save_every=100):
    """
    Same shape as `train_vanderpol.train_pinc`; the only cart-pole-
    specific pieces are the model class (`CartPolePINCModel` instead of
    the base `PINCModel`) and a wider default backbone (`hidden=64`
    instead of Van der Pol's `hidden=20`) -- the cart-pole's dynamics
    are a meaningfully harder function to fit (4 states instead of 2,
    strongly nonlinear sin/cos coupling terms, and a much larger
    excursion range for a genuine swing-up), so it benefits from more
    network capacity than the 2-state oscillator does.
    """
    meta = {
        "state_dim": physics.state_dim,
        "control_dim": physics.control_dim,
        "T": T,
        "hidden": hidden,
        "depth": depth,
    }

    model = CartPolePINCModel(
        backbone=MLP(in_dim=1 + 5 + 1, out_dim=4, hidden=hidden, depth=depth),
        T=T,
    )

    sampler = make_cartpole_sampler(physics, T, device=device)
    loss_fn = PINCLoss(physics, T=T, lambda_phys=lambda_phys,
                        lambda_endpoint=1.0, lambda_multistep=1.0)

    y0_val, u_val, y_true_val = build_validation_trajectory(physics, T)

    def validate_fn(m):
        return mse_gen(m, y0_val, u_val, y_true_val)

    trainer = Trainer(model, sampler, loss_fn,
                       n_boundary=n_boundary, n_collocation=n_collocation,
                       n_multistep=n_multistep, multistep_k=multistep_k,
                       lr=1e-3, device=device)

    history = trainer.fit(k1_epochs=k1_epochs, k2_iters=k2_iters,
                           validate_fn=validate_fn,
                           checkpoint_path=checkpoint_path, meta=meta,
                           save_every=save_every, resume=resume)

    return trainer.model, history


def plot_training_curves(history):
    plt.figure(figsize=(10, 5))
    plt.semilogy(history["total"], label="Total")
    plt.semilogy(history["data"], label="Data (MSE_y)")
    plt.semilogy(history["physics"], label="Physics (MSE_F)")
    plt.semilogy(history["endpoint"], label="Endpoint (MSE, t=T vs RK4)")
    plt.semilogy(history["multistep"], label="Multistep (chained rollout vs RK4)")
    val = [v for v in history["val"] if v is not None]
    if val:
        plt.semilogy(range(len(history["total"]) - len(val), len(history["total"])), val, label="Validation")
    plt.xlabel("Iteration (ADAM epochs then L-BFGS iters)")
    plt.ylabel("MSE (log scale)")
    plt.title("PINC training loss - Cart-pole")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/cartpole/cartpole_training_curves.png", dpi=150)
    plt.close()


def plot_long_range_prediction(model, physics, T):
    _plot_rollout_diagnostic(
        model, physics, T,
        *build_validation_trajectory(physics, T, n_steps=40, seed=1),
        title="PINC self-loop vs true (random-switching input)",
        out_path="plots/cartpole/cartpole_long_range_prediction.png",
    )
    _plot_rollout_diagnostic(
        model, physics, T,
        *build_easy_validation_trajectory(physics, T, n_steps=40, u_const=0.0),
        title="PINC self-loop vs true (constant u=0, free response)",
        out_path="plots/cartpole/cartpole_long_range_prediction_easy.png",
    )


def _plot_rollout_diagnostic(model, physics, T, y0, u_seq, y_true, title, out_path):
    """Four-state-panel + control-input diagnostic (state labels swapped
    in for Van der Pol's generic x1/x2, otherwise the exact same
    "true alone / pred alone / overlaid / abs error" structure as
    `train_vanderpol._plot_rollout_diagnostic`)."""
    y_pred = pinc_rollout(model, y0, u_seq).detach()
    t_axis = torch.arange(y_pred.shape[0]) * T

    labels = ["x (m)", "x_dot (m/s)", "theta (rad)", "theta_dot (rad/s)"]
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    for i in range(4):
        axes[i].plot(t_axis, y_true[:, i], "-", color="black", linewidth=1.5, label="true (RK4)")
        axes[i].plot(t_axis, y_pred[:, i], "o", color="tab:orange", markersize=4, label="PINC self-loop")
        axes[i].set_ylabel(labels[i])
        axes[i].legend(fontsize=8, loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    mse = torch.mean((y_pred - y_true) ** 2).item()
    print(f"[{out_path}] Long-range self-loop generalization MSE: {mse:.3e}")


def integral_metrics(y_ref, y):
    err = (y_ref - y).abs()
    iae = err.sum().item()
    rmse = torch.sqrt(torch.mean((y_ref - y) ** 2)).item()
    return rmse, iae


def _solve_nmpc_worker(kind, model, physics, T, y0, y_ref, control_dim,
                        N1, N2, Nu, Q, R, u_min, u_max, maxiter):
    """Same structure as `train_vanderpol._solve_nmpc_worker`: `kind`
    selects whether the trained PINC net or a fresh RK4 predictive
    interface drives the NMPC controller; the plant (ground truth) is
    always RK4."""
    if kind == "pinc":
        control_interface = model.step
    else:
        control_interface = rk4_control_interface(physics, T, substeps=20)
    plant = rk4_control_interface(physics, T, substeps=20)

    t0 = time.time()
    y, u = run_nmpc_simulation(
        control_interface=control_interface,
        plant_step=plant,
        y0=y0, y_ref_full=y_ref, control_dim=control_dim,
        N1=N1, N2=N2, Nu=Nu, Q=Q, R=R, u_min=u_min, u_max=u_max,
        maxiter=maxiter,
        desc="PINC" if kind == "pinc" else "ODE/RK",
    )
    elapsed = time.time() - t0
    return y, u, elapsed


def run_control_experiment(model, physics, T, N_ref=250):
    """
    Generates the swing-up reference trajectory, then runs closed-loop
    NMPC tracking it with both the PINC net and the RK4/ODE baseline as
    predictive model (reusing `NMPCController`/`run_nmpc_simulation`
    completely unmodified, exactly as `nmpc_pde.py`'s module docstring
    notes is possible for the ODE case).
    """
    model = model.to("cpu")

    print("Solving offline swing-up reference trajectory...")
    u_ref, y_ref = generate_swingup_reference(physics, T, N=N_ref)

    y0 = torch.tensor([0.0, 0.0, np.pi, 0.0])  # start hanging down, at rest

    Q = torch.diag(torch.tensor([2.0, 1.0, 40.0, 4.0]))
    R = torch.diag(torch.tensor([0.01]))

    N1, N2, Nu = 1, 20, 10
    u_min, u_max = [-15.0], [15.0]

    common = dict(physics=physics, T=T, y0=y0, y_ref=y_ref[1:], control_dim=1,
                  N1=N1, N2=N2, Nu=Nu, Q=Q, R=R, u_min=u_min, u_max=u_max,
                  maxiter=30)

    print("Running PINC NMPC (tracking the swing-up reference)...")
    y_pinc, u_pinc, t_pinc = _solve_nmpc_worker("pinc", model, **common)

    print("Running RK4 NMPC (tracking the swing-up reference)...")
    y_ode, u_ode, t_ode = _solve_nmpc_worker("ode", model, **common)

    rmse_pinc, iae_pinc = integral_metrics(y_ref[1:], y_pinc[1:])
    rmse_ode, iae_ode = integral_metrics(y_ref[1:], y_ode[1:])

    print("\nControl performance (cart-pole swing-up, reference-tracking NMPC):")
    print(f"{'Model':<10}{'RMSE':>10}{'IAE':>10}{'time(s)':>12}")
    print(f"{'PINC':<10}{rmse_pinc:>10.3f}{iae_pinc:>10.2f}{t_pinc:>12.3f}")
    print(f"{'ODE/RK':<10}{rmse_ode:>10.3f}{iae_ode:>10.2f}{t_ode:>12.3f}")

    t_axis = torch.arange(N_ref + 1) * T

    labels = ["x (m)", "x_dot (m/s)", "theta (rad)", "theta_dot (rad/s)"]
    fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
    for i in range(4):
        axes[i].plot(t_axis, y_ref[:, i], "k:", linewidth=1.5, label="reference")
        axes[i].plot(t_axis, y_pinc[:, i], color="tab:orange", label=f"{labels[i]} (PINC)")
        axes[i].plot(t_axis, y_ode[:, i], "--", color="tab:green", label=f"{labels[i]} (ODE/RK)")
        axes[i].set_ylabel(labels[i])
        axes[i].legend(fontsize=8, loc="upper right")
    axes[0].set_title("NMPC tracking of the cart-pole swing-up reference")

    axes[4].step(t_axis[:-1], u_pinc[:, 0], where="post", color="tab:orange", label="u (PINC)")
    axes[4].step(t_axis[:-1], u_ode[:, 0], where="post", linestyle="--", color="tab:green", label="u (ODE/RK)")
    axes[4].set_ylabel("control u (N)")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("plots/cartpole/cartpole_nmpc_control.png", dpi=150)
    plt.close()

    return y_pinc, u_pinc, y_ref, u_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         help="training device, e.g. 'cpu', 'cuda', 'cuda:0' (default: auto-detect)")
    parser.add_argument("--checkpoint", default="checkpoints/cartpole.pt",
                         help="path to save/load the training checkpoint")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                         help="resume training from --checkpoint if it exists (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                         help="ignore any existing checkpoint and train from scratch")
    parser.add_argument("--no-checkpoint", action="store_true",
                         help="disable checkpointing entirely")
    parser.add_argument("--load-only", default=None,
                         help="skip training entirely and load a trained model from this "
                              "checkpoint path (e.g. for re-running the control experiment only)")
    parser.add_argument("--k1", type=int, default=8000, help="number of ADAM epochs")
    parser.add_argument("--k2", type=int, default=2000, help="number of L-BFGS iterations")
    parser.add_argument("--no-animation", action="store_true",
                         help="skip generating the cartpole_swingup.gif animation")
    args = parser.parse_args()

    physics = CartPole(M=1.0, m=0.1, L=1.0, g=9.8)
    T = 0.02

    if args.load_only is not None:
        print(f"Loading trained model from '{args.load_only}' (skipping training)...")
        model, payload = load_cartpole_pinc_model(args.load_only, map_location="cpu")
        history = payload["extra"].get("history")
    else:
        print(f"Training PINC net for the cart-pole on device='{args.device}'...")
        checkpoint_path = None if args.no_checkpoint else args.checkpoint
        model, history = train_pinc(physics, T, k1_epochs=args.k1, k2_iters=args.k2,
                                     device=args.device,
                                     checkpoint_path=checkpoint_path,
                                     resume=args.resume)

    if history is not None:
        plot_training_curves(history)
    plot_long_range_prediction(model, physics, T)
    y_pinc, u_pinc, y_ref, u_ref = run_control_experiment(model, physics, T)

    print("\nSaved figures: cartpole_training_curves.png, "
          "cartpole_long_range_prediction.png, cartpole_long_range_prediction_easy.png, "
          "cartpole_nmpc_control.png")

    if not args.no_animation:
        animate_cartpole(y_pinc, u_pinc, T, physics,
                          out_path="cartpole_swingup.gif",
                          title="Cart-pole swing-up -- PINC-driven NMPC")
        print("Saved animation: cartpole_swingup.gif")


if __name__ == "__main__":
    main()