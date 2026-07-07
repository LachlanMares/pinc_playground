"""
Reproduces the four-tank system experiments of Section 4.2 of Antonelo
et al. (arXiv:2104.02556):

  1) Train a PINC net (5 hidden layers, 20 neurons, T = 10s) for the
     quadruple-tank process.
  2) Long-range self-loop prediction on a random control signal (Fig. 12).
  3) Closed-loop NMPC regulating h1, h2 to setpoints, with h3, h4
     constrained to [0.6, 5.5] cm (Fig. 13), compared against the
     ODE/RK4 baseline predictive model (Fig. 14, Table 1).

Run with:  python -m pinc.training.train_fourtank
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import torch
import matplotlib.pyplot as plt

from pinc.physics.fourtank import FourTank
from pinc.nn.mlp import MLP
from pinc.models.pinc import PINCModel
from pinc.core.trainer import Trainer
from pinc.losses.pinc_loss import PINCLoss
from pinc.datasets.fourtank import make_fourtank_sampler
from pinc.evaluation.rollout import pinc_rollout, mse_gen
from pinc.simulation.rk4 import simulate, rk4_control_interface
from pinc.control.nmpc import run_nmpc_simulation
from pinc.utils.checkpoint import load_pinc_model


def random_control_signal(n_steps, control_dim=2, u_range=(0.0, 5.0), seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.empty(n_steps, control_dim).uniform_(*u_range)


def build_validation_trajectory(physics, T, n_steps=35, seed=0):
    torch.manual_seed(seed)
    y0 = torch.tensor([8.0, 8.0, 8.0, 8.0])
    u_seq = random_control_signal(n_steps, control_dim=2, seed=seed)
    y_true = simulate(physics, y0, u_seq, dt=T, substeps=20)
    return y0, u_seq, y_true


def train_pinc(physics, T, k1_epochs=500, k2_iters=2000, hidden=20, depth=5,
               n_boundary=1000, n_collocation=100000, lambda_phys=1.0,
               device="cpu", checkpoint_path=None, resume=False, save_every=100):
    """
    device          : "cpu", "cuda", or "cuda:N" -- see the note in
                       train_vanderpol.py's train_pinc; the four-tank
                       net (5 x 20) is still small, but with
                       n_collocation=100000 points per iteration a GPU
                       can meaningfully speed up training.
    checkpoint_path : if given, periodically saves training progress
                       here so a killed/interrupted run can be resumed.
    resume          : if True and a checkpoint already exists at
                       `checkpoint_path`, restores model/optimizer state
                       and continues training instead of starting over.
    """

    meta = {
        "state_dim": physics.state_dim,
        "control_dim": physics.control_dim,
        "T": T,
        "hidden": hidden,
        "depth": depth,
    }

    model = PINCModel(
        backbone=MLP(in_dim=1 + physics.state_dim + physics.control_dim,
                     out_dim=physics.state_dim, hidden=hidden, depth=depth),
        state_dim=physics.state_dim,
        control_dim=physics.control_dim,
        T=T,
    )

    sampler = make_fourtank_sampler(physics, T)
    loss_fn = PINCLoss(physics, lambda_phys=lambda_phys)

    y0_val, u_val, y_true_val = build_validation_trajectory(physics, T)

    def validate_fn(m):
        return mse_gen(m, y0_val, u_val, y_true_val)

    trainer = Trainer(model, sampler, loss_fn,
                       n_boundary=n_boundary, n_collocation=n_collocation,
                       lr=1e-3, device=device)

    history = trainer.fit(k1_epochs=k1_epochs, k2_iters=k2_iters,
                           validate_fn=validate_fn, log_every=max(1, k1_epochs // 10),
                           checkpoint_path=checkpoint_path, meta=meta,
                           save_every=save_every, resume=resume)

    return trainer.model, history


def plot_training_curves(history):
    plt.figure(figsize=(10, 5))
    plt.semilogy(history["total"], label="Total")
    plt.semilogy(history["data"], label="Data (MSE_y)")
    plt.semilogy(history["physics"], label="Physics (MSE_F)")
    plt.xlabel("Iteration")
    plt.ylabel("MSE (log scale)")
    plt.title("PINC training loss - Four tanks")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("fourtank_training_curves.png", dpi=150)
    plt.close()


def plot_long_range_prediction(model, physics, T):
    y0, u_seq, y_true = build_validation_trajectory(physics, T, n_steps=35, seed=1)
    y_pred = pinc_rollout(model, y0, u_seq)
    t_axis = torch.arange(y_pred.shape[0]) * T

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(t_axis, y_true[:, 0], "k-", label="true h1")
    axes[0].plot(t_axis, y_true[:, 1], "k--", label="true h2")
    axes[0].plot(t_axis, y_pred[:, 0].detach(), "o-", color="tab:blue", label="PINC h1")
    axes[0].plot(t_axis, y_pred[:, 1].detach(), "o-", color="tab:pink", label="PINC h2")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("h1, h2")
    axes[0].legend()

    axes[1].plot(t_axis, y_true[:, 2], "k-", label="true h3")
    axes[1].plot(t_axis, y_true[:, 3], "k--", label="true h4")
    axes[1].plot(t_axis, y_pred[:, 2].detach(), "o-", color="tab:blue", label="PINC h3")
    axes[1].plot(t_axis, y_pred[:, 3].detach(), "o-", color="tab:pink", label="PINC h4")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("h3, h4")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("fourtank_long_range_prediction.png", dpi=150)
    plt.close()

    mse = torch.mean((y_pred - y_true) ** 2).item()
    print(f"Long-range self-loop generalization MSE: {mse:.3e}")


def integral_metrics(y_ref, y):
    err = (y_ref - y).abs()
    iae = err.sum().item()
    rmse = torch.sqrt(torch.mean((y_ref - y) ** 2)).item()
    return rmse, iae


def _solve_nmpc_worker(kind, model, physics, T, y0, y_ref, control_dim,
                        N1, N2, Nu, Q, R, u_min, u_max, state_constraints, maxiter):
    """
    Runs a single closed-loop NMPC simulation in its own process.

    `kind` selects which predictive model drives the controller:
      - "pinc" : the trained PINC net (`model`, already CPU-resident)
      - "ode"  : a fresh RK4 predictive interface (cheaper substep count)

    The plant (ground-truth RK4 integrator) and, for "ode", the
    predictive interface itself are rebuilt here rather than passed in,
    since the nested closures `rk4_control_interface` returns aren't
    reliably picklable across a process boundary.

    torch.set_num_threads(1) keeps each worker from spinning up its own
    intra-op thread pool -- with tensors this small (batch size 1)
    that overhead is pure contention on top of the process-level
    parallelism, not useful compute.
    """
    torch.set_num_threads(1)

    plant = rk4_control_interface(physics, T, substeps=20)
    control_interface = model.step if kind == "pinc" else rk4_control_interface(physics, T, substeps=5)

    t0 = time.time()
    y, u = run_nmpc_simulation(
        control_interface=control_interface,
        plant_step=plant,
        y0=y0, y_ref_full=y_ref, control_dim=control_dim,
        N1=N1, N2=N2, Nu=Nu, Q=Q, R=R, u_min=u_min, u_max=u_max,
        state_constraints=state_constraints,
        maxiter=maxiter, warm_start=True,
        desc="PINC" if kind == "pinc" else "ODE/RK", position=0 if kind == "pinc" else 1,
    )
    elapsed = time.time() - t0
    return y, u, elapsed


def run_control_experiment(model, physics, T, n_steps=45):
    """Fig. 13/14, Table 1 style experiment: regulate h1, h2 to a step
    reference while keeping h3, h4 within [0.6, 5.5] cm.

    Compared to the first pass, this version fixes two issues that were
    causing the controller to get stuck at a physically-consistent but
    wrong steady state (u1 pinned at its bound, u2 never moving):

    - N2 is increased from 5 to 15 steps (150s lookahead instead of
      50s). The four-tank system has time constants on the order of a
      few hundred seconds, so a 50s horizon simply couldn't "see" the
      delayed payoff of raising u2 to indirectly fill tank 1 through
      the cross-coupled valve -- the exact non-minimum-phase behavior
      this benchmark is meant to exercise.
    - Warm-starting is enabled (default in run_nmpc_simulation): each
      timestep's solve is initialized from the shifted solution of the
      previous timestep rather than from zero, which matters a lot
      when a control channel's benefit is delayed and the per-solve
      iteration budget (maxiter) is limited.

    The predictive-model rollout inside NMPC uses fewer RK4 substeps
    than the "true plant" simulation (5 vs 20) purely for tractable
    runtime; this only affects the internal prediction accuracy used
    for optimization, not the fidelity of the simulated closed loop.

    NMPC solves go through scipy.optimize (CPU-only, numpy-backed), so
    the model is moved to CPU here regardless of what device it was
    trained on.
    """
    model = model.to("cpu")

    y_ref = torch.zeros(n_steps, 4)
    y_ref[:, 0] = 10.0
    y_ref[:, 1] = 14.0
    y_ref[n_steps // 2:, 0] = 8.0
    y_ref[n_steps // 2:, 1] = 12.5

    y0 = torch.tensor([2.0, 2.0, 2.0, 2.0])

    Q = torch.diag(torch.tensor([10.0, 10.0, 0.0, 0.0]))
    R = torch.diag(torch.tensor([1.0, 1.0]))

    N1, N2, Nu = 1, 15, 5
    u_min, u_max = [0.0, 0.0], [5.0, 5.0]
    state_constraints = [(2, 0.6, 5.5), (3, 0.6, 5.5)]

    # The PINC-driven and ODE/RK-driven NMPC runs are fully independent
    # (same plant, same reference, no shared mutable state), so run them
    # in separate processes instead of back-to-back -- each one otherwise
    # pins a single core for the whole simulation.
    common = dict(physics=physics, T=T, y0=y0, y_ref=y_ref, control_dim=2,
                  N1=N1, N2=N2, Nu=Nu, Q=Q, R=R, u_min=u_min, u_max=u_max,
                  state_constraints=state_constraints, maxiter=80)

    with ProcessPoolExecutor(max_workers=2) as ex:
        fut_pinc = ex.submit(_solve_nmpc_worker, "pinc", model, **common)
        fut_ode = ex.submit(_solve_nmpc_worker, "ode", model, **common)
        y_pinc, u_pinc, t_pinc = fut_pinc.result()
        y_ode, u_ode, t_ode = fut_ode.result()

    rmse_pinc, iae_pinc = integral_metrics(y_ref[:, :2], y_pinc[1:, :2])
    rmse_ode, iae_ode = integral_metrics(y_ref[:, :2], y_ode[1:, :2])

    def violation(y):
        h3, h4 = y[1:, 2], y[1:, 3]
        lo_viol = torch.clamp(0.6 - h3, min=0).sum() + torch.clamp(0.6 - h4, min=0).sum()
        hi_viol = torch.clamp(h3 - 5.5, min=0).sum() + torch.clamp(h4 - 5.5, min=0).sum()
        return (lo_viol + hi_viol).item()

    print("\nControl performance (four tanks, Table 1 style):")
    print(f"{'Model':<10}{'RMSE':>10}{'IAE':>10}{'time(s)':>12}{'h3/h4 viol':>14}")
    print(f"{'PINC':<10}{rmse_pinc:>10.3f}{iae_pinc:>10.2f}{t_pinc:>12.3f}{violation(y_pinc):>14.3f}")
    print(f"{'ODE/RK':<10}{rmse_ode:>10.3f}{iae_ode:>10.2f}{t_ode:>12.3f}{violation(y_ode):>14.3f}")

    t_axis = torch.arange(n_steps + 1) * T

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(t_axis, y_pinc[:, 0], label="h1 (PINC)")
    axes[0].plot(t_axis, y_pinc[:, 1], label="h2 (PINC)")
    axes[0].plot(t_axis, y_ode[:, 0], "--", color="olive", label="h1 (ODE/RK)")
    axes[0].plot(t_axis, y_ode[:, 1], "--", color="darkkhaki", label="h2 (ODE/RK)")
    axes[0].plot(t_axis, torch.cat([y_ref[:, 0], y_ref[-1:, 0]]), "k:", label="ref h1")
    axes[0].plot(t_axis, torch.cat([y_ref[:, 1], y_ref[-1:, 1]]), "k-.", label="ref h2")
    axes[0].set_ylabel("controlled levels")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title("NMPC control of the four-tank system")

    axes[1].plot(t_axis, y_pinc[:, 2], label="h3 (PINC)")
    axes[1].plot(t_axis, y_pinc[:, 3], label="h4 (PINC)")
    axes[1].plot(t_axis, y_ode[:, 2], "--", color="olive", label="h3 (ODE/RK)")
    axes[1].plot(t_axis, y_ode[:, 3], "--", color="darkkhaki", label="h4 (ODE/RK)")
    axes[1].axhline(0.6, color="grey", linestyle=":")
    axes[1].axhline(5.5, color="grey", linestyle=":")
    axes[1].set_ylabel("constrained levels")
    axes[1].legend(fontsize=7, ncol=2)

    axes[2].step(t_axis[:-1], u_pinc[:, 0], where="post", label="u1 (PINC)")
    axes[2].step(t_axis[:-1], u_pinc[:, 1], where="post", label="u2 (PINC)")
    axes[2].step(t_axis[:-1], u_ode[:, 0], where="post", linestyle="--", color="olive", label="u1 (ODE/RK)")
    axes[2].step(t_axis[:-1], u_ode[:, 1], where="post", linestyle="--", color="darkkhaki", label="u2 (ODE/RK)")
    axes[2].set_ylabel("pump voltage")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig("fourtank_nmpc_control.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         help="training device, e.g. 'cpu', 'cuda', 'cuda:0' (default: auto-detect)")
    parser.add_argument("--checkpoint", default="checkpoints/fourtank.pt",
                         help="path to save/load the training checkpoint")
    parser.add_argument("--resume", action="store_true", default=True,
                         help="resume training from --checkpoint if it exists")
    parser.add_argument("--no-checkpoint", action="store_true",
                         help="disable checkpointing entirely")
    parser.add_argument("--load-only", default=None,
                         help="skip training entirely and load a trained model from this "
                              "checkpoint path (e.g. for re-running the control experiment only)")
    args = parser.parse_args()

    physics = FourTank()
    T = 10.0

    if args.load_only:
        print(f"Loading trained model from '{args.load_only}' (skipping training)...")
        model, payload = load_pinc_model(args.load_only, map_location="cpu")
        history = payload["extra"].get("history")
    else:
        print(f"Training PINC net for the four-tank system on device='{args.device}'...")
        checkpoint_path = None if args.no_checkpoint else args.checkpoint
        print(f"{checkpoint_path=} {args.resume=}")
        model, history = train_pinc(physics, T, k1_epochs=5000, k2_iters=2000,
                                     device=args.device,
                                     checkpoint_path=checkpoint_path,
                                     resume=args.resume)

    if history is not None:
        plot_training_curves(history)
    plot_long_range_prediction(model, physics, T)
    run_control_experiment(model, physics, T)

    print("\nSaved figures: fourtank_training_curves.png, "
          "fourtank_long_range_prediction.png, fourtank_nmpc_control.png")


if __name__ == "__main__":
    main()