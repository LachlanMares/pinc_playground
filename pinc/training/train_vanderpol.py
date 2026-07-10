"""
Reproduces the Van der Pol oscillator experiments of Section 4.1 of
Antonelo et al., "Physics-Informed Neural Nets for Control of Dynamical
Systems" (arXiv:2104.02556):

  1) Train a single PINC net with T = 0.5s (Sec. 4.1.2 / 4.1.4).
  2) Long-range self-loop prediction on a randomly generated control
     signal (Fig. 9).
  3) Closed-loop NMPC control using the PINC net as the predictive
     model, compared against the same NMPC using the true ODE (RK4)
     as predictive model (Fig. 10, Table 1).

Run with:  python -m pinc.training.train_vanderpol
Resume a previously interrupted run with:
           python -m pinc.training.train_vanderpol --resume
Train on GPU (if available):
           python -m pinc.training.train_vanderpol --device cuda
"""
import argparse
import time

import torch
import matplotlib.pyplot as plt

from pinc.physics.vanderpol import VanDerPol
from pinc.nn.mlp import MLP
from pinc.models.pinc import PINCModel
from pinc.core.trainer import Trainer
from pinc.losses.pinc_loss import PINCLoss
from pinc.datasets.vanderpol import make_vanderpol_sampler, random_control_signal
from pinc.evaluation.rollout import pinc_rollout, mse_gen
from pinc.simulation.rk4 import simulate, rk4_control_interface
from pinc.control.nmpc import run_nmpc_simulation
from pinc.utils.checkpoint import load_pinc_model


def build_validation_trajectory(physics: VanDerPol, T: float, n_steps: int = 180, seed: int = 0):
    torch.manual_seed(seed)
    y0 = torch.tensor([-2.14, 0.25])
    u_seq = random_control_signal(n_steps, control_dim=1, u_range=(-1.0, 1.0), seed=seed)
    y_true = simulate(physics, y0, u_seq, dt=T, substeps=20)

    return y0, u_seq, y_true


def build_easy_validation_trajectory(physics: VanDerPol, T, n_steps=20, u_const=0.0):
    """
    A much simpler long-range test than `build_validation_trajectory`:
    a single constant input (default u=0, i.e. the free/unforced Van der
    Pol oscillator) instead of a rapidly-switching random signal.

    Use this to separate two different failure modes that otherwise get
    tangled together in the random-signal plot:
      1) self-loop error compounding step-to-step even when the system
         is doing something simple/smooth, vs.
      2) the network reacting poorly to input *changes* it may be
         under-sampled on (large or frequent jumps in u).
    If PINC tracks this trajectory well but still diverges on the
    random-signal one, the problem is more likely (2); if it already
    diverges here, it's (1) -- e.g. insufficient multistep-loss weight,
    or the collocation sampler not covering the states actually visited
    once errors start compounding.
    """
    y0 = torch.tensor([-2.14, 0.25])
    u_seq = torch.full((n_steps, physics.control_dim), u_const)
    y_true = simulate(physics, y0, u_seq, dt=T, substeps=20)

    return y0, u_seq, y_true


def train_pinc(physics: VanDerPol, T, k1_epochs=500, k2_iters=2000, hidden=20, depth=4,
               n_boundary=4000, n_collocation=100000,
               n_multistep=4000, multistep_k=3, lambda_phys=1.0,
               device="cpu", checkpoint_path=None, resume=False, save_every=100):
    """
    device          : "cpu", "cuda", or "cuda:N". Training (the ADAM/
                      L-BFGS loop over boundary + collocation batches)
                      is the part that actually benefits from a GPU;
                      NMPC/rollout afterwards are left on CPU (see
                      `run_control_experiment`) since they're dominated
                      by small per-step SciPy/autograd overhead rather
                      than raw matmul throughput.
    n_boundary, n_collocation, n_multistep, multistep_k :
                      batch sizes for the three loss terms in PINCLoss.
                      All independent per-sample work (aside from the
                      short length-multistep_k sequential chain within
                      each multistep sample), so raising these is the
                      most direct way to give the GPU more parallel
                      work per iteration if utilization looks low --
                      try this before enlarging the network, and check
                      `nvidia-smi`/`nvtop` to see where it saturates.
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

    sampler = make_vanderpol_sampler(physics, T, device=device)
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
    plt.title("PINC training loss - Van der Pol")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/ode/vanderpol_training_curves.png", dpi=150)
    plt.close()


def plot_long_range_prediction(model, physics, T):
    _plot_rollout_diagnostic(
        model, physics, T,
        *build_validation_trajectory(physics, T, n_steps=20, seed=1),
        title="PINC self-loop vs true (random-switching input)",
        out_path="plots/ode/vanderpol_long_range_prediction.png",
    )
    _plot_rollout_diagnostic(
        model, physics, T,
        *build_easy_validation_trajectory(physics, T, n_steps=20, u_const=0.0),
        title="PINC self-loop vs true (constant u=0, unforced)",
        out_path="plots/ode/vanderpol_long_range_prediction_easy.png",
    )


def _plot_rollout_diagnostic(model, physics, T, y0, u_seq, y_true, title, out_path):
    """
    Four-panel diagnostic, replacing the old single-overlay plot (where
    the true and predicted curves drew on top of each other and it was
    hard to tell whether they matched or one was just hidden under the
    other):

      1) True trajectory alone
      2) PINC prediction alone (same axis limits as panel 1, so shape/
         scale differences are visible at a glance)
      3) Both overlaid, true as thin black lines, PINC as markers on top
      4) Per-step absolute error, x1 and x2 separately

    Panel 4 is the one to actually look at for a pass/fail read: it
    should be small and roughly flat/decaying, not visibly growing
    step-by-step. Growth there is compounding self-loop error even if
    panels 1-3 "look" close at this scale.
    """
    y_pred = pinc_rollout(model, y0, u_seq).detach()
    t_axis = torch.arange(y_pred.shape[0]) * T
    ylim = (min(y_true.min(), y_pred.min()) - 0.3, max(y_true.max(), y_pred.max()) + 0.3)

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    ax = axes[0]
    ax.plot(t_axis, y_true[:, 0], "k-", label="true x1")
    ax.plot(t_axis, y_true[:, 1], "k--", label="true x2")
    ax.set_ylabel("state")
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("1. True (RK4) trajectory")

    ax = axes[1]
    ax.plot(t_axis, y_pred[:, 0], "o-", color="tab:blue", label="PINC x1")
    ax.plot(t_axis, y_pred[:, 1], "o-", color="tab:pink", label="PINC x2")
    ax.set_ylabel("state")
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("2. PINC self-loop prediction")

    ax = axes[2]
    ax.plot(t_axis, y_true[:, 0], "-", color="black", linewidth=1, alpha=0.6)
    ax.plot(t_axis, y_true[:, 1], "--", color="black", linewidth=1, alpha=0.6)
    ax.plot(t_axis, y_pred[:, 0], "o", color="tab:blue", markersize=4, label="PINC x1")
    ax.plot(t_axis, y_pred[:, 1], "o", color="tab:pink", markersize=4, label="PINC x2")
    ax.set_ylabel("state")
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("3. Combined (thin black = true, dots = PINC)")

    ax = axes[3]
    err = (y_pred - y_true).abs()
    ax.plot(t_axis, err[:, 0], "o-", color="tab:blue", label="|error| x1")
    ax.plot(t_axis, err[:, 1], "o-", color="tab:pink", label="|error| x2")
    ax.set_ylabel("abs error")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("4. Per-step error -- should stay flat/small, not grow")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    mse = torch.mean((y_pred - y_true) ** 2).item()
    print(f"Long-range self-loop generalization MSE: {mse:.3e}")


def integral_metrics(y_ref, y):
    err = (y_ref - y).abs()
    iae = err.sum().item()
    rmse = torch.sqrt(torch.mean((y_ref - y) ** 2)).item()
    return rmse, iae


def _solve_nmpc_worker(kind, model, physics, T, y0, y_ref, control_dim,
                        N1, N2, Nu, Q, R, u_min, u_max, maxiter):
    """
    Runs a single closed-loop NMPC simulation in its own process.

    `kind` selects which predictive model drives the controller:
      - "pinc" : the trained PINC net (`model`, already CPU-resident)
      - "ode"  : a fresh RK4 predictive interface, also used as the
                 plant here (matches the original Van der Pol
                 experiment, which -- unlike the four-tank one --
                 uses the same substep count for both)

    The RK4 interface is rebuilt here rather than passed in, since the
    nested closure `rk4_control_interface` returns isn't reliably
    picklable across a process boundary.

    torch.set_num_threads(1) keeps each worker from spinning up its own
    intra-op thread pool -- with tensors this small (batch size 1) that
    overhead is pure contention on top of the process-level
    parallelism, not useful compute.
    """
    torch.set_num_threads(1)

    if kind == "pinc":
        control_interface = model.step
        plant = rk4_control_interface(physics, T, substeps=20)
    else:
        rk_interface = rk4_control_interface(physics, T, substeps=20)
        control_interface = rk_interface
        plant = rk_interface

    t0 = time.time()
    y, u = run_nmpc_simulation(
        control_interface=control_interface,
        plant_step=plant,
        y0=y0, y_ref_full=y_ref, control_dim=control_dim,
        N1=N1, N2=N2, Nu=Nu, Q=Q, R=R, u_min=u_min, u_max=u_max,
        maxiter=maxiter,
        desc="PINC" if kind == "pinc" else "ODE/RK", position=0 if kind == "pinc" else 1,
    )
    elapsed = time.time() - t0
    return y, u, elapsed


def run_control_experiment(model, physics, T, n_steps=120):
    """Fig. 10 / Table 1 style experiment: regulate x1, x2 to zero via NMPC,
    using PINC as the predictive model, and compare against the ODE/RK4
    baseline predictive model.

    NMPC solves go through scipy.optimize (CPU-only, numpy-backed), so
    the model is moved to CPU here regardless of what device it was
    trained on -- there's no benefit to keeping it on GPU for the tiny,
    Python-overhead-dominated per-step solves in `nmpc.py`.
    """
    model = model.to("cpu")

    y_ref = torch.zeros(n_steps, 2)  # reference: keep x1 = x2 = 0

    y0 = torch.tensor([1.0, 0.0])

    Q = torch.diag(torch.tensor([10.0, 10.0]))
    R = torch.diag(torch.tensor([1.0]))

    N1, N2, Nu = 1, 5, 5
    u_min, u_max = [-1.0], [1.0]

    # The PINC-driven and ODE/RK-driven NMPC runs are fully independent
    # (same reference, no shared mutable state), so run them in separate
    # processes instead of back-to-back -- each one otherwise pins a
    # single core for the whole simulation.
    common = dict(physics=physics, T=T, y0=y0, y_ref=y_ref, control_dim=1,
                  N1=N1, N2=N2, Nu=Nu, Q=Q, R=R, u_min=u_min, u_max=u_max,
                  maxiter=30)

    print("Running PINC NMPC...")

    y_pinc, u_pinc, t_pinc = _solve_nmpc_worker(
        "pinc",
        model,
        **common
    )

    print("Running RK4 NMPC...")

    y_ode, u_ode, t_ode = _solve_nmpc_worker(
        "ode",
        model,
        **common
    )

    rmse_pinc, iae_pinc = integral_metrics(y_ref, y_pinc[1:])
    rmse_ode, iae_ode = integral_metrics(y_ref, y_ode[1:])

    print("\nControl performance (Van der Pol, Table 1 style):")
    print(f"{'Model':<10}{'RMSE':>10}{'IAE':>10}{'time(s)':>12}")
    print(f"{'PINC':<10}{rmse_pinc:>10.3f}{iae_pinc:>10.2f}{t_pinc:>12.3f}")
    print(f"{'ODE/RK':<10}{rmse_ode:>10.3f}{iae_ode:>10.2f}{t_ode:>12.3f}")

    t_axis = torch.arange(n_steps + 1) * T

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t_axis, y_pinc[:, 0], label="x1 (PINC)")
    axes[0].plot(t_axis, y_pinc[:, 1], label="x2 (PINC)")
    axes[0].plot(t_axis, y_ode[:, 0], "--", color="olive", label="x1 (ODE/RK)")
    axes[0].plot(t_axis, y_ode[:, 1], "--", color="darkkhaki", label="x2 (ODE/RK)")
    axes[0].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[0].set_ylabel("state")
    axes[0].legend(fontsize=8)
    axes[0].set_title("NMPC control of the Van der Pol oscillator")

    axes[1].step(t_axis[:-1], u_pinc[:, 0], where="post", label="u (PINC)")
    axes[1].step(t_axis[:-1], u_ode[:, 0], where="post", linestyle="--", color="olive", label="u (ODE/RK)")
    axes[1].set_ylabel("control u")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("plots/ode/vanderpol_nmpc_control.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         help="training device, e.g. 'cpu', 'cuda', 'cuda:0' (default: auto-detect)")
    parser.add_argument("--checkpoint", default="checkpoints/vanderpol.pt",
                         help="path to save/load the training checkpoint")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                         help="resume training from --checkpoint if it exists (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                         help="ignore any existing checkpoint and train from scratch")
    parser.add_argument("--no-checkpoint", action="store_true",
                         help="disable checkpointing entirely")
    parser.add_argument("--load-only", default="checkpoints/vanderpol.pt",
                         help="skip training entirely and load a trained model from this "
                              "checkpoint path (e.g. for re-running the control experiment only)")
    args = parser.parse_args()

    physics = VanDerPol(mu=1.0)
    T = 0.5

    if args.load_only is not None:
        print(f"Loading trained model from '{args.load_only}' (skipping training)...")
        model, payload = load_pinc_model(args.load_only, map_location="cpu")
        history = payload["extra"].get("history")
    else:
        print(f"Training PINC net for the Van der Pol oscillator on device='{args.device}'...")
        checkpoint_path = None if args.no_checkpoint else args.checkpoint
        model, history = train_pinc(physics, T, k1_epochs=10000, k2_iters=2000,
                                     device=args.device,
                                     checkpoint_path=checkpoint_path,
                                     resume=args.resume)

    if history is not None:
        plot_training_curves(history)
    plot_long_range_prediction(model, physics, T)
    run_control_experiment(model, physics, T)

    print("\nSaved figures: vanderpol_training_curves.png, "
          "vanderpol_long_range_prediction.png, vanderpol_nmpc_control.png")


if __name__ == "__main__":
    main()