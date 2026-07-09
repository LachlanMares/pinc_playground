import argparse
import time

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
from pinc.control.nmpc_casadi import (
    CasadiSingleShootingNMPC,
    CasadiMultipleShootingNMPC,
    CasadiRTIController,
)
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


def build_easy_validation_trajectory(physics, T, n_steps=35, u_const=2.5):
    """
    Simpler long-range test than `build_validation_trajectory`: a single
    constant pump voltage on both channels instead of a fresh random
    setpoint every step. Use this to separate two failure modes that
    are otherwise tangled together in the random-signal plot:
      1) self-loop error compounding step-to-step even under a simple,
         unchanging input, vs.
      2) the network reacting poorly to input *changes* it may be
         under-sampled on.
    If PINC tracks this trajectory well but still diverges on the
    random-signal one, look at input-transition coverage in the
    collocation sampler; if it already diverges here, look at
    `lambda_multistep` / multistep_k / network capacity instead.
    """
    y0 = torch.tensor([8.0, 8.0, 8.0, 8.0])
    u_seq = torch.full((n_steps, physics.control_dim), u_const)
    y_true = simulate(physics, y0, u_seq, dt=T, substeps=20)
    return y0, u_seq, y_true


def train_pinc(physics, T, k1_epochs=500, k2_iters=2000, hidden=128, depth=3,
               n_boundary=4000, n_collocation=100000,
               n_multistep=4000, multistep_k=3, lambda_phys=1.0,
               device="cpu", checkpoint_path=None, resume=False, save_every=100):
    """
    device          : "cpu", "cuda", or "cuda:N" -- see the note in
                       train_vanderpol.py's train_pinc; the four-tank
                       net (5 x 20) is still small, but with
                       n_collocation=100000 points per iteration a GPU
                       can meaningfully speed up training.
    n_boundary, n_collocation, n_multistep, multistep_k :
                       batch sizes for the three loss terms in
                       PINCLoss (boundary/physics, endpoint reuses the
                       boundary batch, multistep gets its own batch of
                       n_multistep chains of length multistep_k). All
                       three are independent per-sample computations
                       (only multistep_k is a short *sequential* chain
                       within each sample), so raising n_boundary /
                       n_collocation / n_multistep is the most direct
                       way to give the GPU more parallel work per
                       iteration if utilization looks low -- try
                       increasing these before reaching for a bigger
                       network, and watch `nvidia-smi` (or `nvtop`) to
                       see where utilization actually saturates for
                       your GPU.
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

    sampler = make_fourtank_sampler(physics, T, device=device)
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
    plt.xlabel("Iteration")
    plt.ylabel("MSE (log scale)")
    plt.title("PINC training loss - Four tanks")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("fourtank_training_curves.png", dpi=150)
    plt.close()


def plot_long_range_prediction(model, physics, T):
    """
    Writes two 4-row x 2-col diagnostic figures (rows: true / PINC
    prediction / combined / abs error; columns: h1&h2 / h3&h4), same
    idea as the Van der Pol version:

      - fourtank_long_range_prediction.png      : random control signal
      - fourtank_long_range_prediction_easy.png : constant control input

    Row 4 (abs error) is the one to actually check: flat/small on both
    -> self-loop is fine. Growing on the constant-input version too ->
    genuine compounding self-loop error (raise lambda_multistep /
    multistep_k). Growing only on the random-signal version -> input-
    transition coverage in the collocation sampler, not the self-loop
    mechanism itself.
    """
    _plot_rollout_diagnostic(
        model, physics, T,
        *build_validation_trajectory(physics, T, n_steps=35, seed=1),
        title="PINC self-loop vs true (random control signal)",
        out_path="fourtank_long_range_prediction.png",
    )
    _plot_rollout_diagnostic(
        model, physics, T,
        *build_easy_validation_trajectory(physics, T, n_steps=35, u_const=2.5),
        title="PINC self-loop vs true (constant u1=u2=2.5)",
        out_path="fourtank_long_range_prediction_easy.png",
    )


def _plot_rollout_diagnostic(model, physics, T, y0, u_seq, y_true, title, out_path):
    y_pred = pinc_rollout(model, y0, u_seq).detach()
    t_axis = torch.arange(y_pred.shape[0]) * T

    col_specs = [
        (0, 1, "h1", "h2", "h1, h2"),
        (2, 3, "h3", "h4", "h3, h4"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)

    for c, (i, j, name_i, name_j, ylabel) in enumerate(col_specs):
        ylim = (min(y_true[:, i].min(), y_true[:, j].min(), y_pred[:, i].min(), y_pred[:, j].min()) - 0.5,
                max(y_true[:, i].max(), y_true[:, j].max(), y_pred[:, i].max(), y_pred[:, j].max()) + 0.5)

        ax = axes[0, c]
        ax.plot(t_axis, y_true[:, i], "k-", label=f"true {name_i}")
        ax.plot(t_axis, y_true[:, j], "k--", label=f"true {name_j}")
        ax.set_ylabel(ylabel); ax.set_ylim(*ylim); ax.legend(fontsize=8)
        ax.set_title(f"1. True (RK4) -- {ylabel}")

        ax = axes[1, c]
        ax.plot(t_axis, y_pred[:, i], "o-", color="tab:blue", label=f"PINC {name_i}")
        ax.plot(t_axis, y_pred[:, j], "o-", color="tab:pink", label=f"PINC {name_j}")
        ax.set_ylabel(ylabel); ax.set_ylim(*ylim); ax.legend(fontsize=8)
        ax.set_title(f"2. PINC prediction -- {ylabel}")

        ax = axes[2, c]
        ax.plot(t_axis, y_true[:, i], "-", color="black", linewidth=1, alpha=0.6)
        ax.plot(t_axis, y_true[:, j], "--", color="black", linewidth=1, alpha=0.6)
        ax.plot(t_axis, y_pred[:, i], "o", color="tab:blue", markersize=4, label=f"PINC {name_i}")
        ax.plot(t_axis, y_pred[:, j], "o", color="tab:pink", markersize=4, label=f"PINC {name_j}")
        ax.set_ylabel(ylabel); ax.set_ylim(*ylim); ax.legend(fontsize=8)
        ax.set_title(f"3. Combined -- {ylabel}")

        ax = axes[3, c]
        err_i = (y_pred[:, i] - y_true[:, i]).abs()
        err_j = (y_pred[:, j] - y_true[:, j]).abs()
        ax.plot(t_axis, err_i, "o-", color="tab:blue", label=f"|error| {name_i}")
        ax.plot(t_axis, err_j, "o-", color="tab:pink", label=f"|error| {name_j}")
        ax.set_ylabel("abs error"); ax.set_xlabel("Time (s)"); ax.legend(fontsize=8)
        ax.set_title(f"4. Per-step error -- {ylabel} (should stay flat/small)")

    fig.suptitle(title)
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


def _run_nmpc(kind, model, physics, T, y0, y_ref, control_dim,
              N1, N2, Nu, Q, R, u_min, u_max, state_constraints, maxiter):
    """
    Runs a single closed-loop NMPC simulation, using `kind` to pick both
    the predictive model and the controller architecture:

      - "pinc"              : trained PINC net,  scipy/SLSQP single-shoot
                               (the original NMPCController, via
                               run_nmpc_simulation's own default)
      - "ode"                : ODE/RK4 baseline,  scipy/SLSQP single-shoot
      - "pinc_casadi_single" : trained PINC net,  CasADi/IPOPT single-shoot
      - "pinc_casadi_multi"  : trained PINC net,  CasADi/IPOPT multiple-shoot
      - "pinc_casadi_rti"    : trained PINC net,  CasADi RTI (single QP/step)

    The three "pinc_casadi_*" variants all reuse the exact same trained
    `model.step` as the predictive model; only the *solver/architecture*
    changes, which is the point of nmpc_casadi.py -- each one is a
    drop-in for `NMPCController` via `run_nmpc_simulation`'s
    `controller=` argument.
    """
    plant = rk4_control_interface(physics, T, substeps=20)

    if kind == "pinc":
        control_interface, controller, desc = model.step, None, "PINC (SLSQP)"
    elif kind == "ode":
        control_interface = rk4_control_interface(physics, T, substeps=5)
        controller, desc = None, "ODE/RK (SLSQP)"
    elif kind == "pinc_casadi_single":
        control_interface = model.step
        controller = CasadiSingleShootingNMPC(
            model.step, control_dim, N1, N2, Nu, Q, R,
            u_min=u_min, u_max=u_max, state_constraints=state_constraints)
        desc = "PINC (IPOPT single-shoot)"
    elif kind == "pinc_casadi_multi":
        control_interface = model.step
        controller = CasadiMultipleShootingNMPC(
            model.step, control_dim, N1, N2, Nu, Q, R,
            u_min=u_min, u_max=u_max, state_constraints=state_constraints)
        desc = "PINC (IPOPT multi-shoot)"
    elif kind == "pinc_casadi_rti":
        control_interface = model.step
        controller = CasadiRTIController(
            model.step, control_dim, N1, N2, Nu, Q, R,
            u_min=u_min, u_max=u_max, state_constraints=state_constraints)
        desc = "PINC (RTI-QP)"
    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    t0 = time.time()
    y, u = run_nmpc_simulation(
        control_interface=control_interface,
        plant_step=plant,
        y0=y0, y_ref_full=y_ref, control_dim=control_dim,
        N1=N1, N2=N2, Nu=Nu, Q=Q, R=R, u_min=u_min, u_max=u_max,
        state_constraints=state_constraints,
        maxiter=maxiter, warm_start=True,
        controller=controller, desc=desc,
    )
    elapsed = time.time() - t0
    return y, u, elapsed


def run_control_experiment(model, physics, T, n_steps=45,
                            architectures=("pinc", "ode",
                                           "pinc_casadi_single",
                                           "pinc_casadi_multi")):
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

    architectures : which controller(s) to run and compare, from
        {"pinc", "ode", "pinc_casadi_single", "pinc_casadi_multi",
        "pinc_casadi_rti"} -- see `_run_nmpc`. All share the same
        plant, reference, and constraints, so results are directly
        comparable; drop entries from the default tuple to skip them
        (e.g. `architectures=("pinc", "pinc_casadi_multi")` for a quick
        two-way comparison).

        "pinc_casadi_rti" is deliberately left out of the default: the
        four-tank orifice equation's sqrt(h) term has a steep, rapidly
        varying local gradient, and RTI's single-linearization-per-step
        approach re-uses that gradient across the whole prediction
        horizon without ever re-deriving it from the true nonlinear
        model mid-horizon. In testing this reliably drove the QP
        infeasible partway through most timesteps (silently falling
        back to a zero-increment control action -- see the fallback
        in `CasadiRTIController.solve`), giving noticeably worse
        constraint satisfaction than the other three. It works fine on
        the gentler Van der Pol oscillator (`compare_nmpc_architectures.py`);
        pass it explicitly here if you want to see the four-tank
        failure mode for yourself.

    The scipy-driven variants ("pinc", "ode") are CPU-only anyway, and
    CasADi likewise runs on CPU here, so the model is moved to CPU
    regardless of what device it was trained on.
    """
    model = model.to("cpu")

    y_ref = torch.zeros(n_steps, 4)
    y_ref[:, 0] = 10.0
    y_ref[:, 1] = 14.0
    y_ref[n_steps // 2:, 0] = 8.0
    y_ref[n_steps // 2:, 1] = 12.5

    y0 = torch.tensor([8.0, 8.0, 8.0, 8.0])

    Q = torch.diag(torch.tensor([10.0, 10.0, 0.0, 0.0]))
    R = torch.diag(torch.tensor([1.0, 1.0]))

    N1, N2, Nu = 1, 15, 5
    u_min, u_max = [0.0, 0.0], [5.0, 5.0]
    state_constraints = [(2, 0.6, 5.5), (3, 0.6, 5.5)]

    common = dict(physics=physics, T=T, y0=y0, y_ref=y_ref, control_dim=2,
                  N1=N1, N2=N2, Nu=Nu, Q=Q, R=R, u_min=u_min, u_max=u_max,
                  state_constraints=state_constraints, maxiter=80)

    results = {}
    for kind in architectures:
        print(f"Running {kind} NMPC...")
        y, u, elapsed = _run_nmpc(kind=kind, model=model, **common)
        rmse, iae = integral_metrics(y_ref[:, :2], y[1:, :2])
        results[kind] = dict(y=y, u=u, elapsed=elapsed, rmse=rmse, iae=iae)

    def violation(y):
        h3, h4 = y[1:, 2], y[1:, 3]
        lo_viol = torch.clamp(0.6 - h3, min=0).sum() + torch.clamp(0.6 - h4, min=0).sum()
        hi_viol = torch.clamp(h3 - 5.5, min=0).sum() + torch.clamp(h4 - 5.5, min=0).sum()
        return (lo_viol + hi_viol).item()

    print("\nControl performance (four tanks, Table 1 style):")
    print(f"{'Architecture':<26}{'RMSE':>10}{'IAE':>10}{'time(s)':>12}{'h3/h4 viol':>14}")
    for kind, r in results.items():
        print(f"{kind:<26}{r['rmse']:>10.3f}{r['iae']:>10.2f}{r['elapsed']:>12.3f}"
              f"{violation(r['y']):>14.3f}")

    t_axis = torch.arange(n_steps + 1) * T
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for i, (kind, r) in enumerate(results.items()):
        y, u, color = r["y"], r["u"], colors[i % len(colors)]
        axes[0].plot(t_axis, y[:, 0], color=color, label=f"h1 ({kind})")
        axes[0].plot(t_axis, y[:, 1], "--", color=color, label=f"h2 ({kind})")
        axes[1].plot(t_axis, y[:, 2], color=color, label=f"h3 ({kind})")
        axes[1].plot(t_axis, y[:, 3], "--", color=color, label=f"h4 ({kind})")
        axes[2].step(t_axis[:-1], u[:, 0], where="post", color=color, label=f"u1 ({kind})")
        axes[2].step(t_axis[:-1], u[:, 1], where="post", linestyle="--", color=color, label=f"u2 ({kind})")

    axes[0].plot(t_axis, torch.cat([y_ref[:, 0], y_ref[-1:, 0]]), "k:", label="ref h1")
    axes[0].plot(t_axis, torch.cat([y_ref[:, 1], y_ref[-1:, 1]]), "k-.", label="ref h2")
    axes[0].set_ylabel("controlled levels")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].set_title("NMPC control of the four-tank system")

    axes[1].axhline(0.6, color="grey", linestyle=":")
    axes[1].axhline(5.5, color="grey", linestyle=":")
    axes[1].set_ylabel("constrained levels")
    axes[1].legend(fontsize=6, ncol=2)

    axes[2].set_ylabel("pump voltage")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig("fourtank_nmpc_control.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         help="training device, e.g. 'cpu', 'cuda', 'cuda:0' (default: auto-detect)")
    parser.add_argument("--checkpoint", default="checkpoints/fourtank.pt",
                         help="path to save/load the training checkpoint")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                         help="resume training from --checkpoint if it exists (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                         help="ignore any existing checkpoint and train from scratch")
    parser.add_argument("--no-checkpoint", action="store_true", default=False,
                         help="disable checkpointing entirely")
    parser.add_argument("--load-only", default="checkpoints/fourtank.pt",
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

        model, history = train_pinc(physics, T, k1_epochs=25000, k2_iters=3000,
                                     device=args.device,
                                     checkpoint_path=checkpoint_path,
                                     resume=args.resume)

    if history is not None:
        plot_training_curves(history)
    plot_long_range_prediction(model, physics, T)
    run_control_experiment(model, physics, T)

    print("\nSaved figures: fourtank_training_curves.png, "
          "fourtank_long_range_prediction.png, fourtank_long_range_prediction_easy.png, "
          "fourtank_nmpc_control.png")


if __name__ == "__main__":
    main()