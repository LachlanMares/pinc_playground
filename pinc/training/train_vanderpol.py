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
"""
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


def build_validation_trajectory(physics, T, n_steps=180, seed=0):
    torch.manual_seed(seed)
    y0 = torch.tensor([-2.14, 0.25])
    u_seq = random_control_signal(n_steps, control_dim=1, u_range=(-1.0, 1.0), seed=seed)
    y_true = simulate(physics, y0, u_seq, dt=T, substeps=20)
    return y0, u_seq, y_true


def train_pinc(physics, T, k1_epochs=500, k2_iters=2000, hidden=20, depth=4,
               n_boundary=1000, n_collocation=100000, lambda_phys=1.0):

    model = PINCModel(
        backbone=MLP(in_dim=1 + physics.state_dim + physics.control_dim,
                     out_dim=physics.state_dim, hidden=hidden, depth=depth),
        state_dim=physics.state_dim,
        control_dim=physics.control_dim,
        T=T,
    )

    sampler = make_vanderpol_sampler(physics, T)
    loss_fn = PINCLoss(physics, lambda_phys=lambda_phys)

    y0_val, u_val, y_true_val = build_validation_trajectory(physics, T)

    def validate_fn(m):
        return mse_gen(m, y0_val, u_val, y_true_val)

    trainer = Trainer(model, sampler, loss_fn,
                       n_boundary=n_boundary, n_collocation=n_collocation, lr=1e-3)

    history = trainer.fit(k1_epochs=k1_epochs, k2_iters=k2_iters,
                           validate_fn=validate_fn, log_every=max(1, k1_epochs // 10))

    return model, history


def plot_training_curves(history):
    plt.figure(figsize=(10, 5))
    plt.semilogy(history["total"], label="Total")
    plt.semilogy(history["data"], label="Data (MSE_y)")
    plt.semilogy(history["physics"], label="Physics (MSE_F)")
    val = [v for v in history["val"] if v is not None]
    if val:
        plt.semilogy(range(len(history["total"]) - len(val), len(history["total"])), val, label="Validation")
    plt.xlabel("Iteration (ADAM epochs then L-BFGS iters)")
    plt.ylabel("MSE (log scale)")
    plt.title("PINC training loss - Van der Pol")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("vanderpol_training_curves.png", dpi=150)
    plt.close()


def plot_long_range_prediction(model, physics, T):
    y0, u_seq, y_true = build_validation_trajectory(physics, T, n_steps=20, seed=1)
    y_pred = pinc_rollout(model, y0, u_seq)

    t_axis = torch.arange(y_pred.shape[0]) * T

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t_axis, y_true[:, 0], "k-", label="true x1")
    ax1.plot(t_axis, y_true[:, 1], "k--", label="true x2")
    ax1.plot(t_axis, y_pred[:, 0].detach(), "o-", color="tab:blue", label="PINC x1")
    ax1.plot(t_axis, y_pred[:, 1].detach(), "o-", color="tab:pink", label="PINC x2")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("outputs y1, y2")
    ax1.legend(loc="upper left")
    ax1.set_title("PINC net prediction for the Van der Pol oscillator (self-loop)")

    ax2 = ax1.twinx()
    ax2.step(t_axis[:-1], u_seq[:, 0], where="post", color="grey", linestyle="--", alpha=0.6)
    ax2.set_ylabel("input u", color="grey")

    plt.tight_layout()
    plt.savefig("vanderpol_long_range_prediction.png", dpi=150)
    plt.close()

    mse = torch.mean((y_pred - y_true) ** 2).item()
    print(f"Long-range self-loop generalization MSE: {mse:.3e}")


def integral_metrics(y_ref, y):
    err = (y_ref - y).abs()
    iae = err.sum().item()
    rmse = torch.sqrt(torch.mean((y_ref - y) ** 2)).item()
    return rmse, iae


def run_control_experiment(model, physics, T, n_steps=120):
    """Fig. 10 / Table 1 style experiment: regulate x1, x2 to zero via NMPC,
    using PINC as the predictive model, and compare against the ODE/RK4
    baseline predictive model."""

    y_ref = torch.zeros(n_steps, 2)  # reference: keep x1 = x2 = 0

    y0 = torch.tensor([1.0, 0.0])

    Q = torch.diag(torch.tensor([10.0, 10.0]))
    R = torch.diag(torch.tensor([1.0]))

    N1, N2, Nu = 1, 5, 5
    u_min, u_max = [-1.0], [1.0]

    # --- PINC-based NMPC ---
    t0 = time.time()
    y_pinc, u_pinc = run_nmpc_simulation(
        control_interface=model.step,
        plant_step=rk4_control_interface(physics, T, substeps=20),
        y0=y0, y_ref_full=y_ref, control_dim=1,
        N1=N1, N2=N2, Nu=Nu, Q=Q, R=R, u_min=u_min, u_max=u_max,
    )
    t_pinc = time.time() - t0

    # --- ODE/RK4-based NMPC (baseline, uses the true model as predictor) ---
    rk_interface = rk4_control_interface(physics, T, substeps=20)
    t0 = time.time()
    y_ode, u_ode = run_nmpc_simulation(
        control_interface=rk_interface,
        plant_step=rk_interface,
        y0=y0, y_ref_full=y_ref, control_dim=1,
        N1=N1, N2=N2, Nu=Nu, Q=Q, R=R, u_min=u_min, u_max=u_max,
    )
    t_ode = time.time() - t0

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
    plt.savefig("vanderpol_nmpc_control.png", dpi=150)
    plt.close()


def main():
    physics = VanDerPol(mu=1.0)
    T = 0.5

    print("Training PINC net for the Van der Pol oscillator...")
    model, history = train_pinc(physics, T, k1_epochs=5000, k2_iters=2000)

    plot_training_curves(history)
    plot_long_range_prediction(model, physics, T)
    run_control_experiment(model, physics, T)

    print("\nSaved figures: vanderpol_training_curves.png, "
          "vanderpol_long_range_prediction.png, vanderpol_nmpc_control.png")


if __name__ == "__main__":
    main()
