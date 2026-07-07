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
               n_boundary=1000, n_collocation=100000, lambda_phys=1.0):

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
                       n_boundary=n_boundary, n_collocation=n_collocation, lr=1e-3)

    history = trainer.fit(k1_epochs=k1_epochs, k2_iters=k2_iters,
                           validate_fn=validate_fn, log_every=max(1, k1_epochs // 10))

    return model, history


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


def run_control_experiment(model, physics, T, n_steps=60):
    """Fig. 13/14, Table 1 style experiment: regulate h1, h2 to a step
    reference while keeping h3, h4 within [0.6, 5.5] cm."""

    y_ref = torch.zeros(n_steps, 4)
    y_ref[:, 0] = 10.0
    y_ref[:, 1] = 14.0
    y_ref[n_steps // 2:, 0] = 8.0
    y_ref[n_steps // 2:, 1] = 12.5

    y0 = torch.tensor([2.0, 2.0, 2.0, 2.0])

    Q = torch.diag(torch.tensor([10.0, 10.0, 0.0, 0.0]))
    R = torch.diag(torch.tensor([1.0, 1.0]))

    N1, N2, Nu = 1, 5, 5
    u_min, u_max = [0.0, 0.0], [5.0, 5.0]
    state_constraints = [(2, 0.6, 5.5), (3, 0.6, 5.5)]

    t0 = time.time()
    y_pinc, u_pinc = run_nmpc_simulation(
        control_interface=model.step,
        plant_step=rk4_control_interface(physics, T, substeps=20),
        y0=y0, y_ref_full=y_ref, control_dim=2,
        N1=N1, N2=N2, Nu=Nu, Q=Q, R=R, u_min=u_min, u_max=u_max,
        state_constraints=state_constraints,
    )
    t_pinc = time.time() - t0

    rk_interface = rk4_control_interface(physics, T, substeps=20)
    t0 = time.time()
    y_ode, u_ode = run_nmpc_simulation(
        control_interface=rk_interface,
        plant_step=rk_interface,
        y0=y0, y_ref_full=y_ref, control_dim=2,
        N1=N1, N2=N2, Nu=Nu, Q=Q, R=R, u_min=u_min, u_max=u_max,
        state_constraints=state_constraints,
    )
    t_ode = time.time() - t0

    rmse_pinc, iae_pinc = integral_metrics(y_ref[:, :2], y_pinc[1:, :2])
    rmse_ode, iae_ode = integral_metrics(y_ref[:, :2], y_ode[1:, :2])

    print("\nControl performance (four tanks, Table 1 style):")
    print(f"{'Model':<10}{'RMSE':>10}{'IAE':>10}{'time(s)':>12}")
    print(f"{'PINC':<10}{rmse_pinc:>10.3f}{iae_pinc:>10.2f}{t_pinc:>12.3f}")
    print(f"{'ODE/RK':<10}{rmse_ode:>10.3f}{iae_ode:>10.2f}{t_ode:>12.3f}")

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
    axes[1].axhline(0.6, color="grey", linestyle=":")
    axes[1].axhline(5.5, color="grey", linestyle=":")
    axes[1].set_ylabel("constrained levels")
    axes[1].legend(fontsize=7)

    axes[2].step(t_axis[:-1], u_pinc[:, 0], where="post", label="u1 (PINC)")
    axes[2].step(t_axis[:-1], u_pinc[:, 1], where="post", label="u2 (PINC)")
    axes[2].set_ylabel("pump voltage")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("fourtank_nmpc_control.png", dpi=150)
    plt.close()


def main():
    physics = FourTank()
    T = 10.0

    print("Training PINC net for the four-tank system...")
    model, history = train_pinc(physics, T, k1_epochs=500, k2_iters=2000)

    plot_training_curves(history)
    plot_long_range_prediction(model, physics, T)
    run_control_experiment(model, physics, T)

    print("\nSaved figures: fourtank_training_curves.png, "
          "fourtank_long_range_prediction.png, fourtank_nmpc_control.png")


if __name__ == "__main__":
    main()
