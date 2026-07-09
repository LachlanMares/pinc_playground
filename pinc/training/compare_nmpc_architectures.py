"""
Compares the four NMPC controller architectures available in this
codebase, all driving the *same* trained PINC net (Van der Pol
oscillator, Sec. 4.1 of the paper) on the same reference-tracking task
as `train_vanderpol.py`'s `run_control_experiment`:

  1. NMPCController          (nmpc.py)         -- scipy SLSQP, single shooting
  2. CasadiSingleShootingNMPC (nmpc_casadi.py) -- IPOPT,       single shooting
  3. CasadiMultipleShootingNMPC (nmpc_casadi.py) -- IPOPT,     multiple shooting
  4. CasadiRTIController     (nmpc_casadi.py)  -- qrqp QP,     linear time-varying RTI

Each is a drop-in for the others: same constructor arguments (control
interface, N1/N2/Nu, Q/R, bounds), same `.solve(...)` return signature,
and all can be handed straight to `run_nmpc_simulation` via its
`controller=` argument.

Run with:  python -m pinc.training.compare_nmpc_architectures
(requires a trained checkpoint at checkpoints/vanderpol.pt -- see
train_vanderpol.py)
"""
import os
import time

import torch
import matplotlib.pyplot as plt

from pinc.physics.vanderpol import VanDerPol
from pinc.simulation.rk4 import rk4_control_interface
from pinc.control.nmpc import run_nmpc_simulation
from pinc.control.nmpc_casadi import (
    CasadiSingleShootingNMPC,
    CasadiMultipleShootingNMPC,
    CasadiRTIController,
)
from pinc.utils.checkpoint import load_pinc_model

_DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "vanderpol.pt")


def integral_metrics(y_ref, y):
    err = (y_ref - y).abs()
    iae = err.sum().item()
    rmse = torch.sqrt(torch.mean((y_ref - y) ** 2)).item()
    return rmse, iae


def main(checkpoint_path=_DEFAULT_CHECKPOINT, n_steps=120):
    model, _ = load_pinc_model(checkpoint_path, map_location="cpu")
    model.eval()

    physics = VanDerPol(mu=1.0)
    T = 0.5
    plant = rk4_control_interface(physics, T, substeps=20)

    y0 = torch.tensor([1.0, 0.0])
    y_ref = torch.zeros(n_steps, 2)
    Q = torch.diag(torch.tensor([10.0, 10.0]))
    R = torch.diag(torch.tensor([1.0]))
    N1, N2, Nu = 1, 5, 5
    u_min, u_max = [-1.0], [1.0]

    architectures = {
        "SLSQP (scipy, single-shoot)": None,
        "IPOPT (CasADi, single-shoot)": lambda: CasadiSingleShootingNMPC(
            model.step, 1, N1, N2, Nu, Q, R, u_min=u_min, u_max=u_max),
        "IPOPT (CasADi, multi-shoot)": lambda: CasadiMultipleShootingNMPC(
            model.step, 1, N1, N2, Nu, Q, R, u_min=u_min, u_max=u_max),
        "RTI-QP (CasADi, qrqp)": lambda: CasadiRTIController(
            model.step, 1, N1, N2, Nu, Q, R, u_min=u_min, u_max=u_max),
    }

    results = {}
    print(f"{'Architecture':<32}{'RMSE':>8}{'IAE':>10}{'time(s)':>10}")
    for name, ctor in architectures.items():
        controller = ctor() if ctor is not None else None
        t0 = time.time()
        y, u = run_nmpc_simulation(
            model.step, plant, y0, y_ref, control_dim=1,
            N1=N1, N2=N2, Nu=Nu, Q=Q, R=R, u_min=u_min, u_max=u_max,
            maxiter=30, controller=controller, desc=name, leave=False,
        )
        elapsed = time.time() - t0
        rmse, iae = integral_metrics(y_ref, y[1:])
        results[name] = (y, u, rmse, iae, elapsed)
        print(f"{name:<32}{rmse:>8.3f}{iae:>10.2f}{elapsed:>10.3f}")

    t_axis = torch.arange(n_steps + 1) * T
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for name, (y, u, rmse, iae, elapsed) in results.items():
        axes[0].plot(t_axis, y[:, 0], label=name)
        axes[1].step(t_axis[:-1], u[:, 0], where="post", label=name)
    axes[0].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[0].set_ylabel("x1")
    axes[0].set_title("NMPC architecture comparison (Van der Pol, PINC net)")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("control u")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("nmpc_architecture_comparison.png", dpi=150)
    plt.close()
    print("\nSaved nmpc_architecture_comparison.png")


if __name__ == "__main__":
    main()
