"""
Compares the MPC controller architectures available for the PDE-PINC
transient net (incompressible pipe flow, Sec. 5.1 of arXiv:2506.06188),
all driving the *same* trained `PINCTransientPDE` on the same
production-maximizing pressure-tracking task as
`train_incompressible_pde.py`'s `run_mpc_demo`:

  1. PDEMPCController        (nmpc_pde.py)         -- scipy SLSQP
  2. CasadiPDEMPC            (nmpc_pde_casadi.py)  -- IPOPT, single shooting
  3. CasadiPDERTIController  (nmpc_pde_casadi.py)  -- qrqp QP, linearized RTI

Each is a drop-in for the others: same constructor arguments (transient
model, x_bar, target, Np/Nc, rate limit, bounds), same `.solve(...)`
return signature, and all can be handed straight to
`run_pde_mpc_simulation` via its `controller=` argument.

No multiple-shooting variant exists on the PDE side -- see
`nmpc_pde_casadi.py`'s module docstring for why that's a deliberate
omission rather than a missing feature.

Run with:  python -m pinc.training.compare_nmpc_pde_architectures
(loads a trained PINC-SteadyState/Transient pair from the checkpoints
written by train_incompressible_pde.py -- run that script first, e.g.
`python -m pinc.training.train_incompressible_pde`; pass
--checkpoint-steady-state/--checkpoint-transient if you used non-default
paths there, or --k1/--k2 > 0 to train a fresh pair instead of loading)
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import torch

from pinc.physics.pde_incompressible_flow import IncompressibleFlowParams, IncompressibleFlowPhysics
from pinc.simulation.pipe_flow_plant import IncompressiblePipePlant
from pinc.simulation.rk4 import rk4_control_interface
from pinc.control.nmpc_pde import PDEMPCController, run_pde_mpc_simulation
from pinc.control.nmpc_pde_casadi import CasadiPDEMPC, CasadiPDERTIController
from pinc.training.train_incompressible_pde import train_steady_state, train_transient, U_RANGE
from pinc.utils.checkpoint import load_pinc_steady_state_pde_model, load_pinc_transient_pde_model

_DEFAULT_SS_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "pde_steady_state.pt")
_DEFAULT_TR_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "pde_transient.pt")


def _get_models(ss_checkpoint, tr_checkpoint, k1, k2):
    """
    Loads a trained PINC-SteadyState/Transient pair from the given
    checkpoint paths if both exist and k1/k2 weren't explicitly
    requested (>0); otherwise trains a fresh pair (k1/k2 > 0, or no
    checkpoint found) -- mirroring `compare_nmpc_architectures.py`'s
    `load_pinc_model(checkpoint_path)` on the ODE-PINC side, just for
    two chained models instead of one.
    """
    have_checkpoints = os.path.exists(ss_checkpoint) and os.path.exists(tr_checkpoint)

    if have_checkpoints and k1 == 0 and k2 == 0:
        print(f"Loading trained models from '{ss_checkpoint}' and '{tr_checkpoint}'...")
        ss_model, _ = load_pinc_steady_state_pde_model(ss_checkpoint, map_location="cpu")
        tr_model, _ = load_pinc_transient_pde_model(tr_checkpoint, map_location="cpu")
        return ss_model, tr_model

    physics = IncompressibleFlowPhysics(IncompressibleFlowParams())
    k1 = k1 or 800
    k2 = k2 or 400
    print(f"No usable checkpoints found (or training explicitly requested) -- "
          f"training a fresh pair (k1={k1}, k2={k2})...")
    ss_model, _ = train_steady_state(physics, k1_epochs=k1, k2_iters=k2)
    tr_model, _ = train_transient(physics, ss_model.to("cpu"), k1_epochs=k1, k2_iters=k2)
    return ss_model, tr_model


def main(ss_checkpoint=_DEFAULT_SS_CHECKPOINT, tr_checkpoint=_DEFAULT_TR_CHECKPOINT,
         k1=0, k2=0, n_steps=30, x_bar=0.1):
    physics = IncompressibleFlowPhysics(IncompressibleFlowParams())
    _, tr_model = _get_models(ss_checkpoint, tr_checkpoint, k1, k2)
    tr_model = tr_model.to("cpu")

    plant = IncompressiblePipePlant(physics)
    plant_step = rk4_control_interface(plant, T=1.0, substeps=20)

    target = torch.tensor(0.0)
    Np, Nc = 6, 2
    dy_max = 0.08
    u0_init = 0.9

    architectures = {
        "SLSQP (scipy)": None,
        "IPOPT (CasADi, single-shoot)": lambda: CasadiPDEMPC(
            tr_model, x_bar, target, Np, Nc, lambda_smooth=0.05,
            dy_max=dy_max, u_min=U_RANGE[0], u_max=U_RANGE[1]),
        "RTI-QP (CasADi, qrqp)": lambda: CasadiPDERTIController(
            tr_model, x_bar, target, Np, Nc, lambda_smooth=0.05,
            dy_max=dy_max, u_min=U_RANGE[0], u_max=U_RANGE[1]),
    }

    results = {}
    print(f"{'Architecture':<32}{'final P':>10}{'time(s)':>10}")
    for name, ctor in architectures.items():
        controller = ctor() if ctor is not None else None
        t0 = time.time()
        V_hist, P_hist, u_hist = run_pde_mpc_simulation(
            tr_model, plant_step, physics, x_bar=x_bar, target=target,
            u0_init=u0_init, n_steps=n_steps, Np=Np, Nc=Nc,
            lambda_smooth=0.05, dy_max=dy_max,
            u_min=U_RANGE[0], u_max=U_RANGE[1], maxiter=50,
            controller=controller, desc=name,
        )
        elapsed = time.time() - t0
        results[name] = (P_hist, u_hist, elapsed)
        print(f"{name:<32}{P_hist[-1].item():>10.4f}{elapsed:>10.3f}")

    t_axis = torch.arange(n_steps + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for name, (P_hist, u_hist, elapsed) in results.items():
        axes[0].plot(t_axis, P_hist, label=name)
        axes[1].step(t_axis[:-1], u_hist, where="post", label=name)
    axes[0].axhline(target.item(), color="black", linestyle=":", linewidth=1)
    axes[0].set_ylabel(r"$\tilde{P}(\bar{x})$")
    axes[0].set_title("PDE-MPC architecture comparison (incompressible pipe flow, PDE-PINC)")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("control u")
    axes[1].set_xlabel("window index k")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("pde_mpc_architecture_comparison.png", dpi=150)
    plt.close()
    print("\nSaved pde_mpc_architecture_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-steady-state", default=_DEFAULT_SS_CHECKPOINT)
    parser.add_argument("--checkpoint-transient", default=_DEFAULT_TR_CHECKPOINT)
    parser.add_argument("--k1", type=int, default=10000,
                         help="if > 0, force training a fresh pair for this many ADAM "
                              "epochs instead of loading from checkpoint")
    parser.add_argument("--k2", type=int, default=1000,
                         help="if > 0, force training a fresh pair for this many L-BFGS "
                              "iters instead of loading from checkpoint")
    parser.add_argument("--n-steps", type=int, default=30)
    args = parser.parse_args()
    main(ss_checkpoint=args.checkpoint_steady_state, tr_checkpoint=args.checkpoint_transient,
         k1=args.k1, k2=args.k2, n_steps=args.n_steps)