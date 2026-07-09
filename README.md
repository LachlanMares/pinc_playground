# PINC — Physics-Informed Neural Nets for Control

Implementation of Antonelo et al., *"Physics-Informed Neural Nets for
Control of Dynamical Systems"* (arXiv:2104.02556), built on top of the
scaffolding in this repo.

## Running it

```bash
pip install -r requirements.txt

python -m pinc.training.train_vanderpol
python -m pinc.training.train_fourtank
```

Each script trains a PINC net, then saves three figures to the
current directory (training curves, long-range self-loop prediction,
and NMPC control trajectories), and prints RMSE/IAE/timing metrics
comparing PINC-driven NMPC against ODE/RK4-driven NMPC.

The defaults (`k1_epochs=500`, `k2_iters=2000`) are much smaller than
the paper's (which trains for tens of thousands of L-BFGS iterations,
see Fig. 8) so a full run finishes in a few minutes rather than hours.
Increase them for higher-fidelity reproduction of the paper's numbers.

## Notes / simplifications vs. the paper

- **NMPC solver**: the paper mentions SQP/interior-point solvers
  (SNOPT, IPOPT); this repo uses `scipy.optimize.minimize(method="SLSQP")`
  with autograd-supplied gradients, which is slower per-solve but
  requires no extra native dependencies. For long control simulations,
  consider lowering `N2`/`Nu`, the RK sub-steps, or `maxiter` in
  `nmpc.py` if you need faster iteration during experimentation.
- **Four-tank parameters** default to the classic non-minimum-phase
  operating point commonly cited for Johansson (2000); the paper
  doesn't reprint the exact numeric table, so double check against
  your reference if exact numbers matter.
- **State constraints** in `NMPCController` are simple box constraints
  on chosen state indices (sufficient for the `h3`,`h4` bounds used in
  Sec. 4.2.2); Eq. (12e)/(12f)'s fully general `h`/`g` constraint
  functions aren't modeled beyond that.
- Robustness/sensitivity-to-perturbation experiments (Sec. 4.2.3,
  Figs. 15-16) and the traditional-PINN-vs-PINC long-range comparison
  (Fig. 6-7) aren't reproduced, but everything needed to build them is
  now in place (swap `physics` parameters for the sensitivity study;
  train a `t`-only `MLP` with `y0`/`u` fixed for the traditional-PINN
  comparison).

## PDE extension: incompressible single-phase pipe flow

This repo also implements the PDE generalization of PINC introduced in
Miyatake et al., *"Physics-Informed Neural Networks for Control of
Single-Phase Flow Systems Governed by Partial Differential Equations"*
(arXiv:2506.06188). The ODE-PINC above conditions a single network on
`(t, y(0), u)`; the PDE-PINC instead trains **two** networks:

- **PINC-SteadyState**, `y(x, u) = f_w(x, u)` (Sec. 4.3): learns
  equilibrium pressure/velocity profiles across a whole family of
  controls, with no time input at all.
- **PINC-Transient**, `y(x, t, u0, u) = f_w(x, t, u0, u)` (Sec. 4.4):
  learns the dynamic response within one control window, conditioned on
  the *previous* window's control `u0` (used as a compact stand-in for
  the initial condition, instead of the full spatially-resolved state)
  and the *current* window's control `u`. The already-trained,
  frozen `PINC-SteadyState` net supplies the initial-condition targets
  during training (Eq. 34) -- there's no autoregressive feedback at
  inference time, so errors don't accumulate across windows the way
  they can for the ODE-PINC's self-loop mode.

Currently only the **incompressible flow** case (Sec. 2.3) is
implemented; the compressible/gas case (Sec. 2.4, 5.2) needs an
equation of state, a much larger Optuna-tuned architecture (skip
connections, Swish activations -- Sec. 5.2.1), and isn't included yet
(see "Extending further" below).

### Running it

```bash
python -m pinc.training.train_incompressible_pde
```

This trains both nets (defaults: `k1_epochs=300`, `k2_iters=300` for a
few-minute run; bump these up, e.g. `--k1 800 --k2 400`, for the ~1%
steady-state MAPE / ~1e-3 transient MSE this repo achieves in testing),
then saves five figures:

- `incompressible_pde_steady_state_training.png` / `_transient_training.png`
  -- PDE / BC / IC loss curves (mirrors Fig. 7/9 of the paper).
- `incompressible_pde_steady_state.png` -- PINC-SteadyState pressure and
  velocity profiles vs. the exact analytic steady state, for several
  control values (mirrors Fig. 8).
- `incompressible_pde_transient_openloop.png` -- chained,
  no-feedback-across-windows forward simulation vs. the exact plant
  (mirrors Fig. 10).
- `incompressible_pde_mpc_control.png` -- closed-loop MPC tracking an
  unattainable low target pressure at a fixed "PDG" sensor position,
  subject to a per-window rate constraint (mirrors Fig. 11).

### The key simplification that makes this tractable

For **incompressible** flow, mass conservation forces `dV/dx = 0`
*identically* (Eq. 12-13 of the paper), not just at steady state.
Combined with the momentum equation, this means the whole PDE system
collapses exactly onto a single scalar ODE for `V(t)` (spatially
uniform), driven by the control `u(t)`, with the full pressure profile
recovered afterwards as an algebraic, x-linear expression. This is
implemented in `pinc/physics/pde_incompressible_flow.py` /
`pinc/simulation/pipe_flow_plant.py`, and lets the "ground truth" plant
reuse this repo's existing `pinc/simulation/rk4.py` integrator exactly
as the ODE-PINC case does -- no new finite-difference solver needed,
and the ground truth is *exact* rather than a discretization
approximation.

### Architecture additions

| File | Role |
|---|---|
| `pinc/physics/pde_incompressible_flow.py` | Normalized PDE residuals, BCs, friction (Blasius), and the exact scalar-ODE plant reduction |
| `pinc/models/pinc_pde.py` | `PINCSteadyStatePDE`, `PINCTransientPDE` network wrappers |
| `pinc/datasets/pde_incompressible.py` | LHS samplers for PDE collocation / BC / IC points (Sec. 4.3/4.4.1) |
| `pinc/losses/pinc_pde_loss.py` | `SteadyStatePDELoss`, `TransientPDELoss` (the latter consumes the frozen steady-state net for IC targets) |
| `pinc/core/pde_trainer.py` | Generic ADAM-then-L-BFGS trainer shared by both stages |
| `pinc/simulation/pipe_flow_plant.py` | `IncompressiblePipePlant`, a `PhysicsModel` for the exact reduced-ODE ground truth |
| `pinc/control/nmpc_pde.py` | `PDEMPCController` / `run_pde_mpc_simulation`, implementing Eq. 37 / Algorithm 2 |
| `pinc/training/train_incompressible_pde.py` | End-to-end script tying the above together |
| `pinc/utils/autodiff.py` | `time_derivative` generalized to a coordinate-agnostic `derivative(y, x)` (reused for both `d/dt` and `d/dx`) |

### Notes / simplifications vs. the PDE paper

- **Window length == MPC sampling period.** The paper allows the PINC
  window `T` (`tref`) to be much longer than the MPC sampling period
  `Ts`, querying the transient net at intermediate `t` inside each MPC
  step (Fig. 5/10). Here `tref` is set equal to `Ts`, so one network
  call (`t=1`) is exactly one control step -- this reuses the same
  "network step == one MPC step" pattern as the ODE-PINC/NMPC code
  above, at the cost of not demonstrating intra-window continuous-time
  querying (which `PINCTransientPDE.forward` still supports for any
  `t`, if you want to explore it).
- **MPC solver**: implemented from scratch in `nmpc_pde.py` (not a
  reuse of `NMPCController`), because the PDE-PINC's forward rollout
  propagates the *previous control* rather than a fed-back predicted
  *state* (Algorithm 1) -- a structurally different recursion from the
  ODE-PINC's `f(y, u) -> y_next`. Uses `scipy.optimize.minimize(method="SLSQP")`
  with autograd gradients/Jacobians, same as `nmpc.py`.
- **No error-correction filtering.** Sec. 4.5.1/5.1.2 note that model-plant
  mismatch is corrected only through re-solving with the true previous
  control at each step (Algorithm 2); the optional Kalman-style
  correction (Jordanou et al., 2022) mentioned in the paper isn't
  implemented.
- **Compressible/gas flow** (Sec. 2.4, 5.2) is not implemented: it
  needs an equation of state (ideal gas law), a genuine finite-difference
  "plant" (the `dV/dx=0` shortcut above no longer holds), and the
  Optuna-tuned deeper architecture with skip connections described in
  Sec. 5.2.1. The module layout mirrors the ODE case closely enough
  that adding a `pde_compressible_flow.py` physics module and a
  `PINCTransientPDE`-compatible finite-difference ground truth should
  slot in the same way the incompressible case did here.

### Extending further

- **Compressible/gas flow**: add `pinc/physics/pde_compressible_flow.py`
  with the ideal-gas EOS and the coupled mass/momentum residuals of
  Sec. 2.4.1, a semi-implicit finite-difference "plant" (Sec. 3, Harlow
  & Welch staggered grid) since the exact scalar-ODE shortcut no longer
  applies, and reuse `PINCSteadyStatePDE`/`PINCTransientPDE` with 3
  outputs `(P, V, rho)` or keep 2 outputs and derive `rho` from the EOS
  in the forward pass, as the paper does.
- **Optuna hyperparameter search** (Sec. 5.2.1): swap `PDETrainer.fit`'s
  fixed architecture for an Optuna `study.optimize` loop over
  hidden-size/depth/activation, following the same pattern as
  `train_vanderpol.py`'s grid search (Fig. 5) but with Optuna's
  sampler instead of a manual grid.
