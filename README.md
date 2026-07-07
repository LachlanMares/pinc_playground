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
