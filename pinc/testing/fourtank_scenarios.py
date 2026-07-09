"""
Scenario definitions for stress-testing a trained four-tank PINC model.

Two independent scenario sets are provided:

  - ROLLOUT scenarios: cheap, open-loop self-loop tests. Given a fixed
    initial state y0 and a control sequence u[0..n_steps-1], compare
    the PINC net's chained free-run prediction against RK4 ground
    truth. No optimization involved -- just repeated forward passes --
    so it's fine to run many of these.

  - NMPC scenarios: closed-loop control tests. Each defines a
    reference trajectory, cost weights, actuator bounds, and optional
    state constraints, then runs `run_nmpc_simulation` with the PINC
    net (and, optionally, an RK4 baseline) as the predictive model.
    These are much more expensive (one SLSQP solve per timestep), so
    the default set is smaller and every scenario carries a rough
    relative-cost hint.

Nothing here is executed at import time; `pinc.testing.run_fourtank_tests`
is the entry point that actually runs these.
"""
import torch


# ---------------------------------------------------------------------------
# Control-signal generators for rollout scenarios.
# Each returns a (n_steps, control_dim) tensor.
# ---------------------------------------------------------------------------

def const_u(n_steps, control_dim, value):
    """Same control value held for the whole horizon, e.g. value=(2.5, 2.5)."""
    return torch.tensor([value], dtype=torch.float32).repeat(n_steps, 1)[:, :control_dim]


def random_u(n_steps, control_dim, lo=0.0, hi=5.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.empty(n_steps, control_dim).uniform_(lo, hi, generator=g)


def step_u(n_steps, control_dim, low, high, switch_frac=0.5):
    """Held at `low` for the first switch_frac of the horizon, then jumps to `high`."""
    u = torch.zeros(n_steps, control_dim)
    switch = int(n_steps * switch_frac)
    u[:switch] = torch.tensor(low, dtype=torch.float32)
    u[switch:] = torch.tensor(high, dtype=torch.float32)
    return u


def bang_bang_u(n_steps, control_dim, low, high):
    """Alternates every step between `low` and `high` -- exercises fast input transitions."""
    u = torch.zeros(n_steps, control_dim)
    for k in range(n_steps):
        u[k] = torch.tensor(high if k % 2 == 0 else low, dtype=torch.float32)
    return u


def sinusoidal_u(n_steps, control_dim, mid=2.5, amp=2.4, periods=2.0):
    t = torch.linspace(0, periods * 2 * 3.14159265, n_steps)
    u = torch.stack([mid + amp * torch.sin(t + phase) for phase in
                      torch.linspace(0, 3.14159265 / 2, control_dim)], dim=-1)
    return u.clamp(0.0, 5.0)


def single_pump_u(n_steps, control_dim, active_idx, level):
    u = torch.zeros(n_steps, control_dim)
    u[:, active_idx] = level
    return u


# ---------------------------------------------------------------------------
# ROLLOUT scenarios
# ---------------------------------------------------------------------------
# Training data covers y in [2, 20] cm per tank and u in [0, 5] V (see
# pinc/datasets/fourtank.py). Scenarios tagged "extrapolation" deliberately
# start outside that range, since that's exactly the kind of input a real
# deployment could hand the model and it's worth knowing how it degrades.

def build_rollout_scenarios(n_steps=40):
    y_mid = torch.tensor([8.0, 8.0, 8.0, 8.0])
    y_low = torch.tensor([2.0, 2.0, 2.0, 2.0])
    y_high = torch.tensor([18.0, 18.0, 18.0, 18.0])
    y_uneven = torch.tensor([3.0, 15.0, 18.0, 4.0])

    return [
        dict(name="nominal_random", category="baseline",
             y0=y_mid, u_seq=random_u(n_steps, 2, seed=1), n_steps=n_steps,
             notes="Mid-range start, fresh random setpoint each step (same recipe as training-time validation)."),

        dict(name="nominal_constant_mid", category="control pattern",
             y0=y_mid, u_seq=const_u(n_steps, 2, (2.5, 2.5)), n_steps=n_steps,
             notes="Isolates compounding self-loop error from input-transition sensitivity."),

        dict(name="low_levels_constant", category="initial condition",
             y0=y_low, u_seq=const_u(n_steps, 2, (2.5, 2.5)), n_steps=n_steps,
             notes="Starts near the bottom of the trained range (2 cm)."),

        dict(name="high_levels_constant", category="initial condition",
             y0=y_high, u_seq=const_u(n_steps, 2, (2.5, 2.5)), n_steps=n_steps,
             notes="Starts near the top of the trained range (18 cm)."),

        dict(name="asymmetric_start", category="initial condition",
             y0=y_uneven, u_seq=random_u(n_steps, 2, seed=2), n_steps=n_steps,
             notes="Uneven tanks (some low, some high) rather than a symmetric start."),

        dict(name="extrapolation_near_empty", category="extrapolation",
             y0=torch.tensor([0.5, 0.5, 0.5, 0.5]), u_seq=const_u(n_steps, 2, (2.5, 2.5)),
             n_steps=n_steps,
             notes="Below the trained y-range [2, 20]; tests behavior near/at the sqrt(h) clamp."),

        dict(name="extrapolation_above_range", category="extrapolation",
             y0=torch.tensor([22.0, 22.0, 22.0, 22.0]), u_seq=const_u(n_steps, 2, (2.5, 2.5)),
             n_steps=n_steps,
             notes="Above the trained y-range [2, 20]."),

        dict(name="step_change_control", category="control pattern",
             y0=y_mid, u_seq=step_u(n_steps, 2, low=(0.5, 0.5), high=(4.5, 4.5)), n_steps=n_steps,
             notes="One clean step change in both channels halfway through the horizon."),

        dict(name="bang_bang_control", category="control pattern",
             y0=y_mid, u_seq=bang_bang_u(n_steps, 2, low=(0.0, 0.0), high=(5.0, 5.0)), n_steps=n_steps,
             notes="Full-range switching every single step -- the hardest input-transition case."),

        dict(name="sinusoidal_control", category="control pattern",
             y0=y_mid, u_seq=sinusoidal_u(n_steps, 2), n_steps=n_steps,
             notes="Smooth periodic input, phase-shifted between the two pumps."),

        dict(name="single_pump_only", category="control pattern",
             y0=y_mid, u_seq=single_pump_u(n_steps, 2, active_idx=0, level=4.0), n_steps=n_steps,
             notes="Only pump 1 active (u2=0 throughout) -- an asymmetric actuation regime."),

        dict(name="max_actuation", category="control pattern",
             y0=y_mid, u_seq=const_u(n_steps, 2, (5.0, 5.0)), n_steps=n_steps,
             notes="Both pumps pinned at their upper bound the whole time -- tests saturation behavior."),

        dict(name="min_actuation_drain", category="control pattern",
             y0=y_high, u_seq=const_u(n_steps, 2, (0.0, 0.0)), n_steps=n_steps,
             notes="Pumps off, high initial levels -- pure gravity drainage down toward zero."),
    ]


# ---------------------------------------------------------------------------
# NMPC (closed-loop) scenarios
# ---------------------------------------------------------------------------
# `cost_hint` is a rough relative-runtime label only (based on n_steps x
# maxiter x whether an ODE baseline is also solved) -- not a hard number.

def _step_ref(n_steps, h1_a, h2_a, h1_b, h2_b):
    y_ref = torch.zeros(n_steps, 4)
    y_ref[:, 0] = h1_a
    y_ref[:, 1] = h2_a
    y_ref[n_steps // 2:, 0] = h1_b
    y_ref[n_steps // 2:, 1] = h2_b
    return y_ref


def build_nmpc_scenarios(n_steps=30, maxiter=60):
    Q_default = torch.diag(torch.tensor([10.0, 10.0, 0.0, 0.0]))
    R_default = torch.diag(torch.tensor([1.0, 1.0]))
    u_min_full, u_max_full = [0.0, 0.0], [5.0, 5.0]
    y0_mid = torch.tensor([8.0, 8.0, 8.0, 8.0])
    y0_low = torch.tensor([2.5, 2.5, 2.5, 2.5])

    scenarios = [
        dict(name="baseline_step_reference", category="baseline", cost_hint="medium",
             y0=y0_mid, y_ref=_step_ref(n_steps, 10.0, 14.0, 8.0, 12.5),
             Q=Q_default, R=R_default, N1=1, N2=15, Nu=5,
             u_min=u_min_full, u_max=u_max_full,
             state_constraints=[(2, 0.6, 5.5), (3, 0.6, 5.5)],
             maxiter=maxiter, tracked_idx=[0, 1],
             notes="Same setup as train_fourtank.py's run_control_experiment, for a direct sanity check."),

        dict(name="aggressive_setpoint_jump", category="setpoint", cost_hint="medium",
             y0=y0_mid, y_ref=_step_ref(n_steps, 16.0, 18.0, 16.0, 18.0),
             Q=Q_default, R=R_default, N1=1, N2=15, Nu=5,
             u_min=u_min_full, u_max=u_max_full,
             state_constraints=[(2, 0.6, 5.5), (3, 0.6, 5.5)],
             maxiter=maxiter, tracked_idx=[0, 1],
             notes="Large jump toward the top of the operating range -- tests aggressive tracking, not just a small step."),

        dict(name="drain_down_setpoint", category="setpoint", cost_hint="medium",
             y0=y0_mid, y_ref=_step_ref(n_steps, 4.0, 5.0, 4.0, 5.0),
             Q=Q_default, R=R_default, N1=1, N2=15, Nu=5,
             u_min=u_min_full, u_max=u_max_full,
             state_constraints=[(2, 0.6, 5.5), (3, 0.6, 5.5)],
             maxiter=maxiter, tracked_idx=[0, 1],
             notes="Reference below the start -- the controller must mostly close pumps and let tanks drain."),

        dict(name="oscillating_reference", category="setpoint", cost_hint="high",
             y0=y0_mid,
             y_ref=torch.cat([
                 _step_ref(n_steps // 2, 12.0, 15.0, 8.0, 11.0),
                 _step_ref(n_steps - n_steps // 2, 12.0, 15.0, 8.0, 11.0),
             ], dim=0)[:n_steps],
             Q=Q_default, R=R_default, N1=1, N2=15, Nu=5,
             u_min=u_min_full, u_max=u_max_full,
             state_constraints=[(2, 0.6, 5.5), (3, 0.6, 5.5)],
             maxiter=maxiter, tracked_idx=[0, 1],
             notes="Reference double-steps rather than a single step -- more transitions for the controller to react to."),

        dict(name="tight_h3_h4_constraints", category="constraints", cost_hint="high",
             y0=y0_mid, y_ref=_step_ref(n_steps, 10.0, 14.0, 8.0, 12.5),
             Q=Q_default, R=R_default, N1=1, N2=15, Nu=5,
             u_min=u_min_full, u_max=u_max_full,
             state_constraints=[(2, 2.0, 3.5), (3, 2.0, 3.5)],
             maxiter=maxiter, tracked_idx=[0, 1],
             notes="Much narrower h3/h4 band than the paper's [0.6, 5.5] -- forces the constraint to actively bind."),

        dict(name="no_state_constraints", category="constraints", cost_hint="medium",
             y0=y0_mid, y_ref=_step_ref(n_steps, 10.0, 14.0, 8.0, 12.5),
             Q=Q_default, R=R_default, N1=1, N2=15, Nu=5,
             u_min=u_min_full, u_max=u_max_full,
             state_constraints=[],
             maxiter=maxiter, tracked_idx=[0, 1],
             notes="Same reference as baseline but h3/h4 completely unconstrained -- isolates constraint-handling cost/benefit."),

        dict(name="narrow_actuator_range", category="actuator limits", cost_hint="medium",
             y0=y0_mid, y_ref=_step_ref(n_steps, 10.0, 14.0, 8.0, 12.5),
             Q=Q_default, R=R_default, N1=1, N2=15, Nu=5,
             u_min=[1.0, 1.0], u_max=[4.0, 4.0],
             state_constraints=[(2, 0.6, 5.5), (3, 0.6, 5.5)],
             maxiter=maxiter, tracked_idx=[0, 1],
             notes="Pump voltages restricted to [1, 4] V instead of the full [0, 5] V range."),

        dict(name="far_from_reference_start", category="initial condition", cost_hint="medium",
             y0=y0_low, y_ref=_step_ref(n_steps, 12.0, 14.0, 12.0, 14.0),
             Q=Q_default, R=R_default, N1=1, N2=15, Nu=5,
             u_min=u_min_full, u_max=u_max_full,
             state_constraints=[(2, 0.6, 5.5), (3, 0.6, 5.5)],
             maxiter=maxiter, tracked_idx=[0, 1],
             notes="Starts near-empty (2.5 cm) with a mid/high fixed reference -- large initial tracking error to recover from."),

        dict(name="aggressive_tracking_weights", category="tuning", cost_hint="medium",
             y0=y0_mid, y_ref=_step_ref(n_steps, 10.0, 14.0, 8.0, 12.5),
             Q=torch.diag(torch.tensor([50.0, 50.0, 0.0, 0.0])),
             R=torch.diag(torch.tensor([0.1, 0.1])),
             N1=1, N2=15, Nu=5,
             u_min=u_min_full, u_max=u_max_full,
             state_constraints=[(2, 0.6, 5.5), (3, 0.6, 5.5)],
             maxiter=maxiter, tracked_idx=[0, 1],
             notes="High tracking weight / low control-effort penalty -- expect more aggressive, possibly noisier control moves."),

        dict(name="energy_conservative_weights", category="tuning", cost_hint="medium",
             y0=y0_mid, y_ref=_step_ref(n_steps, 10.0, 14.0, 8.0, 12.5),
             Q=torch.diag(torch.tensor([2.0, 2.0, 0.0, 0.0])),
             R=torch.diag(torch.tensor([5.0, 5.0])),
             N1=1, N2=15, Nu=5,
             u_min=u_min_full, u_max=u_max_full,
             state_constraints=[(2, 0.6, 5.5), (3, 0.6, 5.5)],
             maxiter=maxiter, tracked_idx=[0, 1],
             notes="Low tracking weight / high control-effort penalty -- expect slower, smoother, more sluggish tracking."),

        dict(name="short_horizon", category="tuning", cost_hint="low",
             y0=y0_mid, y_ref=_step_ref(n_steps, 10.0, 14.0, 8.0, 12.5),
             Q=Q_default, R=R_default, N1=1, N2=5, Nu=3,
             u_min=u_min_full, u_max=u_max_full,
             state_constraints=[(2, 0.6, 5.5), (3, 0.6, 5.5)],
             maxiter=maxiter, tracked_idx=[0, 1],
             notes="N2=5 (50s) lookahead -- per the training script's own notes, too short to see the delayed cross-coupled payoff; expect this one to struggle."),
    ]
    return scenarios


ROLLOUT_SCENARIO_NAMES = [s["name"] for s in build_rollout_scenarios()]
NMPC_SCENARIO_NAMES = [s["name"] for s in build_nmpc_scenarios()]