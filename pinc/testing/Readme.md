# Four-tank PINC test suite — what each test is for

This suite checks a trained four-tank PINC checkpoint against two different
jobs it needs to do well:

1. **Predict** — given a starting state and a control input, can the network's
   self-loop rollout track the true (RK4) plant over a long horizon?
2. **Control** — when that same network is used as the predictive model inside
   NMPC, does the resulting closed-loop controller behave sensibly, and how
   does it compare to using the real ODE as the predictive model instead?

Every scenario below exists to isolate one specific way those two jobs can go
wrong. Run `python -m pinc.testing.run_fourtank_tests --list` to see this same
information from the code itself.

---

## Rollout tests (open-loop self-loop prediction)

These feed a fixed control sequence to both the true plant (RK4) and the
PINC net's self-loop, then compare trajectories. No control/optimization is
involved — it's purely "does the network's own forward simulation match
reality." Cheap to run, so there's a wide spread of conditions here.

| Scenario | What it's testing |
|---|---|
| `nominal_random` | The baseline case: mid-range start, fresh random setpoint every step. This is the same recipe used for validation during training, so it's the reference point everything else gets compared against — if other scenarios are much worse than this one, that's the signal worth chasing. |
| `nominal_constant_mid` | Same starting point, but one unchanging input instead of a new random one each step. Separates two failure modes that are otherwise tangled together: does error grow just from chaining predictions step after step (this test), or does the network specifically struggle when the *input changes* (the random-signal test above)? If this one is clean but `nominal_random` isn't, look at input-transition coverage in training, not the self-loop mechanism itself. |
| `low_levels_constant` | Starts near the *bottom* of the trained range (2 cm — the lower edge of `[2, 20]` used during training). Checks the network hasn't just learned the middle of the range well while the edges are shakier. |
| `high_levels_constant` | Same idea, near the *top* of the trained range (18 cm). |
| `asymmetric_start` | Tanks start at very different levels (3, 15, 18, 4 cm) rather than a symmetric all-equal start. Training data is sampled independently per tank, but it's worth confirming an unusual, lopsided combination still works, not just the "nice" symmetric ones a hand-picked test might default to. |
| `extrapolation_near_empty` | Starts at 0.5 cm — *below* the trained range entirely. Tank levels can't go negative and the physics has a clamp at zero (to avoid `sqrt` of a negative number), so this checks the network hasn't learned something brittle right at that boundary. This is the kind of input a real deployment could hand the model by accident (e.g. after a drain event), so it's worth knowing how gracefully it degrades rather than assuming it'll never happen. |
| `extrapolation_above_range` | Starts at 22 cm — *above* the trained range. Same idea as above, testing the opposite edge. |
| `step_change_control` | Input held low, then jumps once to a much higher value halfway through. A single, clean transition — checks the network reacts correctly to a control change without needing constant novelty. |
| `bang_bang_control` | Input switches between minimum and maximum every single step. This is the hardest input-transition case there is — if `nominal_constant_mid` is clean but this one isn't, the network is fine at chaining predictions but struggles specifically with rapid, large input transitions. |
| `sinusoidal_control` | Smooth, continuously-varying periodic input, phase-shifted between the two pumps. A middle ground between "constant" and "bang-bang" — checks behavior under a input that's always changing but never abruptly. |
| `single_pump_only` | Only pump 1 is active; pump 2 stays at zero the whole time. Training samples both pumps independently across their full range, so this checks an asymmetric, one-actuator-idle combination the network may have seen less often than "both pumps doing something." |
| `max_actuation` | Both pumps pinned at their absolute maximum (5V) for the whole run. Tests behavior at the saturated edge of the actuator range, where inflow is as large as the physics allows. |
| `min_actuation_drain` | Pumps fully off, starting from *high* tank levels. Pure gravity drainage down toward zero, with no inflow to counteract it — the mirror image of `max_actuation`, and another way of probing behavior as levels approach (and cross near) the low edge of the trained range. |

**How to read the results:** `report.md` sorts these by RMSE and flags (⚠️) any
scenario whose RMSE is more than 3x the `nominal_random` baseline. That's a
relative flag, not a pass/fail line — a scenario 3x worse than baseline but
still numerically tiny in absolute terms may not matter for your application,
so check `max_abs_err` and the per-scenario PNG (true vs. predicted, all four
tanks) before deciding something's actually wrong.

---

## NMPC tests (closed-loop control)

These wrap the same PINC net inside the NMPC controller from Antonelo et al.
and run a full closed-loop simulation: at every timestep, solve for the best
control move using the predictive model, apply it to the *real* RK4 plant,
measure the outcome, repeat. Each scenario also runs the identical NMPC setup
with the real ODE as the predictive model instead of PINC, so you get a direct
"how much does using the learned network cost you, if anything" comparison on
every scenario. These are much more expensive than the rollout tests (one
constrained optimization solve per timestep), so there are fewer of them.

| Scenario | What it's testing |
|---|---|
| `baseline_step_reference` | The reference case, matching the setup in `train_fourtank.py`'s own control experiment (and, in spirit, Table 1 / Fig. 13-14 of the paper): track a step change in h1/h2 while keeping h3/h4 inside a wide `[0.6, 5.5]` cm band. Everything else is a variation on this. |
| `aggressive_setpoint_jump` | Reference jumps to *near the top* of the operating range (16-18 cm) instead of a modest step. Checks tracking quality isn't just good for small, easy moves. |
| `drain_down_setpoint` | Reference is *below* the starting levels, so the controller mostly has to close pumps and let tanks drain rather than fill them. The opposite control direction from the baseline test. |
| `oscillating_reference` | Reference double-steps within the horizon instead of changing once. More transitions to react to, closer to a real setpoint schedule than a single step. |
| `tight_h3_h4_constraints` | Same reference as baseline, but h3/h4 are boxed into a much narrower `[2.0, 3.5]` cm band (vs. the paper's `[0.6, 5.5]`). This forces the constraint to actively bind rather than sit comfortably satisfied in the background, which is a much harder test of the constraint-handling machinery — and, because h3/h4 are exactly the *indirectly*-coupled tanks (Sec. 4.2.2's non-minimum-phase point), a good place to see whether PINC's predictions are accurate enough to respect a tight bound or whether it needs the ODE's precision to avoid violating it. |
| `no_state_constraints` | Same reference as baseline with h3/h4 constraints removed entirely. Isolates what the state constraints are actually costing/buying you — compare this scenario's tracking RMSE and control moves against `baseline_step_reference` to see the effect the constraint alone has. |
| `narrow_actuator_range` | Pump voltage restricted to `[1, 4]` V instead of the full `[0, 5]` V. Checks the controller still finds reasonable moves when it has less headroom to work with — a more realistic constraint if the real pumps can't safely run fully open or fully closed. |
| `far_from_reference_start` | Starts near-empty (2.5 cm) with a mid/high fixed reference, i.e. a large initial tracking error to recover from, rather than starting already close to the target. Checks large-transient behavior, not just fine-tuning around a nearby setpoint. |
| `aggressive_tracking_weights` | Same reference as baseline, but the cost function is retuned toward "track hard, don't worry about control effort" (high Q, low R). Expect larger, more aggressive control moves — watch whether PINC's predictions stay reliable enough to support that more aggressive behavior, or whether inaccuracies get amplified into control chatter. |
| `energy_conservative_weights` | The opposite retune: "smooth, cheap control, don't chase the reference too hard" (low Q, high R). Expect slower, more sluggish tracking — a check that the controller (and PINC's predictions feeding it) still behaves sensibly at the other end of the tuning spectrum. |
| `short_horizon` | Prediction horizon cut to `N2=5` (50s lookahead) instead of 15 (150s). `train_fourtank.py`'s own notes call this out as too short to see the delayed, cross-coupled payoff of the non-minimum-phase dynamics — this scenario exists specifically to confirm that expected failure mode still shows up (e.g. a pump getting stuck pinned at a bound), rather than assuming it from the comment alone. |

**How to read the results:** `report.md` lists RMSE and constraint-violation
totals for PINC and, where run, the ODE baseline, side by side. A few things
worth specifically comparing:
- **PINC RMSE vs. ODE RMSE** — if PINC is dramatically worse than ODE on a
  scenario where it was fine on `baseline_step_reference`, that scenario's
  operating region is where the learned model's accuracy is breaking down.
- **Violation columns** — any nonzero number means the constrained state (h3
  or h4) went outside its bound at some point in the simulation; compare PINC's
  violation total to the ODE baseline's to see whether that's a controller
  problem (both violate) or specifically a PINC-accuracy problem (PINC
  violates, ODE doesn't).
- **The per-scenario PNG** — shows tracked levels vs. reference, the other
  (possibly constrained) levels vs. their bounds, and the actual control
  signal for both PINC and ODE overlaid, so you can see *how* a difference in
  the numbers actually plays out, not just the summary statistic.

---

## Adding your own scenario

Both scenario lists are plain Python dicts in `fourtank_scenarios.py`
(`build_rollout_scenarios()` / `build_nmpc_scenarios()`) — copy an existing
entry, change the fields that matter (`y0`, `u_seq`, `y_ref`, constraints,
weights...), give it a name and a one-line note explaining what it's
checking, and it'll show up automatically in `--list`, the CSVs, and the
report next time you run the suite.