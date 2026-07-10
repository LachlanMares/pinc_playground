"""
Runs a battery of varied tests against a trained four-tank PINC checkpoint:

  1. ROLLOUT tests -- open-loop, self-loop generalization under varied
     initial tank levels and control-signal shapes (constant, step,
     bang-bang, sinusoidal, extrapolation outside the trained range...).

  2. NMPC tests -- closed-loop control under varied setpoints, state
     constraints, actuator limits, and cost-function tuning, each
     comparing the PINC-driven controller against one or more other
     architectures on the same plant (an RK4/ODE-driven baseline, and/or
     the CasADi-based NMPC controllers from `pinc.control.nmpc_casadi`).

Usage (from the repo root, i.e. the directory containing `pinc/`):

    python3 -m pinc.testing.run_fourtank_tests
    python3 -m pinc.testing.run_fourtank_tests --list
    python3 -m pinc.testing.run_fourtank_tests --rollout-only
    python3 -m pinc.testing.run_fourtank_tests --nmpc-only --skip-ode-baseline
    python3 -m pinc.testing.run_fourtank_tests --scenarios nominal_random,bang_bang_control
    python3 -m pinc.testing.run_fourtank_tests --quick
    python3 -m pinc.testing.run_fourtank_tests --nmpc-only --architectures pinc,pinc_casadi_multi
    python3 -m pinc.testing.run_fourtank_tests --nmpc-only --architectures pinc_casadi_rti

See --help for the full option list. Results (CSVs, per-scenario plots,
and a markdown summary report) are written to --out-dir (default
`fourtank_test_results/`).
"""
import argparse
import csv
import os
import time

import torch
import matplotlib.pyplot as plt

from pinc.physics.fourtank import FourTank
from pinc.utils.checkpoint import load_pinc_model
from pinc.evaluation.rollout import pinc_rollout
from pinc.simulation.rk4 import simulate, rk4_control_interface
from pinc.control.nmpc import run_nmpc_simulation
from pinc.control.nmpc_casadi import (
    CasadiSingleShootingNMPC,
    CasadiMultipleShootingNMPC,
    CasadiRTIController,
)
from pinc.testing.fourtank_scenarios import build_rollout_scenarios, build_nmpc_scenarios


# ---------------------------------------------------------------------------
# NMPC controller architectures
# ---------------------------------------------------------------------------
# Every architecture below shares the same predictive-model options (the
# trained PINC net, except "ode" which uses the true RK4 model instead) and
# the same scenario definitions from fourtank_scenarios.py -- only the
# solver/formulation changes. See pinc/control/nmpc_casadi.py for details on
# each CasADi variant.

ARCHITECTURE_LABELS = {
    "pinc": "PINC (SLSQP)",
    "ode": "ODE/RK (SLSQP)",
    "pinc_casadi_single": "PINC (IPOPT single-shoot)",
    "pinc_casadi_multi": "PINC (IPOPT multi-shoot)",
    "pinc_casadi_rti": "PINC (RTI-QP)",
}

# "pinc_casadi_rti" is deliberately left out of the default set: the
# four-tank orifice equation's sqrt(h) term has a steep, rapidly varying
# local gradient, and RTI's single-linearization-per-step approach reuses
# that gradient across the whole prediction horizon without ever
# re-deriving it from the true nonlinear model mid-horizon. In testing
# this reliably drove the QP infeasible partway through most timesteps on
# this system (see train_fourtank.py's run_control_experiment docstring
# for the same finding). Pass it explicitly via --architectures if you
# want to see that failure mode for yourself, or to compare it against
# the others anyway.
DEFAULT_NMPC_ARCHITECTURES = ("pinc", "ode", "pinc_casadi_single", "pinc_casadi_multi")

_CASADI_CONTROLLER_CLASSES = {
    "pinc_casadi_single": CasadiSingleShootingNMPC,
    "pinc_casadi_multi": CasadiMultipleShootingNMPC,
    "pinc_casadi_rti": CasadiRTIController,
}


def _build_control_interface_and_controller(kind, model, physics, T, control_dim, scenario):
    """
    Returns (control_interface, controller) for `run_nmpc_simulation`.
    `controller=None` means "let run_nmpc_simulation build the default
    scipy/SLSQP NMPCController itself" (the original behavior); a
    non-None controller is one of the CasADi drop-in replacements from
    nmpc_casadi.py, pre-configured for this scenario's cost/bounds.
    """
    if kind == "pinc":
        return model.step, None
    if kind == "ode":
        return rk4_control_interface(physics, T, substeps=5), None
    if kind not in _CASADI_CONTROLLER_CLASSES:
        raise ValueError(f"Unknown architecture: {kind!r}. Choose from {list(ARCHITECTURE_LABELS)}.")

    controller = _CASADI_CONTROLLER_CLASSES[kind](
        model.step, control_dim, scenario["N1"], scenario["N2"], scenario["Nu"],
        scenario["Q"], scenario["R"],
        u_min=scenario["u_min"], u_max=scenario["u_max"],
        state_constraints=scenario["state_constraints"],
    )
    return model.step, controller


# ---------------------------------------------------------------------------
# Rollout tests
# ---------------------------------------------------------------------------
# (Unaffected by --architectures: rollout tests are a direct PINC forward
# pass vs. RK4 ground truth, with no controller/optimizer involved.)

def run_rollout_scenario(model, physics, T, scenario, out_dir):
    y0 = scenario["y0"]
    u_seq = scenario["u_seq"]

    y_true = simulate(physics, y0, u_seq, dt=T, substeps=20)
    y_pred = pinc_rollout(model, y0, u_seq)

    err = y_pred - y_true
    mse = torch.mean(err ** 2).item()
    rmse = torch.sqrt(torch.mean(err ** 2)).item()
    max_abs_err = err.abs().max().item()
    per_tank_rmse = torch.sqrt(torch.mean(err ** 2, dim=0))
    final_abs_err = err[-1].abs()

    _plot_rollout(y_true, y_pred, T, scenario, out_dir)

    return dict(
        name=scenario["name"], category=scenario["category"],
        mse=mse, rmse=rmse, max_abs_err=max_abs_err,
        rmse_h1=per_tank_rmse[0].item(), rmse_h2=per_tank_rmse[1].item(),
        rmse_h3=per_tank_rmse[2].item(), rmse_h4=per_tank_rmse[3].item(),
        final_abs_err_max=final_abs_err.max().item(),
        notes=scenario.get("notes", ""),
    )


def _plot_rollout(y_true, y_pred, T, scenario, out_dir):
    t_axis = torch.arange(y_true.shape[0]) * T
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    labels = ["h1", "h2", "h3", "h4"]
    for i, ax in enumerate(axes.flat):
        ax.plot(t_axis, y_true[:, i], "k-", label="true (RK4)")
        ax.plot(t_axis, y_pred[:, i], "o--", color="tab:blue", markersize=3, label="PINC")
        ax.set_title(labels[i])
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Rollout: {scenario['name']}  ({scenario['category']})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"rollout_{scenario['name']}.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# NMPC tests
# ---------------------------------------------------------------------------

def run_nmpc_scenario(model, physics, T, scenario, out_dir,
                       architectures=DEFAULT_NMPC_ARCHITECTURES,
                       n_steps_override=None, maxiter_override=None):
    y0 = scenario["y0"]
    y_ref = scenario["y_ref"]
    if n_steps_override is not None:
        n_steps = min(n_steps_override, y_ref.shape[0])
        y_ref = y_ref[:n_steps]
    control_dim = physics.control_dim
    maxiter = maxiter_override or scenario["maxiter"]

    plant = rk4_control_interface(physics, T, substeps=20)
    tracked = scenario.get("tracked_idx", [0, 1])

    def integral_metrics(y_ref_full, y):
        err = (y_ref_full[:, tracked] - y[1:, tracked]).abs()
        iae = err.sum().item()
        rmse = torch.sqrt(torch.mean(err ** 2)).item()
        return rmse, iae

    def violation(y, constraints):
        if not constraints:
            return 0.0
        total = 0.0
        for idx, lo, hi in constraints:
            h = y[1:, idx]
            total += torch.clamp(lo - h, min=0).sum().item()
            total += torch.clamp(h - hi, min=0).sum().item()
        return total

    result = dict(name=scenario["name"], category=scenario["category"], notes=scenario.get("notes", ""))
    runs = {}

    for kind in architectures:
        control_interface, controller = _build_control_interface_and_controller(
            kind, model, physics, T, control_dim, scenario)

        t0 = time.time()
        y, u = run_nmpc_simulation(
            control_interface=control_interface, controller=controller,
            plant_step=plant, y0=y0, y_ref_full=y_ref, control_dim=control_dim,
            N1=scenario["N1"], N2=scenario["N2"], Nu=scenario["Nu"],
            Q=scenario["Q"], R=scenario["R"],
            u_min=scenario["u_min"], u_max=scenario["u_max"],
            state_constraints=scenario["state_constraints"],
            maxiter=maxiter, warm_start=True, leave=False,
            desc=f"{scenario['name']} [{ARCHITECTURE_LABELS[kind]}]", position=0,
        )
        elapsed = time.time() - t0

        rmse, iae = integral_metrics(y_ref, y)
        result[f"rmse_{kind}"] = rmse
        result[f"iae_{kind}"] = iae
        result[f"time_{kind}"] = elapsed
        result[f"violation_{kind}"] = violation(y, scenario["state_constraints"])
        runs[kind] = dict(y=y, u=u)

    _plot_nmpc(runs, y_ref, T, scenario, out_dir, architectures)

    return result


def _plot_nmpc(runs, y_ref, T, scenario, out_dir, architectures):
    n_steps = y_ref.shape[0]
    t_axis = torch.arange(n_steps + 1) * T
    tracked = scenario.get("tracked_idx", [0, 1])
    constraints = scenario["state_constraints"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["-", "--", "-.", ":"]
    other_idx = [i for i in range(4) if i not in tracked]

    for a, kind in enumerate(architectures):
        y, u = runs[kind]["y"], runs[kind]["u"]
        color = colors[a % len(colors)]
        label = ARCHITECTURE_LABELS[kind]

        for j, idx in enumerate(tracked):
            axes[0].plot(t_axis, y[:, idx], linestyles[j % len(linestyles)],
                         color=color, label=f"h{idx+1} ({label})")
        for j, idx in enumerate(other_idx):
            axes[1].plot(t_axis, y[:, idx], linestyles[j % len(linestyles)],
                         color=color, label=f"h{idx+1} ({label})")
        for d in range(u.shape[1]):
            axes[2].step(t_axis[:-1], u[:, d], linestyles[d % len(linestyles)], where="post",
                        color=color, label=f"u{d+1} ({label})")

    for idx in tracked:
        axes[0].plot(t_axis, torch.cat([y_ref[:, idx], y_ref[-1:, idx]]), ":", color="black", alpha=0.5)
    for idx, lo, hi in constraints:
        axes[1].axhline(lo, color="grey", linestyle=":")
        axes[1].axhline(hi, color="grey", linestyle=":")

    axes[0].set_ylabel("tracked levels")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].set_title(f"NMPC: {scenario['name']}  ({scenario['category']})")

    axes[1].set_ylabel("other levels" + (" (constrained)" if constraints else ""))
    axes[1].legend(fontsize=6, ncol=2)

    axes[2].set_ylabel("pump voltage")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"nmpc_{scenario['name']}.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(out_dir, rollout_rows, nmpc_rows, architectures):
    lines = ["# PINC four-tank test suite results\n"]

    if rollout_rows:
        lines.append("## Rollout (open-loop self-loop) tests\n")
        baseline = next((r for r in rollout_rows if r["name"] == "nominal_random"), rollout_rows[0])
        lines.append(f"Baseline (`{baseline['name']}`) RMSE: {baseline['rmse']:.4f}\n")
        lines.append("| Scenario | Category | RMSE | Max abs err | Final-step max err | Notes |")
        lines.append("|---|---|---|---|---|---|")
        for r in sorted(rollout_rows, key=lambda r: -r["rmse"]):
            flag = " ⚠️" if r["rmse"] > 3 * baseline["rmse"] else ""
            lines.append(f"| {r['name']}{flag} | {r['category']} | {r['rmse']:.4f} | "
                          f"{r['max_abs_err']:.4f} | {r['final_abs_err_max']:.4f} | {r['notes']} |")
        lines.append("\n⚠️ = RMSE more than 3x the nominal-random baseline -- worth a closer look.\n")

    if nmpc_rows:
        lines.append("\n## NMPC (closed-loop control) tests\n")
        lines.append(f"Architectures compared: {', '.join(ARCHITECTURE_LABELS[k] for k in architectures)}\n")
        rmse_headers = " | ".join(f"RMSE ({ARCHITECTURE_LABELS[k]})" for k in architectures)
        viol_headers = " | ".join(f"Viol. ({ARCHITECTURE_LABELS[k]})" for k in architectures)
        lines.append(f"| Scenario | Category | {rmse_headers} | {viol_headers} | Notes |")
        lines.append("|---|---|" + "---|" * len(architectures) + "---|" * len(architectures) + "---|")
        for r in nmpc_rows:
            rmse_cells = " | ".join(f"{r.get(f'rmse_{k}', float('nan')):.3f}" for k in architectures)
            viol_cells = " | ".join(f"{r.get(f'violation_{k}', float('nan')):.3f}" for k in architectures)
            lines.append(f"| {r['name']} | {r['category']} | {rmse_cells} | {viol_cells} | {r['notes']} |")

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="pinc/training/checkpoints/fourtank.pt")
    parser.add_argument("--out-dir", default="fourtank_test_results")
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--nmpc-only", action="store_true")
    parser.add_argument("--scenarios", default=None,
                         help="comma-separated scenario names to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list available scenarios and exit")
    parser.add_argument("--quick", action="store_true",
                         help="fast smoke test: fewer rollout steps, shorter NMPC horizons, "
                              "and drops 'ode' from --architectures")
    parser.add_argument("--architectures", default=None,
                         help="comma-separated NMPC controller architectures to compare per "
                              f"scenario, from {{{', '.join(ARCHITECTURE_LABELS)}}} "
                              f"(default: {','.join(DEFAULT_NMPC_ARCHITECTURES)}; "
                              "pinc_casadi_rti is opt-in only, see --list / README for why)")
    parser.add_argument("--skip-ode-baseline", action="store_true",
                         help="shorthand for dropping 'ode' from --architectures")
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--nmpc-steps", type=int, default=None)
    parser.add_argument("--nmpc-maxiter", type=int, default=None)
    args = parser.parse_args()

    rollout_steps = args.rollout_steps or (15 if args.quick else 40)
    nmpc_steps = args.nmpc_steps or (12 if args.quick else 30)
    nmpc_maxiter = args.nmpc_maxiter or (30 if args.quick else 60)

    architectures = tuple(args.architectures.split(",")) if args.architectures else DEFAULT_NMPC_ARCHITECTURES
    if args.skip_ode_baseline or args.quick:
        architectures = tuple(a for a in architectures if a != "ode")
    for kind in architectures:
        if kind not in ARCHITECTURE_LABELS:
            parser.error(f"Unknown architecture {kind!r} in --architectures; "
                         f"choose from {list(ARCHITECTURE_LABELS)}")

    rollout_scenarios = build_rollout_scenarios(n_steps=rollout_steps)
    nmpc_scenarios = build_nmpc_scenarios(n_steps=nmpc_steps, maxiter=nmpc_maxiter)

    if args.list:
        print("NMPC architectures available (--architectures):")
        for kind, label in ARCHITECTURE_LABELS.items():
            default_tag = "" if kind in DEFAULT_NMPC_ARCHITECTURES else "  (opt-in only)"
            print(f"  - {kind:<20} {label}{default_tag}")
        print("\nRollout scenarios:")
        for s in rollout_scenarios:
            print(f"  - {s['name']:<28} [{s['category']}] {s['notes']}")
        print("\nNMPC scenarios:")
        for s in nmpc_scenarios:
            print(f"  - {s['name']:<28} [{s['category']}] (cost: {s['cost_hint']}) {s['notes']}")
        return

    selected = set(args.scenarios.split(",")) if args.scenarios else None

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model from '{args.checkpoint}'...")
    model, _ = load_pinc_model(args.checkpoint, map_location="cpu")
    model.eval()
    physics = FourTank()
    T = model.T

    rollout_rows, nmpc_rows = [], []

    if not args.nmpc_only:
        print(f"\n=== Rollout tests ({rollout_steps} steps each) ===")
        for s in rollout_scenarios:
            if selected and s["name"] not in selected:
                continue
            print(f"  running: {s['name']} ...", end=" ", flush=True)
            row = run_rollout_scenario(model, physics, T, s, args.out_dir)
            print(f"RMSE={row['rmse']:.4f}  max_err={row['max_abs_err']:.4f}")
            rollout_rows.append(row)

    if not args.rollout_only:
        print(f"\n=== NMPC tests ({nmpc_steps} steps, maxiter={nmpc_maxiter}) ===")
        print(f"Architectures: {', '.join(ARCHITECTURE_LABELS[k] for k in architectures)}")
        for s in nmpc_scenarios:
            if selected and s["name"] not in selected:
                continue
            print(f"  running: {s['name']} [{s['cost_hint']} cost] ...")
            row = run_nmpc_scenario(model, physics, T, s, args.out_dir, architectures=architectures)
            summary = "  ".join(f"RMSE({k})={row[f'rmse_{k}']:.3f}" for k in architectures)
            print(f"    {summary}")
            nmpc_rows.append(row)

    if rollout_rows:
        _write_csv(os.path.join(args.out_dir, "rollout_results.csv"), rollout_rows,
                   fieldnames=["name", "category", "mse", "rmse", "max_abs_err",
                               "rmse_h1", "rmse_h2", "rmse_h3", "rmse_h4",
                               "final_abs_err_max", "notes"])
    if nmpc_rows:
        fieldnames = ["name", "category"]
        for kind in architectures:
            fieldnames += [f"rmse_{kind}", f"iae_{kind}", f"time_{kind}", f"violation_{kind}"]
        fieldnames.append("notes")
        _write_csv(os.path.join(args.out_dir, "nmpc_results.csv"), nmpc_rows, fieldnames=fieldnames)

    _write_report(args.out_dir, rollout_rows, nmpc_rows, architectures)
    print(f"\nDone. Results written to '{args.out_dir}/' (CSVs, per-scenario PNGs, report.md).")


if __name__ == "__main__":
    main()