"""
Runs a battery of varied tests against a trained four-tank PINC checkpoint:

  1. ROLLOUT tests -- open-loop, self-loop generalization under varied
     initial tank levels and control-signal shapes (constant, step,
     bang-bang, sinusoidal, extrapolation outside the trained range...).

  2. NMPC tests -- closed-loop control under varied setpoints, state
     constraints, actuator limits, and cost-function tuning, each
     comparing the PINC-driven controller against an RK4/ODE-driven
     one on the same plant.

Usage (from the repo root, i.e. the directory containing `pinc/`):

    python -m pinc.testing.run_fourtank_tests
    python -m pinc.testing.run_fourtank_tests --list
    python -m pinc.testing.run_fourtank_tests --rollout-only
    python -m pinc.testing.run_fourtank_tests --nmpc-only --skip-ode-baseline
    python -m pinc.testing.run_fourtank_tests --scenarios nominal_random,bang_bang_control
    python -m pinc.testing.run_fourtank_tests --quick

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
from pinc.testing.fourtank_scenarios import build_rollout_scenarios, build_nmpc_scenarios


# ---------------------------------------------------------------------------
# Rollout tests
# ---------------------------------------------------------------------------

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

def run_nmpc_scenario(model, physics, T, scenario, out_dir, run_ode=True, n_steps_override=None,
                       maxiter_override=None):
    y0 = scenario["y0"]
    y_ref = scenario["y_ref"]
    if n_steps_override is not None:
        n_steps = min(n_steps_override, y_ref.shape[0])
        y_ref = y_ref[:n_steps]
    control_dim = physics.control_dim
    maxiter = maxiter_override or scenario["maxiter"]

    plant = rk4_control_interface(physics, T, substeps=20)

    common = dict(
        plant_step=plant, y0=y0, y_ref_full=y_ref, control_dim=control_dim,
        N1=scenario["N1"], N2=scenario["N2"], Nu=scenario["Nu"],
        Q=scenario["Q"], R=scenario["R"],
        u_min=scenario["u_min"], u_max=scenario["u_max"],
        state_constraints=scenario["state_constraints"],
        maxiter=maxiter, warm_start=True, leave=False,
    )

    t0 = time.time()
    y_pinc, u_pinc = run_nmpc_simulation(control_interface=model.step, desc=f"{scenario['name']} [PINC]",
                                          position=0, **common)
    t_pinc = time.time() - t0

    result = dict(name=scenario["name"], category=scenario["category"], notes=scenario.get("notes", ""))

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

    rmse_pinc, iae_pinc = integral_metrics(y_ref, y_pinc)
    result.update(rmse_pinc=rmse_pinc, iae_pinc=iae_pinc, time_pinc=t_pinc,
                   violation_pinc=violation(y_pinc, scenario["state_constraints"]))

    y_ode, u_ode = None, None
    if run_ode:
        ode_interface = rk4_control_interface(physics, T, substeps=5)
        t0 = time.time()
        y_ode, u_ode = run_nmpc_simulation(control_interface=ode_interface, desc=f"{scenario['name']} [ODE]",
                                            position=0, **common)
        t_ode = time.time() - t0
        rmse_ode, iae_ode = integral_metrics(y_ref, y_ode)
        result.update(rmse_ode=rmse_ode, iae_ode=iae_ode, time_ode=t_ode,
                      violation_ode=violation(y_ode, scenario["state_constraints"]))

    _plot_nmpc(y_pinc, u_pinc, y_ode, u_ode, y_ref, T, scenario, out_dir)

    return result


def _plot_nmpc(y_pinc, u_pinc, y_ode, u_ode, y_ref, T, scenario, out_dir):
    n_steps = y_ref.shape[0]
    t_axis = torch.arange(n_steps + 1) * T
    tracked = scenario.get("tracked_idx", [0, 1])
    constraints = scenario["state_constraints"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    for c, idx in enumerate(tracked):
        axes[0].plot(t_axis, y_pinc[:, idx], color=colors[c % len(colors)], label=f"h{idx+1} (PINC)")
        if y_ode is not None:
            axes[0].plot(t_axis, y_ode[:, idx], "--", color=colors[c % len(colors)], alpha=0.6, label=f"h{idx+1} (ODE)")
        axes[0].plot(t_axis, torch.cat([y_ref[:, idx], y_ref[-1:, idx]]), ":", color="black", alpha=0.5)
    axes[0].set_ylabel("tracked levels")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title(f"NMPC: {scenario['name']}  ({scenario['category']})")

    other_idx = [i for i in range(4) if i not in tracked]
    for c, idx in enumerate(other_idx):
        axes[1].plot(t_axis, y_pinc[:, idx], color=colors[c % len(colors)], label=f"h{idx+1} (PINC)")
        if y_ode is not None:
            axes[1].plot(t_axis, y_ode[:, idx], "--", color=colors[c % len(colors)], alpha=0.6, label=f"h{idx+1} (ODE)")
    for idx, lo, hi in constraints:
        axes[1].axhline(lo, color="grey", linestyle=":")
        axes[1].axhline(hi, color="grey", linestyle=":")
    axes[1].set_ylabel("other levels" + (" (constrained)" if constraints else ""))
    axes[1].legend(fontsize=7, ncol=2)

    control_dim = u_pinc.shape[1]
    for d in range(control_dim):
        axes[2].step(t_axis[:-1], u_pinc[:, d], where="post", color=colors[d % len(colors)], label=f"u{d+1} (PINC)")
        if u_ode is not None:
            axes[2].step(t_axis[:-1], u_ode[:, d], where="post", linestyle="--", color=colors[d % len(colors)],
                         alpha=0.6, label=f"u{d+1} (ODE)")
    axes[2].set_ylabel("pump voltage")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=7, ncol=2)

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


def _write_report(out_dir, rollout_rows, nmpc_rows):
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
        lines.append("| Scenario | Category | RMSE (PINC) | RMSE (ODE) | Viol. (PINC) | Viol. (ODE) | Notes |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in nmpc_rows:
            rmse_ode = f"{r.get('rmse_ode'):.3f}" if "rmse_ode" in r else "n/a"
            viol_ode = f"{r.get('violation_ode'):.3f}" if "violation_ode" in r else "n/a"
            lines.append(f"| {r['name']} | {r['category']} | {r['rmse_pinc']:.3f} | {rmse_ode} | "
                          f"{r['violation_pinc']:.3f} | {viol_ode} | {r['notes']} |")

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
                         help="fast smoke test: fewer rollout steps, shorter NMPC horizons, no ODE baseline")
    parser.add_argument("--skip-ode-baseline", action="store_true",
                         help="only run the PINC-driven NMPC controller (roughly halves NMPC runtime)")
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--nmpc-steps", type=int, default=None)
    parser.add_argument("--nmpc-maxiter", type=int, default=None)
    args = parser.parse_args()

    rollout_steps = args.rollout_steps or (15 if args.quick else 40)
    nmpc_steps = args.nmpc_steps or (12 if args.quick else 30)
    nmpc_maxiter = args.nmpc_maxiter or (30 if args.quick else 60)
    run_ode = not (args.skip_ode_baseline or args.quick)

    rollout_scenarios = build_rollout_scenarios(n_steps=rollout_steps)
    nmpc_scenarios = build_nmpc_scenarios(n_steps=nmpc_steps, maxiter=nmpc_maxiter)

    if args.list:
        print("Rollout scenarios:")
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
        print(f"\n=== NMPC tests ({nmpc_steps} steps, maxiter={nmpc_maxiter}, "
              f"ODE baseline={'on' if run_ode else 'off'}) ===")
        for s in nmpc_scenarios:
            if selected and s["name"] not in selected:
                continue
            print(f"  running: {s['name']} [{s['cost_hint']} cost] ...")
            row = run_nmpc_scenario(model, physics, T, s, args.out_dir, run_ode=run_ode)
            ode_str = f"  RMSE(ODE)={row['rmse_ode']:.3f}" if "rmse_ode" in row else ""
            print(f"    RMSE(PINC)={row['rmse_pinc']:.3f}  viol(PINC)={row['violation_pinc']:.3f}{ode_str}")
            nmpc_rows.append(row)

    if rollout_rows:
        _write_csv(os.path.join(args.out_dir, "rollout_results.csv"), rollout_rows,
                   fieldnames=["name", "category", "mse", "rmse", "max_abs_err",
                               "rmse_h1", "rmse_h2", "rmse_h3", "rmse_h4",
                               "final_abs_err_max", "notes"])
    if nmpc_rows:
        fieldnames = ["name", "category", "rmse_pinc", "iae_pinc", "time_pinc", "violation_pinc"]
        if run_ode:
            fieldnames += ["rmse_ode", "iae_ode", "time_ode", "violation_ode"]
        fieldnames.append("notes")
        _write_csv(os.path.join(args.out_dir, "nmpc_results.csv"), nmpc_rows, fieldnames=fieldnames)

    _write_report(args.out_dir, rollout_rows, nmpc_rows)
    print(f"\nDone. Results written to '{args.out_dir}/' (CSVs, per-scenario PNGs, report.md).")


if __name__ == "__main__":
    main()