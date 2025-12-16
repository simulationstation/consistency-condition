"""
TSE CLI Entry Point

Run with: python -m tse [options]
"""

import argparse
import json
import sys
from pathlib import Path
import time

import numpy as np

from .model import TSEParameters, SCENARIOS, get_scenario_params
from .integrate import compute_capacity, check_nsteps_convergence
from .experiments import (
    run_tmax_convergence,
    run_all_tmax_convergence,
    run_exponent_heatmap,
    run_epsilon_sweep,
    compute_scenario_diagnostics,
    ConvergenceResult,
)
from .plot import (
    plot_tmax_convergence,
    plot_all_tmax_convergence,
    plot_heatmap,
    plot_epsilon_sweep,
    create_summary_figure,
    plot_integrand_vs_t,
    plot_window_capacity,
    plot_family_integrand,
    plot_family_window_capacity,
    plot_family_status_overview,
)
from .report import generate_report, save_results_json, generate_family_report
from .experiments_families import run_family_suite
from .families import get_family_names


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="tse",
        description="Terminal-State Exclusion / Operational Capacity Prototype",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Grid parameters
    parser.add_argument("--tmin", type=float, default=1.0,
                        help="Minimum time (must be >= 1)")
    parser.add_argument("--tmax", type=float, default=1e8,
                        help="Maximum time for integration")
    parser.add_argument("--nsteps", type=int, default=10000,
                        help="Number of log-spaced time steps")
    parser.add_argument("--eps", type=float, default=0.2,
                        help="Weighting exponent offset (epsilon > 0)")

    # Physical parameters
    parser.add_argument("--n0", type=float, default=1.0,
                        help="Number density normalization")
    parser.add_argument("--sv0", type=float, default=1.0,
                        help="Cross-section normalization")
    parser.add_argument("--P0", type=float, default=1.0,
                        help="Free power normalization")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Complexity growth coefficient")

    # Exponents
    parser.add_argument("--gamma_n", type=float, default=1.0,
                        help="Number density exponent")
    parser.add_argument("--gamma_sv", type=float, default=0.0,
                        help="Cross-section exponent")
    parser.add_argument("--gamma_P", type=float, default=1.0,
                        help="Free power exponent")

    # Scenario selection
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(SCENARIOS.keys()),
                        help="Use predefined scenario (overrides gamma_* values)")

    # Experiment modes
    parser.add_argument("--sweep", action="store_true",
                        help="Run full parameter sweep experiments")
    parser.add_argument("--heatmap-resolution", type=int, default=61,
                        help="Resolution for exponent heatmap (NxN grid)")
    parser.add_argument("--heatmap-nsteps", type=int, default=2000,
                        help="Integration steps for heatmap (reduced for speed)")

    # Late-time window diagnostics
    parser.add_argument("--nT", type=int, default=40,
                        help="Number of log-spaced T values for window capacity")
    parser.add_argument("--K_last", type=int, default=5,
                        help="Number of trailing windows to average/median for status")
    parser.add_argument("--K_slope", type=int, default=5,
                        help="Number of trailing windows to use for slope fit")
    parser.add_argument("--C_ZERO_WIN", type=float, default=1e-12,
                        help="Threshold for classifying Nonterminal via window capacity")
    parser.add_argument("--B_MIN_PERSIST", type=float, default=0.1,
                        help="Minimum slope magnitude for persistence (less negative than this is Persistent)")

    # Family suite
    parser.add_argument("--family_suite", action="store_true",
                        help="Run direct-activity family experiments")
    parser.add_argument("--family", type=str, default=None,
                        choices=get_family_names(),
                        help="Optional single family to run")
    parser.add_argument("--family_params", type=str, default=None,
                        help="JSON string of parameter overrides for the selected family")
    parser.add_argument("--nT_family", type=int, default=50,
                        help="Number of T samples for family suite")

    # Output
    parser.add_argument("--outdir", type=str, default="outputs",
                        help="Output directory for plots and results")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    return parser.parse_args()


def run_family(args: argparse.Namespace) -> None:
    """Run the family suite of direct activity experiments."""

    outdir = Path(args.outdir)
    params_override = None
    if args.family_params:
        params_override = json.loads(args.family_params)

    results = run_family_suite(
        tmin=args.tmin,
        tmax=args.tmax,
        nsteps=args.nsteps,
        nT=args.nT_family,
        K_last=args.K_last,
        K_slope=args.K_slope,
        C_ZERO_WIN=args.C_ZERO_WIN,
        B_MIN_PERSIST=args.B_MIN_PERSIST,
        outdir=outdir,
        family=args.family,
        family_params=params_override,
        quiet=args.quiet,
    )

    for diag in results:
        plot_family_integrand(diag, outdir, show=args.show)
        plot_family_window_capacity(diag, outdir, K_slope=args.K_slope, show=args.show)

    plot_family_status_overview(results, outdir, show=args.show)
    generate_family_report(results, outdir)

    if not args.quiet:
        print(f"Family suite results saved to: {outdir}")


def run_single(args: argparse.Namespace) -> None:
    """Run a single scenario computation."""
    if args.scenario:
        params = get_scenario_params(args.scenario, TSEParameters(
            t_min=args.tmin,
            t_max=args.tmax,
            n_steps=args.nsteps,
            eps=args.eps,
            n0=args.n0,
            sv0=args.sv0,
            P0=args.P0,
            alpha=args.alpha,
        ))
        scenario_name = args.scenario
    else:
        params = TSEParameters(
            t_min=args.tmin,
            t_max=args.tmax,
            n_steps=args.nsteps,
            eps=args.eps,
            n0=args.n0,
            sv0=args.sv0,
            P0=args.P0,
            alpha=args.alpha,
            gamma_n=args.gamma_n,
            gamma_sv=args.gamma_sv,
            gamma_P=args.gamma_P,
        )
        scenario_name = "custom"

    if not args.quiet:
        print(f"Computing capacity for scenario: {scenario_name}")
        print(f"  gamma_n={params.gamma_n}, gamma_P={params.gamma_P}, gamma_sv={params.gamma_sv}")
        print(f"  eps={params.eps}, t_max={params.t_max:.0e}, n_steps={params.n_steps}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    diag = compute_scenario_diagnostics(
        scenario_name,
        params,
        nT=args.nT,
        K_last=args.K_last,
        C_ZERO_WIN=args.C_ZERO_WIN,
    )

    print(f"\nResult:")
    print(f"  Operational Capacity C_total = {diag.capacity_total:.6e}")
    print(f"  Window median (last {args.K_last}) = {diag.capacity_window_last_median:.3e}")
    print(f"  Status: {diag.status}")

    plot_integrand_vs_t(diag, outdir, show=args.show)
    plot_window_capacity(diag, outdir, show=args.show)

    convergence_results = {
        scenario_name: ConvergenceResult(
            scenario_name=scenario_name,
            t_max_values=np.array([params.t_max]),
            capacity_values=np.array([diag.capacity_total]),
            converged=diag.integration.converged,
            final_capacity=diag.capacity_total,
        )
    }

    generate_report(convergence_results, None, None, {scenario_name: diag}, params, outdir)
    save_results_json(convergence_results, None, None, {scenario_name: diag}, params, outdir)

    if not args.quiet:
        print(f"Outputs written to {outdir}")


def run_sweep(args: argparse.Namespace) -> None:
    """Run full parameter sweep experiments."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_params = TSEParameters(
        t_min=args.tmin,
        t_max=args.tmax,
        n_steps=args.nsteps,
        eps=args.eps,
        n0=args.n0,
        sv0=args.sv0,
        P0=args.P0,
        alpha=args.alpha,
    )

    start_time = time.time()

    # Late-time diagnostics for each scenario at the base parameters
    scenario_diagnostics = {
        name: compute_scenario_diagnostics(
            name,
            base_params,
            nT=args.nT,
            K_last=args.K_last,
            C_ZERO_WIN=args.C_ZERO_WIN,
        )
        for name in SCENARIOS
    }

    for diag in scenario_diagnostics.values():
        plot_integrand_vs_t(diag, outdir, show=False)
        plot_window_capacity(diag, outdir, show=False)

    # 1. Run t_max convergence for all scenarios
    if not args.quiet:
        print("=" * 60)
        print("Running t_max convergence studies...")
        print("=" * 60)

    convergence_results = run_all_tmax_convergence(
        base_params=base_params,
        t_max_values=[1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
        n_steps=args.nsteps,
    )

    # Print summary
    print("\nScenario Results:")
    print("-" * 50)
    for name, result in convergence_results.items():
        diag_status = scenario_diagnostics[name].status
        print(f"  {name:35s}: C = {result.final_capacity:.3e}  [{diag_status}]")

    # Plot convergence curves
    for name, result in convergence_results.items():
        plot_tmax_convergence(result, outdir=outdir, show=False)
    plot_all_tmax_convergence(convergence_results, outdir=outdir, show=False)

    # 2. Run exponent heatmap
    if not args.quiet:
        print("\n" + "=" * 60)
        print(f"Running exponent heatmap ({args.heatmap_resolution}x{args.heatmap_resolution})...")
        print("=" * 60)

    heatmap_result = run_exponent_heatmap(
        gamma_n_range=(0.0, 3.0),
        gamma_P_range=(0.0, 3.0),
        n_gamma=args.heatmap_resolution,
        base_params=base_params,
        n_steps=args.heatmap_nsteps,
        threshold=1e-12,
    )

    plot_heatmap(heatmap_result, outdir=outdir, show=False)

    # Print boundary info
    if heatmap_result.zero_boundary:
        avg_sum = np.mean([gn + gp for gn, gp in heatmap_result.zero_boundary])
        print(f"\nApproximate 'effectively zero' boundary:")
        print(f"  gamma_n + gamma_P ~ {avg_sum:.2f} (for C < {heatmap_result.threshold:.0e})")

    # 3. Run epsilon sensitivity
    if not args.quiet:
        print("\n" + "=" * 60)
        print("Running epsilon sensitivity analysis...")
        print("=" * 60)

    epsilon_results = [
        run_epsilon_sweep("mild_dilution", base_params=base_params, n_steps=args.nsteps),
        run_epsilon_sweep("strong_dilution", base_params=base_params, n_steps=args.nsteps),
    ]

    plot_epsilon_sweep(epsilon_results, outdir=outdir, show=False)

    # 4. Create summary figure
    create_summary_figure(convergence_results, heatmap_result, outdir=outdir, show=False)

    # 5. Generate report and save results
    if not args.quiet:
        print("\n" + "=" * 60)
        print("Generating report...")
        print("=" * 60)

    generate_report(convergence_results, heatmap_result, epsilon_results, scenario_diagnostics, base_params, outdir)
    save_results_json(convergence_results, heatmap_result, epsilon_results, scenario_diagnostics, base_params, outdir)

    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Results saved to: {outdir.absolute()}")
    print(f"  - report.md")
    print(f"  - results.json")
    print(f"  - *.png plots")

    # Final console summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    terminal = [n for n, d in scenario_diagnostics.items() if d.status == "Terminal"]
    nonterminal = [n for n, d in scenario_diagnostics.items() if d.status == "Nonterminal"]

    print(f"Terminal scenarios (C -> 0):     {', '.join(terminal) if terminal else 'None'}")
    print(f"Nonterminal scenarios (C > 0):   {', '.join(nonterminal) if nonterminal else 'None'}")

    if heatmap_result.zero_boundary:
        print(f"\nExponent boundary (C < 1e-12):")
        print(f"  Approximately gamma_n + gamma_P > {avg_sum:.1f}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        if args.family_suite:
            run_family(args)
        elif args.sweep:
            run_sweep(args)
        else:
            run_single(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
