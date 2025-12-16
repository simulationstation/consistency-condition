"""
TSE CLI Entry Point

Run with: python -m tse [options]
"""

import argparse
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
)
from .plot import (
    plot_tmax_convergence,
    plot_all_tmax_convergence,
    plot_heatmap,
    plot_epsilon_sweep,
    create_summary_figure,
)
from .report import generate_report, save_results_json


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

    # Output
    parser.add_argument("--outdir", type=str, default="outputs",
                        help="Output directory for plots and results")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    return parser.parse_args()


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

    result = compute_capacity(params)

    print(f"\nResult:")
    print(f"  Operational Capacity C = {result.capacity:.6e}")
    print(f"  Converged: {result.converged}")
    print(f"  Relative tail contribution: {result.relative_tail:.2%}")

    if result.capacity < 1e-12:
        print("  Classification: TERMINAL (C -> 0)")
    else:
        print("  Classification: NONTERMINAL (C > 0)")


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
        status = "TERMINAL" if result.final_capacity < 1e-10 else "NONTERMINAL"
        print(f"  {name:35s}: C = {result.final_capacity:.3e}  [{status}]")

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

    generate_report(convergence_results, heatmap_result, epsilon_results, base_params, outdir)
    save_results_json(convergence_results, heatmap_result, epsilon_results, base_params, outdir)

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
    terminal = [n for n, r in convergence_results.items() if r.final_capacity < 1e-10]
    nonterminal = [n for n, r in convergence_results.items() if r.final_capacity >= 1e-10]

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
        if args.sweep:
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
