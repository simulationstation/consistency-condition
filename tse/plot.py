"""
TSE Plotting Utilities

This module provides plotting functions for visualizing TSE model results.
Uses matplotlib only (no seaborn as per requirements).
"""

from pathlib import Path
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .experiments import ConvergenceResult, HeatmapResult, EpsilonSweepResult


def setup_style() -> None:
    """Set up matplotlib style for publication-quality plots."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_tmax_convergence(result: ConvergenceResult,
                          outdir: Optional[Path] = None,
                          show: bool = False) -> Figure:
    """Plot capacity vs t_max for a scenario.

    Args:
        result: ConvergenceResult from experiment
        outdir: Directory to save plot (if provided)
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(result.t_max_values, result.capacity_values, 'o-', linewidth=2,
              markersize=8, color='#2E86AB')

    ax.set_xlabel(r'$t_{\mathrm{max}}$')
    ax.set_ylabel(r'Operational Capacity $C$')
    ax.set_title(f'Capacity Convergence: {result.scenario_name.replace("_", " ").title()}')

    # Add convergence indicator
    status = "Converged" if result.converged else "Not converged"
    ax.text(0.95, 0.95, f"Status: {status}\nFinal C = {result.final_capacity:.3e}",
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()

    if outdir is not None:
        filename = f"capacity_vs_tmax_{result.scenario_name}.png"
        fig.savefig(outdir / filename, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_all_tmax_convergence(results: dict[str, ConvergenceResult],
                              outdir: Optional[Path] = None,
                              show: bool = False) -> Figure:
    """Plot capacity vs t_max for all scenarios on one figure.

    Args:
        results: Dictionary of scenario name -> ConvergenceResult
        outdir: Directory to save plot
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    markers = ['o', 's', '^', 'D']

    for i, (name, result) in enumerate(results.items()):
        label = name.replace('_', ' ').title()
        ax.loglog(result.t_max_values, result.capacity_values,
                  f'{markers[i % len(markers)]}-',
                  linewidth=2, markersize=7,
                  color=colors[i % len(colors)],
                  label=label)

    ax.set_xlabel(r'$t_{\mathrm{max}}$')
    ax.set_ylabel(r'Operational Capacity $C$')
    ax.set_title('Capacity Convergence Across Scenarios')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()

    if outdir is not None:
        fig.savefig(outdir / "capacity_vs_tmax_all_scenarios.png", bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_heatmap(result: HeatmapResult,
                 outdir: Optional[Path] = None,
                 show: bool = False) -> Figure:
    """Plot heatmap of log10(C) over exponent space.

    Args:
        result: HeatmapResult from experiment
        outdir: Directory to save plot
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(9, 7))

    # Create the heatmap
    extent = [result.gamma_n_values.min(), result.gamma_n_values.max(),
              result.gamma_P_values.min(), result.gamma_P_values.max()]

    # Clip extreme values for better visualization
    log10_C = np.clip(result.log10_capacity_grid, -15, 5)

    im = ax.imshow(log10_C, extent=extent, origin='lower', aspect='auto',
                   cmap='viridis', interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=r'$\log_{10}(C)$')

    # Add contour for the threshold
    threshold_log = np.log10(result.threshold)
    contour = ax.contour(result.gamma_n_values, result.gamma_P_values,
                         result.log10_capacity_grid,
                         levels=[threshold_log], colors=['red'], linewidths=2)
    ax.clabel(contour, fmt=f'C = {result.threshold:.0e}', fontsize=9)

    # Mark scenario points
    scenario_points = {
        'Mild Dilution': (1.0, 1.0),
        'Strong Dilution': (2.0, 2.0),
        'de Sitter-like': (0.0, 5.0),
        'Persistent Power': (1.0, 0.0),
    }

    for name, (gn, gp) in scenario_points.items():
        if (result.gamma_n_values.min() <= gn <= result.gamma_n_values.max() and
            result.gamma_P_values.min() <= gp <= result.gamma_P_values.max()):
            ax.plot(gn, gp, 'w*', markersize=12, markeredgecolor='black')
            ax.annotate(name, (gn, gp), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='white',
                       path_effects=[path_effects.withStroke(
                           linewidth=2, foreground='black')])

    ax.set_xlabel(r'$\gamma_n$ (density exponent)')
    ax.set_ylabel(r'$\gamma_P$ (power exponent)')
    ax.set_title(r'Operational Capacity in Exponent Space ($\gamma_{sv} = 0$)')

    plt.tight_layout()

    if outdir is not None:
        fig.savefig(outdir / "heatmap_log10C_gamma_n_gamma_P.png", bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_epsilon_sweep(results: list[EpsilonSweepResult],
                       outdir: Optional[Path] = None,
                       show: bool = False) -> Figure:
    """Plot capacity vs epsilon for multiple scenarios.

    Args:
        results: List of EpsilonSweepResult objects
        outdir: Directory to save plots
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for i, result in enumerate(results):
        label = result.scenario_name.replace('_', ' ').title()
        ax.loglog(result.eps_values, result.capacity_values, 'o-',
                  linewidth=2, markersize=6,
                  color=colors[i % len(colors)],
                  label=label)

    ax.set_xlabel(r'$\varepsilon$ (weighting exponent offset)')
    ax.set_ylabel(r'Operational Capacity $C$')
    ax.set_title(r'Capacity Sensitivity to Weighting Exponent $\varepsilon$')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()

    if outdir is not None:
        # Save combined plot
        fig.savefig(outdir / "capacity_vs_eps_combined.png", bbox_inches='tight')
        # Also save individual scenario plots
        for result in results:
            fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
            ax_ind.loglog(result.eps_values, result.capacity_values, 'o-',
                         linewidth=2, markersize=6, color='#2E86AB')
            ax_ind.set_xlabel(r'$\varepsilon$')
            ax_ind.set_ylabel(r'$C$')
            ax_ind.set_title(f'Capacity vs {chr(949)}: {result.scenario_name.replace("_", " ").title()}')
            ax_ind.grid(True, which='both', alpha=0.3)
            plt.tight_layout()
            fig_ind.savefig(outdir / f"capacity_vs_eps_{result.scenario_name}.png",
                           bbox_inches='tight')
            plt.close(fig_ind)

    if show:
        plt.show()

    return fig


def plot_integrand_profile(t: NDArray[np.float64],
                           integrand_vals: NDArray[np.float64],
                           scenario_name: str,
                           outdir: Optional[Path] = None,
                           show: bool = False) -> Figure:
    """Plot the integrand as a function of time.

    Args:
        t: Time array
        integrand_vals: Integrand values
        scenario_name: Name of the scenario
        outdir: Directory to save plot
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(t, integrand_vals, '-', linewidth=1.5, color='#2E86AB')

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'Integrand $w(t) \cdot I(t) \cdot \frac{dK}{dt}$')
    ax.set_title(f'Integrand Profile: {scenario_name.replace("_", " ").title()}')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()

    if outdir is not None:
        fig.savefig(outdir / f"integrand_profile_{scenario_name}.png", bbox_inches='tight')

    if show:
        plt.show()

    return fig


def create_summary_figure(convergence_results: dict[str, ConvergenceResult],
                          heatmap_result: HeatmapResult,
                          outdir: Optional[Path] = None,
                          show: bool = False) -> Figure:
    """Create a multi-panel summary figure.

    Args:
        convergence_results: t_max convergence results
        heatmap_result: Exponent heatmap result
        outdir: Directory to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    setup_style()

    fig = plt.figure(figsize=(14, 10))

    # Panel 1: Convergence curves
    ax1 = fig.add_subplot(2, 2, 1)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    markers = ['o', 's', '^', 'D']

    for i, (name, result) in enumerate(convergence_results.items()):
        label = name.replace('_', ' ').title()
        ax1.loglog(result.t_max_values, result.capacity_values,
                   f'{markers[i % len(markers)]}-',
                   linewidth=2, markersize=6,
                   color=colors[i % len(colors)],
                   label=label)

    ax1.set_xlabel(r'$t_{\mathrm{max}}$')
    ax1.set_ylabel(r'$C$')
    ax1.set_title('(a) Capacity Convergence')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, which='both', alpha=0.3)

    # Panel 2: Heatmap
    ax2 = fig.add_subplot(2, 2, 2)
    extent = [heatmap_result.gamma_n_values.min(), heatmap_result.gamma_n_values.max(),
              heatmap_result.gamma_P_values.min(), heatmap_result.gamma_P_values.max()]
    log10_C = np.clip(heatmap_result.log10_capacity_grid, -15, 5)
    im = ax2.imshow(log10_C, extent=extent, origin='lower', aspect='auto',
                    cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax2, label=r'$\log_{10}(C)$')
    ax2.set_xlabel(r'$\gamma_n$')
    ax2.set_ylabel(r'$\gamma_P$')
    ax2.set_title(r'(b) Capacity Heatmap')

    # Panel 3: Final capacity comparison (bar chart)
    ax3 = fig.add_subplot(2, 2, 3)
    names = list(convergence_results.keys())
    capacities = [r.final_capacity for r in convergence_results.values()]
    x = np.arange(len(names))
    bars = ax3.bar(x, capacities, color=colors[:len(names)])
    ax3.set_xticks(x)
    ax3.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    ax3.set_ylabel(r'Final $C$ at $t_{max}=10^8$')
    ax3.set_title('(c) Scenario Comparison')
    ax3.set_yscale('log')

    # Panel 4: Text summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary_text = "TSE Model Summary\n" + "=" * 30 + "\n\n"
    for name, result in convergence_results.items():
        status = "Terminal (C->0)" if result.final_capacity < 1e-10 else "Nonterminal (C>0)"
        summary_text += f"{name}:\n"
        summary_text += f"  C = {result.final_capacity:.3e}\n"
        summary_text += f"  Status: {status}\n\n"

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             fontfamily='monospace', fontsize=9, verticalalignment='top')
    ax4.set_title('(d) Summary')

    plt.tight_layout()

    if outdir is not None:
        fig.savefig(outdir / "summary_figure.png", bbox_inches='tight')

    if show:
        plt.show()

    return fig
