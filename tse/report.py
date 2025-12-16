"""
TSE Report Generation

This module generates markdown reports summarizing TSE model results.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import numpy as np

from .model import TSEParameters, SCENARIOS
from .experiments import (
    ConvergenceResult,
    HeatmapResult,
    EpsilonSweepResult,
    ScenarioDiagnostics,
)
from .experiments_families import FamilyDiagnostics


def generate_report(convergence_results: dict[str, ConvergenceResult],
                    heatmap_result: Optional[HeatmapResult],
                    epsilon_results: Optional[list[EpsilonSweepResult]],
                    scenario_diagnostics: Optional[dict[str, ScenarioDiagnostics]],
                    params: TSEParameters,
                    outdir: Path) -> str:
    """Generate the full markdown report.

    Args:
        convergence_results: t_max convergence results by scenario
        heatmap_result: Exponent sweep heatmap result
        epsilon_results: Epsilon sensitivity results
        params: Base parameters used
        outdir: Output directory for report

    Returns:
        Report content as string
    """
    report = []

    # Header
    report.append("# Terminal-State Exclusion (TSE) / Operational Capacity Report")
    report.append("")
    report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    report.append("")

    # Model equations
    report.append("## Model Equations")
    report.append("")
    report.append("The Operational Capacity functional is defined as:")
    report.append("")
    report.append("$$C = \\int_{t_{min}}^{t_{max}} w(t) \\cdot I(t) \\cdot \\max(0, \\frac{dK_{acc}}{dt}) \\, dt$$")
    report.append("")
    report.append("where:")
    report.append("")
    report.append("- **Weighting function**: $w(t) = \\frac{1}{(1+t)^{1+\\varepsilon}}$")
    report.append("- **Interaction density**: $I(t) = n(t) \\cdot \\langle\\sigma v\\rangle(t)$")
    report.append("  - $n(t) = \\frac{n_0}{(1+t)^{\\gamma_n}}$")
    report.append("  - $\\langle\\sigma v\\rangle(t) = \\frac{sv_0}{(1+t)^{\\gamma_{sv}}}$")
    report.append("- **Complexity growth rate**: $\\frac{dK_{acc}}{dt} = \\alpha \\cdot P_{free}(t)$")
    report.append("  - $P_{free}(t) = \\frac{P_0}{(1+t)^{\\gamma_P}}$")
    report.append("")
    report.append("The integrand scales as $(1+t)^{-\\beta}$ where $\\beta = 1 + \\varepsilon + \\gamma_n + \\gamma_{sv} + \\gamma_P$.")
    report.append("")
    report.append("Total capacity is always positive for positive integrand prefactors; terminality is therefore diagnosed using late-time window capacity rather than total capacity.")
    report.append("")

    # Parameters
    report.append("## Parameters Used")
    report.append("")
    report.append("| Parameter | Value |")
    report.append("|-----------|-------|")
    report.append(f"| $t_{{min}}$ | {params.t_min} |")
    report.append(f"| $t_{{max}}$ | {params.t_max:.0e} |")
    report.append(f"| $n_{{steps}}$ | {params.n_steps} |")
    report.append(f"| $\\varepsilon$ | {params.eps} |")
    report.append(f"| $n_0$ | {params.n0} |")
    report.append(f"| $sv_0$ | {params.sv0} |")
    report.append(f"| $P_0$ | {params.P0} |")
    report.append(f"| $\\alpha$ | {params.alpha} |")
    report.append("")

    # Scenario definitions
    report.append("## Scenario Definitions")
    report.append("")
    report.append("| Scenario | $\\gamma_n$ | $\\gamma_P$ | $\\gamma_{sv}$ | Physical Interpretation |")
    report.append("|----------|-----------|-----------|-------------|------------------------|")
    report.append("| mild_dilution | 1.0 | 1.0 | 0.0 | Moderate expansion with proportional power dilution |")
    report.append("| strong_dilution | 2.0 | 2.0 | 0.0 | Rapid expansion (de Sitter-like matter) |")
    report.append("| desitter_like_no_free_power | 0.0 | 5.0 | 0.0 | Constant density but severe energy depletion |")
    report.append("| persistent_power_nonterminal | 1.0 | 0.0 | 0.0 | Diluting density but inexhaustible power |")
    report.append("")

    # Main results table
    report.append("## Results Summary")
    report.append("")
    report.append("### Scenario Capacity Values")
    report.append("")
    report.append("| Scenario | Total $C$ | $C_{win}$ (last median) | Status |")
    report.append("|----------|-----------|-------------------------|--------|")

    for name, result in convergence_results.items():
        diag = scenario_diagnostics.get(name) if scenario_diagnostics else None
        total_C = diag.capacity_total if diag else result.final_capacity
        window_med = diag.capacity_window_last_median if diag else float('nan')
        status = diag.status if diag else ("Nonterminal" if result.final_capacity >= 1e-12 else "Terminal")
        report.append(f"| {name} | {total_C:.3e} | {window_med:.3e} | {status} |")
    report.append("")

    # Convergence analysis
    report.append("### Convergence with $t_{max}$")
    report.append("")
    report.append("The following table shows capacity values at different $t_{max}$ values:")
    report.append("")

    # Build convergence table
    t_max_vals = list(convergence_results.values())[0].t_max_values
    header = "| Scenario | " + " | ".join([f"$10^{{{int(np.log10(t))}}}$" for t in t_max_vals]) + " |"
    separator = "|" + "|".join(["----------"] * (len(t_max_vals) + 1)) + "|"
    report.append(header)
    report.append(separator)

    for name, result in convergence_results.items():
        row = f"| {name} | " + " | ".join([f"{c:.2e}" for c in result.capacity_values]) + " |"
        report.append(row)
    report.append("")

    # Heatmap analysis
    if heatmap_result is not None:
        report.append("### Exponent Space Analysis")
        report.append("")
        report.append(f"Swept $\\gamma_n \\in [{heatmap_result.gamma_n_values.min():.1f}, {heatmap_result.gamma_n_values.max():.1f}]$ and ")
        report.append(f"$\\gamma_P \\in [{heatmap_result.gamma_P_values.min():.1f}, {heatmap_result.gamma_P_values.max():.1f}]$ with {len(heatmap_result.gamma_n_values)} points per axis.")
        report.append("")
        report.append(f"**Threshold for 'effectively zero'**: $C < {heatmap_result.threshold:.0e}$")
        report.append("")

        # Analyze the boundary
        if heatmap_result.zero_boundary:
            avg_sum = np.mean([gn + gp for gn, gp in heatmap_result.zero_boundary])
            report.append(f"The boundary where $C$ becomes effectively zero lies approximately where:")
            report.append(f"$\\gamma_n + \\gamma_P \\approx {avg_sum:.1f}$ (for $\\gamma_{{sv}} = 0$)")
        else:
            report.append("No clear boundary found in the scanned range.")
        report.append("")

    # Epsilon sensitivity
    if epsilon_results:
        report.append("### Sensitivity to Weighting Exponent $\\varepsilon$")
        report.append("")
        report.append("Larger $\\varepsilon$ increases the effective suppression of late-time contributions,")
        report.append("generally reducing $C$ but also improving convergence.")
        report.append("")

        for result in epsilon_results:
            eps_min, eps_max = result.eps_values.min(), result.eps_values.max()
            C_min, C_max = result.capacity_values.min(), result.capacity_values.max()
            report.append(f"- **{result.scenario_name}**: $C$ ranges from {C_min:.2e} to {C_max:.2e}")
            report.append(f"  as $\\varepsilon$ varies from {eps_min:.2f} to {eps_max:.2f}")
        report.append("")

    # Numerical stability notes
    report.append("## Numerical Stability Notes")
    report.append("")
    report.append("1. **Integration method**: Trapezoidal rule on log-spaced grid")
    report.append("2. **Grid construction**: `numpy.logspace` with proper $dt$ in linear space")
    report.append("3. **Underflow protection**: Log-space computation with `np.log1p` for numerical stability")
    report.append("4. **Precision**: All computations in `float64` (double precision)")
    report.append("")
    report.append(f"Default integration uses {params.n_steps} grid points spanning $t \\in [1, 10^8]$.")
    report.append("Heatmap sweeps use reduced resolution (2000 points) for computational efficiency.")
    report.append("")

    # Figure list
    report.append("## Figures")
    report.append("")
    report.append("| Filename | Description |")
    report.append("|----------|-------------|")
    for name in convergence_results:
        report.append(f"| `capacity_vs_tmax_{name}.png` | Convergence of $C$ with $t_{{max}}$ for {name} |")
        if scenario_diagnostics and name in scenario_diagnostics:
            report.append(f"| `integrand_vs_t_{name}.png` | Integrand $f(t)$ vs $t$ for {name} |")
            report.append(f"| `window_capacity_vs_T_{name}.png` | Window capacity $C_{{win}}(T)$ for {name} |")
    report.append("| `capacity_vs_tmax_all_scenarios.png` | All scenarios on one plot |")
    if heatmap_result is not None:
        report.append("| `heatmap_log10C_gamma_n_gamma_P.png` | Heatmap of $\\log_{10}(C)$ in exponent space |")
    if epsilon_results:
        report.append("| `capacity_vs_eps_combined.png` | $C$ vs $\\varepsilon$ for all tested scenarios |")
        for result in epsilon_results:
            report.append(f"| `capacity_vs_eps_{result.scenario_name}.png` | $C$ vs $\\varepsilon$ for {result.scenario_name} |")
    report.append("| `summary_figure.png` | Multi-panel summary figure |")
    report.append("")

    # Conclusions
    report.append("## Key Findings")
    report.append("")

    # Identify terminal vs nonterminal using windowed capacity status
    if scenario_diagnostics:
        terminal = [name for name, diag in scenario_diagnostics.items() if diag.status == "Terminal"]
        nonterminal = [name for name, diag in scenario_diagnostics.items() if diag.status == "Nonterminal"]
    else:
        terminal = [name for name, r in convergence_results.items() if r.final_capacity < 1e-10]
        nonterminal = [name for name, r in convergence_results.items() if r.final_capacity >= 1e-10]

    if terminal:
        report.append(f"**Terminal scenarios** ($C \\to 0$): {', '.join(terminal)}")
        report.append("")
    if nonterminal:
        report.append(f"**Nonterminal scenarios** ($C > 0$): {', '.join(nonterminal)}")
        report.append("")

    report.append("The persistent_power_nonterminal scenario demonstrates that even with diluting")
    report.append("density, inexhaustible free power ($\\gamma_P = 0$) maintains $C > 0$.")
    report.append("")
    report.append("Conversely, rapid power depletion (high $\\gamma_P$) drives $C \\to 0$,")
    report.append("classifying the scenario as terminal regardless of density evolution.")
    report.append("")

    # Write to file
    report_content = "\n".join(report)
    report_path = outdir / "report.md"
    report_path.write_text(report_content)

    return report_content


def save_results_json(convergence_results: dict[str, ConvergenceResult],
                      heatmap_result: Optional[HeatmapResult],
                      epsilon_results: Optional[list[EpsilonSweepResult]],
                      scenario_diagnostics: Optional[dict[str, ScenarioDiagnostics]],
                      params: TSEParameters,
                      outdir: Path) -> dict:
    """Save all numerical results to JSON.

    Args:
        convergence_results: t_max convergence results
        heatmap_result: Exponent heatmap result
        epsilon_results: Epsilon sensitivity results
        params: Base parameters
        outdir: Output directory

    Returns:
        Results dictionary
    """
    results = {
        "parameters": {
            "t_min": params.t_min,
            "t_max": params.t_max,
            "n_steps": params.n_steps,
            "eps": params.eps,
            "n0": params.n0,
            "sv0": params.sv0,
            "P0": params.P0,
            "alpha": params.alpha,
        },
        "scenarios": {},
        "timestamp": datetime.now().isoformat(),
    }

    for name, result in convergence_results.items():
        if name in SCENARIOS:
            gamma_n = SCENARIOS[name]["gamma_n"]
            gamma_P = SCENARIOS[name]["gamma_P"]
            gamma_sv = SCENARIOS[name]["gamma_sv"]
        elif scenario_diagnostics and name in scenario_diagnostics:
            gamma_n = scenario_diagnostics[name].params.gamma_n
            gamma_P = scenario_diagnostics[name].params.gamma_P
            gamma_sv = scenario_diagnostics[name].params.gamma_sv
        else:
            gamma_n = gamma_P = gamma_sv = None

        results["scenarios"][name] = {
            "gamma_n": gamma_n,
            "gamma_P": gamma_P,
            "gamma_sv": gamma_sv,
            "t_max_values": result.t_max_values.tolist(),
            "capacity_values": result.capacity_values.tolist(),
            "final_capacity": float(result.final_capacity),
            "converged": bool(result.converged),
        }

        if scenario_diagnostics and name in scenario_diagnostics:
            diag = scenario_diagnostics[name]
            results["scenarios"][name].update({
                "capacity_total": diag.capacity_total,
                "window_T_values": diag.window_T_values.tolist(),
                "capacity_window_values": diag.capacity_window_values.tolist(),
                "capacity_window_last_median": diag.capacity_window_last_median,
                "status": diag.status,
            })
            if diag.capacity_tail_values is not None:
                results["scenarios"][name]["capacity_tail_values"] = diag.capacity_tail_values.tolist()

    if heatmap_result is not None:
        results["heatmap"] = {
            "gamma_n_range": [float(heatmap_result.gamma_n_values.min()),
                             float(heatmap_result.gamma_n_values.max())],
            "gamma_P_range": [float(heatmap_result.gamma_P_values.min()),
                             float(heatmap_result.gamma_P_values.max())],
            "n_points": len(heatmap_result.gamma_n_values),
            "threshold": heatmap_result.threshold,
            "n_boundary_points": len(heatmap_result.zero_boundary),
            "log10_C_min": float(np.min(heatmap_result.log10_capacity_grid[
                np.isfinite(heatmap_result.log10_capacity_grid)])),
            "log10_C_max": float(np.max(heatmap_result.log10_capacity_grid)),
        }

    if epsilon_results:
        results["epsilon_sensitivity"] = {}
        for result in epsilon_results:
            results["epsilon_sensitivity"][result.scenario_name] = {
                "eps_values": result.eps_values.tolist(),
                "capacity_values": result.capacity_values.tolist(),
            }

    # Write to file
    json_path = outdir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def generate_family_report(results: list[FamilyDiagnostics], outdir: Path) -> str:
    """Generate markdown report for the family suite."""

    report: list[str] = []
    report.append("# Late-Time Activity Families")
    report.append("")
    report.append("Lemma: If $f(t)\\to 0$ then $C_{win}(T)\\to 0$; persistence requires violating that assumption (e.g., constant plateaus or nondecaying pulses).")
    report.append("")
    report.append("| Family | Params | $C_{win}$ last median | Slope $b$ | Status | $f(t_{max})/f(t_{min})$ | Plots |")
    report.append("|--------|--------|-----------------------|-----------|--------|-------------------------|-------|")

    for diag in results:
        param_str = ", ".join([f"{k}={v}" for k, v in diag.params.items()])
        report.append(
            "| "
            f"{diag.family} | "
            f"{param_str} | "
            f"{diag.C_win_last_median:.3e} | "
            f"{diag.slope.b:.3f} | "
            f"{diag.status} | "
            f"{diag.f_tail_ratio:.3e} | "
            f"`family_integrand_vs_t_{diag.family}.png`, `family_window_capacity_vs_T_{diag.family}.png` |")

    report.append("")
    report.append("Overall status plot: `family_suite_status_overview.png`.")

    content = "\n".join(report)
    out_path = outdir / "report_families.md"
    out_path.write_text(content)
    return content
