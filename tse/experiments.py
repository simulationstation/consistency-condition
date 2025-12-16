"""
TSE Experiments

This module provides parameter sweeps and scenario-based experiments
for studying the Terminal-State Exclusion / Operational Capacity model.

Experiments include:
- Convergence studies with t_max
- Sensitivity analysis for exponents (γ_n, γ_P)
- Sensitivity analysis for weighting exponent ε
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from .model import TSEParameters, SCENARIOS, get_scenario_params, integrand
from .integrate import (
    compute_capacity,
    compute_capacity_quick,
    convergence_study,
    make_log_grid,
    trapezoid_log_grid,
    IntegrationResult,
)


@dataclass
class ConvergenceResult:
    """Result of t_max convergence experiment."""
    scenario_name: str
    t_max_values: NDArray[np.float64]
    capacity_values: NDArray[np.float64]
    converged: bool
    final_capacity: float


@dataclass
class HeatmapResult:
    """Result of 2D exponent sweep."""
    gamma_n_values: NDArray[np.float64]
    gamma_P_values: NDArray[np.float64]
    capacity_grid: NDArray[np.float64]  # Shape (n_gamma_P, n_gamma_n)
    log10_capacity_grid: NDArray[np.float64]
    threshold: float
    zero_boundary: list[tuple[float, float]]  # Approximate boundary points


@dataclass
class EpsilonSweepResult:
    """Result of epsilon sensitivity experiment."""
    scenario_name: str
    eps_values: NDArray[np.float64]
    capacity_values: NDArray[np.float64]


def run_tmax_convergence(scenario_name: str,
                         base_params: Optional[TSEParameters] = None,
                         t_max_values: Optional[list[float]] = None,
                         n_steps: int = 10000) -> ConvergenceResult:
    """Run t_max convergence study for a scenario.

    Args:
        scenario_name: Name of the scenario to test
        base_params: Optional base parameters
        t_max_values: List of t_max values to test
        n_steps: Integration steps per run

    Returns:
        ConvergenceResult with data
    """
    if t_max_values is None:
        t_max_values = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

    params = get_scenario_params(scenario_name, base_params)
    t_max_arr, capacity_arr = convergence_study(params, t_max_values, n_steps)

    # Check if converged: last two values should be within 5%
    if len(capacity_arr) >= 2 and capacity_arr[-2] > 0:
        rel_change = abs(capacity_arr[-1] - capacity_arr[-2]) / capacity_arr[-2]
        converged = rel_change < 0.05
    else:
        converged = False

    return ConvergenceResult(
        scenario_name=scenario_name,
        t_max_values=t_max_arr,
        capacity_values=capacity_arr,
        converged=converged,
        final_capacity=capacity_arr[-1],
    )


def run_all_tmax_convergence(base_params: Optional[TSEParameters] = None,
                             t_max_values: Optional[list[float]] = None,
                             n_steps: int = 10000) -> dict[str, ConvergenceResult]:
    """Run t_max convergence for all predefined scenarios.

    Args:
        base_params: Optional base parameters
        t_max_values: List of t_max values
        n_steps: Integration steps

    Returns:
        Dictionary mapping scenario name to result
    """
    results = {}
    for scenario_name in SCENARIOS:
        results[scenario_name] = run_tmax_convergence(
            scenario_name, base_params, t_max_values, n_steps
        )
    return results


def run_exponent_heatmap(gamma_n_range: tuple[float, float] = (0.0, 3.0),
                         gamma_P_range: tuple[float, float] = (0.0, 3.0),
                         n_gamma: int = 61,
                         base_params: Optional[TSEParameters] = None,
                         n_steps: int = 2000,
                         threshold: float = 1e-12) -> HeatmapResult:
    """Run 2D sweep over γ_n and γ_P exponents.

    This is the core sensitivity analysis showing the terminal/nonterminal
    boundary in exponent space.

    Args:
        gamma_n_range: (min, max) for γ_n
        gamma_P_range: (min, max) for γ_P
        n_gamma: Number of points per axis
        base_params: Base parameters (eps, n0, etc.)
        n_steps: Integration steps (reduced for speed)
        threshold: Capacity below this is "effectively zero"

    Returns:
        HeatmapResult with grid data and boundary
    """
    if base_params is None:
        base_params = TSEParameters()

    gamma_n_vals = np.linspace(gamma_n_range[0], gamma_n_range[1], n_gamma)
    gamma_P_vals = np.linspace(gamma_P_range[0], gamma_P_range[1], n_gamma)

    # Pre-compute shared time grid
    t_grid = make_log_grid(base_params.t_min, base_params.t_max, n_steps)

    # Initialize capacity grid
    capacity_grid = np.zeros((n_gamma, n_gamma))

    # Vectorized approach: precompute time-dependent factors
    # w(t) = (1+t)^(-(1+eps))
    # The integrand is: (1+t)^(-(1+eps+gamma_n+gamma_sv+gamma_P)) * norm
    # We precompute log(1+t) and then compute for each parameter combination

    log_1pt = np.log1p(t_grid)
    base_norm = base_params.n0 * base_params.sv0 * base_params.alpha * base_params.P0
    base_exp = 1 + base_params.eps + base_params.gamma_sv

    for i, gamma_P in enumerate(gamma_P_vals):
        for j, gamma_n in enumerate(gamma_n_vals):
            total_exp = base_exp + gamma_n + gamma_P
            log_integrand = np.log(base_norm) - total_exp * log_1pt

            # Convert from log and integrate
            with np.errstate(under='ignore'):
                integrand_vals = np.exp(log_integrand)
            integrand_vals = np.where(np.isfinite(integrand_vals), integrand_vals, 0.0)

            capacity_grid[i, j] = trapezoid_log_grid(integrand_vals, t_grid)

    # Compute log10 capacity, handling zeros
    with np.errstate(divide='ignore'):
        log10_capacity = np.log10(capacity_grid)
    log10_capacity = np.where(np.isfinite(log10_capacity), log10_capacity, -20.0)

    # Find approximate boundary where C crosses threshold
    boundary_points = []
    for i, gamma_P in enumerate(gamma_P_vals):
        for j in range(n_gamma - 1):
            c1, c2 = capacity_grid[i, j], capacity_grid[i, j + 1]
            if (c1 >= threshold and c2 < threshold) or (c1 < threshold and c2 >= threshold):
                # Linear interpolation to find crossing
                gamma_n_cross = gamma_n_vals[j] + (gamma_n_vals[j + 1] - gamma_n_vals[j]) * \
                               (threshold - c1) / (c2 - c1) if c2 != c1 else gamma_n_vals[j]
                boundary_points.append((gamma_n_cross, gamma_P))

    return HeatmapResult(
        gamma_n_values=gamma_n_vals,
        gamma_P_values=gamma_P_vals,
        capacity_grid=capacity_grid,
        log10_capacity_grid=log10_capacity,
        threshold=threshold,
        zero_boundary=boundary_points,
    )


def run_epsilon_sweep(scenario_name: str,
                      eps_range: tuple[float, float] = (0.01, 1.0),
                      n_eps: int = 30,
                      base_params: Optional[TSEParameters] = None,
                      n_steps: int = 10000) -> EpsilonSweepResult:
    """Run sweep over epsilon values for a scenario.

    Args:
        scenario_name: Scenario to test
        eps_range: (min, max) for ε in log space
        n_eps: Number of epsilon values
        base_params: Base parameters
        n_steps: Integration steps

    Returns:
        EpsilonSweepResult with data
    """
    eps_vals = np.logspace(np.log10(eps_range[0]), np.log10(eps_range[1]), n_eps)
    capacity_vals = np.zeros(n_eps)

    scenario = get_scenario_params(scenario_name, base_params)

    for i, eps in enumerate(eps_vals):
        params = TSEParameters(
            t_min=scenario.t_min,
            t_max=scenario.t_max,
            n_steps=n_steps,
            eps=eps,
            n0=scenario.n0,
            sv0=scenario.sv0,
            P0=scenario.P0,
            alpha=scenario.alpha,
            gamma_n=scenario.gamma_n,
            gamma_sv=scenario.gamma_sv,
            gamma_P=scenario.gamma_P,
        )
        result = compute_capacity(params)
        capacity_vals[i] = result.capacity

    return EpsilonSweepResult(
        scenario_name=scenario_name,
        eps_values=eps_vals,
        capacity_values=capacity_vals,
    )


def run_single_scenario(scenario_name: str,
                        base_params: Optional[TSEParameters] = None) -> IntegrationResult:
    """Run a single scenario and return full result.

    Args:
        scenario_name: Scenario to run
        base_params: Optional base parameters

    Returns:
        IntegrationResult with full diagnostics
    """
    params = get_scenario_params(scenario_name, base_params)
    return compute_capacity(params)


def find_boundary_exponent_sum(params: TSEParameters) -> float:
    """Find the critical sum of exponents for convergence.

    For the toy model, the integrand scales as (1+t)^(-β) where:
    β = 1 + ε + γ_n + γ_sv + γ_P

    The integral converges (C finite) when β > 1, i.e.:
    γ_n + γ_sv + γ_P > -ε

    For ε > 0, this is almost always satisfied. The boundary
    in (γ_n, γ_P) space (with γ_sv=0) is approximately:
    γ_n + γ_P ≈ -ε (but since exponents are positive, C > 0 generally)

    The "effectively zero" boundary depends on the threshold and t_max.

    Returns:
        The theoretical convergence threshold for exponent sum
    """
    return -params.eps


def summarize_results(convergence_results: dict[str, ConvergenceResult],
                      heatmap_result: Optional[HeatmapResult] = None) -> dict:
    """Create a summary of all experimental results.

    Args:
        convergence_results: t_max convergence results by scenario
        heatmap_result: Optional heatmap result

    Returns:
        Summary dictionary
    """
    summary = {
        "scenarios": {},
        "heatmap": None,
    }

    for name, result in convergence_results.items():
        summary["scenarios"][name] = {
            "final_capacity": result.final_capacity,
            "converged": result.converged,
            "capacity_at_tmax_1e6": float(result.capacity_values[
                np.searchsorted(result.t_max_values, 1e6)
            ]) if 1e6 in result.t_max_values else None,
        }

    if heatmap_result is not None:
        summary["heatmap"] = {
            "gamma_n_range": [float(heatmap_result.gamma_n_values.min()),
                             float(heatmap_result.gamma_n_values.max())],
            "gamma_P_range": [float(heatmap_result.gamma_P_values.min()),
                             float(heatmap_result.gamma_P_values.max())],
            "threshold": heatmap_result.threshold,
            "n_boundary_points": len(heatmap_result.zero_boundary),
        }

    return summary
