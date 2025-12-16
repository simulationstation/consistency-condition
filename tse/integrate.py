"""
TSE Numerical Integration

This module provides numerical integration routines for computing
the Operational Capacity functional:

C = ∫ w(t) * I(t) * max(0, dK_acc/dt) dt

Key features:
- Log-spaced time grids for stability across decades
- Trapezoidal rule with proper dt handling
- Convergence checking
- Support for both direct and log-space computation
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from .model import TSEParameters, integrand, integrand_log, convergence_criterion


@dataclass
class IntegrationResult:
    """Result of numerical integration.

    Attributes:
        capacity: The computed operational capacity C
        t_grid: Time points used
        integrand_values: Integrand values at grid points
        converged: Whether the integral appears converged
        relative_tail: Fraction of integral from last decade
        params: Parameters used
    """
    capacity: float
    t_grid: NDArray[np.float64]
    integrand_values: NDArray[np.float64]
    converged: bool
    relative_tail: float
    params: TSEParameters


def make_log_grid(t_min: float, t_max: float, n_steps: int) -> NDArray[np.float64]:
    """Create a logarithmically-spaced time grid.

    Args:
        t_min: Minimum time (>= 1)
        t_max: Maximum time
        n_steps: Number of grid points

    Returns:
        Log-spaced array of time values
    """
    return np.logspace(np.log10(t_min), np.log10(t_max), n_steps)


def trapezoid_log_grid(y: NDArray[np.float64], t: NDArray[np.float64]) -> float:
    """Trapezoidal integration on a log-spaced grid.

    For a log-spaced grid, the intervals dt vary. We compute:
    ∫ y dt ≈ Σ (y[i] + y[i+1])/2 * (t[i+1] - t[i])

    Args:
        y: Function values at grid points
        t: Grid points (log-spaced)

    Returns:
        Approximate integral
    """
    # Compute dt in linear space (differences between adjacent t values)
    dt = np.diff(t)
    # Trapezoidal rule: average adjacent y values, multiply by dt
    y_avg = 0.5 * (y[:-1] + y[1:])
    return np.sum(y_avg * dt)


def compute_capacity_total(ts: NDArray[np.float64],
                           f: NDArray[np.float64]) -> float:
    """Compute total capacity using trapezoidal integration on linear ``t``.

    Args:
        ts: Time samples (monotonically increasing)
        f: Integrand samples corresponding to ``ts``

    Returns:
        Total capacity ``C``.
    """
    return float(np.trapezoid(f, ts))


def compute_capacity_tail(ts: NDArray[np.float64],
                          f: NDArray[np.float64],
                          T: float) -> float:
    """Compute tail capacity from ``T`` to ``t_max`` using trapezoid rule.

    Args:
        ts: Time samples (monotonic, linear domain)
        f: Integrand samples
        T: Lower limit for the tail integral

    Returns:
        Tail capacity ``C_tail(T)``.
    """
    mask = ts >= T
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(f[mask], ts[mask]))


def compute_capacity_window(ts: NDArray[np.float64],
                            f: NDArray[np.float64],
                            T: float) -> float:
    """Compute windowed capacity over ``[T, 2T]``.

    Returns ``np.nan`` if the window exceeds the sampled range or
    has insufficient samples.

    Args:
        ts: Time samples (monotonic)
        f: Integrand samples
        T: Window start

    Returns:
        Window capacity ``C_win(T)`` or ``np.nan`` if invalid.
    """
    upper = 2.0 * T
    if upper > ts.max():
        return float('nan')

    mask = (ts >= T) & (ts <= upper)
    if np.count_nonzero(mask) < 2:
        return float('nan')

    return float(np.trapezoid(f[mask], ts[mask]))


def compute_capacity_windows(ts: NDArray[np.float64],
                             f: NDArray[np.float64],
                             T_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute windowed capacities for a sequence of ``T`` values."""
    capacities = np.empty_like(T_values, dtype=float)
    for i, T in enumerate(T_values):
        capacities[i] = compute_capacity_window(ts, f, float(T))
    return capacities


def select_T_values(tmin: float, tmax: float, nT: int, factor: float = 2.0) -> NDArray[np.float64]:
    """Select log-spaced ``T`` values such that ``factor*T <= tmax``.

    Args:
        tmin: Minimum time value
        tmax: Maximum time value
        nT: Number of ``T`` samples
        factor: Window factor (default 2.0)

    Returns:
        Array of log-spaced ``T`` values within the valid range.
    """
    upper = tmax / factor
    if upper <= tmin:
        return np.array([], dtype=float)
    return np.logspace(np.log10(tmin), np.log10(upper), nT)


def compute_capacity(params: TSEParameters,
                     use_log_space: bool = False) -> IntegrationResult:
    """Compute the Operational Capacity for given parameters.

    Args:
        params: TSE model parameters
        use_log_space: If True, compute integrand in log space first
                      (more numerically stable for extreme parameter values)

    Returns:
        IntegrationResult with computed capacity and diagnostics
    """
    # Create log-spaced grid
    t_grid = make_log_grid(params.t_min, params.t_max, params.n_steps)

    # Compute integrand
    if use_log_space:
        # Compute in log space and exponentiate
        log_vals = integrand_log(t_grid, params)
        # Handle underflow gracefully
        with np.errstate(under='ignore'):
            integrand_vals = np.exp(log_vals)
        # Set underflowed values to 0
        integrand_vals = np.where(np.isfinite(integrand_vals), integrand_vals, 0.0)
    else:
        integrand_vals = integrand(t_grid, params)

    # Integrate using trapezoidal rule
    capacity = trapezoid_log_grid(integrand_vals, t_grid)

    # Check convergence: compute tail contribution
    # Find index where t > t_max / 10 (last decade)
    t_threshold = params.t_max / 10.0
    tail_idx = np.searchsorted(t_grid, t_threshold)
    if tail_idx < len(t_grid) - 1:
        tail_integral = trapezoid_log_grid(integrand_vals[tail_idx:], t_grid[tail_idx:])
        relative_tail = tail_integral / capacity if capacity > 0 else 0.0
    else:
        relative_tail = 0.0

    # Check theoretical convergence
    _, theoretically_converges = convergence_criterion(params)

    # Practical convergence: tail should be small fraction of total
    converged = theoretically_converges and relative_tail < 0.1

    return IntegrationResult(
        capacity=capacity,
        t_grid=t_grid,
        integrand_values=integrand_vals,
        converged=converged,
        relative_tail=relative_tail,
        params=params,
    )


def compute_capacity_quick(t_grid: NDArray[np.float64],
                           params: TSEParameters) -> float:
    """Quick capacity computation with pre-supplied grid.

    Useful for parameter sweeps where the grid is fixed.

    Args:
        t_grid: Pre-computed time grid
        params: TSE model parameters

    Returns:
        Operational capacity value
    """
    integrand_vals = integrand(t_grid, params)
    return trapezoid_log_grid(integrand_vals, t_grid)


def convergence_study(params: TSEParameters,
                      t_max_values: Optional[list[float]] = None,
                      n_steps: int = 10000) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Study convergence of capacity with increasing t_max.

    Args:
        params: Base TSE parameters (t_max will be varied)
        t_max_values: List of t_max values to test (default: [1e3, ..., 1e8])
        n_steps: Number of integration steps per run

    Returns:
        Tuple of (t_max_array, capacity_array)
    """
    if t_max_values is None:
        t_max_values = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

    t_max_arr = np.array(t_max_values)
    capacity_arr = np.zeros_like(t_max_arr)

    for i, t_max in enumerate(t_max_values):
        test_params = TSEParameters(
            t_min=params.t_min,
            t_max=t_max,
            n_steps=n_steps,
            eps=params.eps,
            n0=params.n0,
            sv0=params.sv0,
            P0=params.P0,
            alpha=params.alpha,
            gamma_n=params.gamma_n,
            gamma_sv=params.gamma_sv,
            gamma_P=params.gamma_P,
        )
        result = compute_capacity(test_params)
        capacity_arr[i] = result.capacity

    return t_max_arr, capacity_arr


def check_nsteps_convergence(params: TSEParameters,
                             nsteps_values: Optional[list[int]] = None) -> dict:
    """Check that the integration is converged with respect to n_steps.

    Args:
        params: TSE parameters
        nsteps_values: List of n_steps to test

    Returns:
        Dictionary with nsteps -> capacity mapping and relative changes
    """
    if nsteps_values is None:
        nsteps_values = [1000, 2000, 5000, 10000, 20000]

    results = {}
    capacities = []

    for n_steps in nsteps_values:
        test_params = TSEParameters(
            t_min=params.t_min,
            t_max=params.t_max,
            n_steps=n_steps,
            eps=params.eps,
            n0=params.n0,
            sv0=params.sv0,
            P0=params.P0,
            alpha=params.alpha,
            gamma_n=params.gamma_n,
            gamma_sv=params.gamma_sv,
            gamma_P=params.gamma_P,
        )
        result = compute_capacity(test_params)
        capacities.append(result.capacity)
        results[n_steps] = result.capacity

    # Compute relative changes
    capacities = np.array(capacities)
    rel_changes = np.abs(np.diff(capacities)) / capacities[:-1]

    return {
        "capacities": dict(zip(nsteps_values, capacities)),
        "relative_changes": rel_changes.tolist(),
        "converged": all(r < 0.01 for r in rel_changes[-2:]) if len(rel_changes) >= 2 else False,
    }
