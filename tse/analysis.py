"""
Numerical helpers for late-time window diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class SlopeResult:
    b: float
    n_used: int
    valid: bool


def compute_window_capacity(ts: NDArray[np.float64],
                            f_vals: NDArray[np.float64],
                            T_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute window capacity using trapezoidal rule on linear ``t``."""

    capacities = np.empty_like(T_values, dtype=float)
    for i, T in enumerate(T_values):
        upper = 2.0 * T
        if upper > ts.max():
            capacities[i] = np.nan
            continue
        mask = (ts >= T) & (ts <= upper)
        if np.count_nonzero(mask) < 2:
            capacities[i] = np.nan
            continue
        capacities[i] = float(np.trapezoid(f_vals[mask], ts[mask]))
    return capacities


def tail_median(values: NDArray[np.float64], K_last: int) -> float:
    """Compute the median of the last ``K_last`` finite values.

    Returns ``np.nan`` if fewer than ``K_last`` finite values are available.
    """

    if K_last <= 0:
        return float("nan")

    finite_vals = values[np.isfinite(values)]
    if finite_vals.size < K_last:
        return float("nan")
    tail = finite_vals[-K_last:]
    return float(np.median(tail))


def tail_median_f(values: NDArray[np.float64], M_last: int) -> float:
    """Median of the last ``M_last`` finite instantaneous values."""

    return tail_median(values, M_last)


def fit_tail_slope(T_values: NDArray[np.float64],
                   C_values: NDArray[np.float64],
                   K_slope: int,
                   min_points: int = 3) -> SlopeResult:
    """Fit log-log slope over the last ``K_slope`` finite points.

    Returns slope ``b`` from ``log10 C = a + b log10 T``.
    """

    mask = np.isfinite(T_values) & np.isfinite(C_values) & (T_values > 0) & (C_values > 0)
    T_valid = T_values[mask]
    C_valid = C_values[mask]

    if T_valid.size < min_points:
        return SlopeResult(b=float("nan"), n_used=int(T_valid.size), valid=False)

    tail_T = T_valid[-K_slope:] if T_valid.size >= K_slope else T_valid
    tail_C = C_valid[-len(tail_T):]

    with np.errstate(divide="ignore"):
        x = np.log10(tail_T)
        y = np.log10(tail_C)

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return SlopeResult(b=float("nan"), n_used=int(len(x)), valid=False)

    b, a = np.polyfit(x, y, 1)
    return SlopeResult(b=float(b), n_used=int(len(x)), valid=True)
