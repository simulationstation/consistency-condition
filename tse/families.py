"""
Activity family definitions for direct late-time behavior.

Each family provides f(t) directly (rather than via model exponents).
Families include decays that vanish at infinity and explicit
nonterminal constructions (plateau and pulse train) that violate the
f(t) -> 0 assumption. All functions are deterministic.
"""
from __future__ import annotations

from typing import Dict, List
import numpy as np
from numpy.typing import NDArray

FAMILY_NAMES = [
    "power_law",
    "exponential",
    "stretched_exponential",
    "logarithmic",
    "oscillatory_envelope",
    "plateau_then_decay",
    "persistent_plateau",
    "persistent_pulse_train",
]


def get_family_names() -> List[str]:
    """Return the list of supported family names."""

    return FAMILY_NAMES.copy()


def default_family_params(family: str) -> Dict:
    """Return default parameters for a family."""

    if family == "power_law":
        return {"A": 1.0, "t0": 1.0, "p": 2.0}
    if family == "exponential":
        return {"A": 1.0, "t0": 10.0}
    if family == "stretched_exponential":
        return {"A": 1.0, "t0": 20.0, "beta": 0.5}
    if family == "logarithmic":
        return {"A": 1.0, "t0": 5.0, "q": 1.5}
    if family == "oscillatory_envelope":
        return {"A": 1.0, "t0": 5.0, "p": 1.5, "omega": 0.2}
    if family == "plateau_then_decay":
        return {"A": 1.0, "T_plateau": 50.0, "p": 2.0}
    if family == "persistent_plateau":
        return {"A": 1.0}
    if family == "persistent_pulse_train":
        return {"A": 1.0, "t0": 10.0, "frac": 0.2, "cap_height": 50.0}
    raise ValueError(f"Unknown family: {family}")


def _get_params(family: str, params: Dict | None) -> Dict:
    base = default_family_params(family)
    if params:
        base.update(params)
    return base


def make_activity(ts: NDArray[np.float64], family: str, params: Dict | None = None) -> NDArray[np.float64]:
    """Generate activity f(t) for a given family.

    Args:
        ts: Time samples (monotonic, linear domain)
        family: Name of the family
        params: Optional parameter overrides

    Returns:
        Array of f(t) values (nonnegative)
    """

    if family not in FAMILY_NAMES:
        raise ValueError(f"Unsupported family: {family}")

    p = _get_params(family, params)
    t = np.asarray(ts, dtype=float)

    if family == "power_law":
        A, t0, exponent = p["A"], max(p["t0"], 1.0), p["p"]
        return A * np.power(1.0 + t / t0, -exponent)

    if family == "exponential":
        A, t0 = p["A"], max(p["t0"], 1.0)
        return A * np.exp(-t / t0)

    if family == "stretched_exponential":
        A, t0, beta = p["A"], max(p["t0"], 1.0), p["beta"]
        beta = np.clip(beta, 1e-6, 2.0)
        return A * np.exp(-np.power(t / t0, beta))

    if family == "logarithmic":
        A, t0, q = p["A"], max(p["t0"], 1.0), max(p["q"], 1e-6)
        with np.errstate(divide="ignore"):
            denom = np.log(np.e + t / t0)
        denom = np.maximum(denom, 1e-12)
        return A / np.power(denom, q)

    if family == "oscillatory_envelope":
        A, t0, exponent, omega = p["A"], max(p["t0"], 1.0), p["p"], p["omega"]
        envelope = np.power(1.0 + t / t0, -exponent)
        osc = np.square(np.sin(omega * t))
        return A * envelope * osc

    if family == "plateau_then_decay":
        A, T_plateau, exponent = p["A"], max(p["T_plateau"], 1.0), p["p"]
        f_vals = np.where(t < T_plateau, A, A * np.power(np.maximum(t, T_plateau) / T_plateau, -exponent))
        return f_vals

    if family == "persistent_plateau":
        return np.full_like(t, fill_value=p["A"], dtype=float)

    if family == "persistent_pulse_train":
        A = p["A"]
        t0 = max(p["t0"], 1.0)
        frac = float(np.clip(p.get("frac", 0.2), 1e-3, 0.9))
        cap_height = max(float(p.get("cap_height", 50.0)), A)
        f_vals = np.zeros_like(t, dtype=float)
        k = 0
        t_max = float(t.max())
        while True:
            t_k = t0 * (2.0 ** k)
            if t_k > t_max * 1.1:
                break
            width = frac * t_k
            height = min(A / frac, cap_height)
            start = t_k
            end = t_k + width
            mask = (t >= start) & (t <= end)
            f_vals = np.where(mask, height, f_vals)
            k += 1
        return f_vals

    raise ValueError(f"Unsupported family: {family}")
