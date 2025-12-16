"""
TSE Model Definitions

This module provides the mathematical building blocks for the
Terminal-State Exclusion / Operational Capacity framework.

Core definitions (toy model):
- w(t): temporal weighting function
- n(t): number density
- sv(t): velocity-averaged cross section <σv>
- I(t): interaction activity density = n(t) * sv(t)
- P_free(t): free power available
- dK_dt(t): accessible complexity growth rate = α * P_free(t)
- integrand(t): w(t) * I(t) * max(0, dK_dt)
- C: Operational Capacity = ∫ integrand(t) dt
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class TSEParameters:
    """Parameters for the TSE model.

    Attributes:
        t_min: Minimum time (must be >= 1 for numerical stability)
        t_max: Maximum time for integration
        n_steps: Number of log-spaced time steps
        eps: Exponent offset for weighting function (ε > 0)
        n0: Initial number density normalization
        sv0: Initial cross-section normalization
        P0: Initial free power normalization
        alpha: Complexity growth rate coefficient
        gamma_n: Power-law exponent for number density decay
        gamma_sv: Power-law exponent for cross-section decay
        gamma_P: Power-law exponent for free power decay
    """
    t_min: float = 1.0
    t_max: float = 1e8
    n_steps: int = 10000
    eps: float = 0.2
    n0: float = 1.0
    sv0: float = 1.0
    P0: float = 1.0
    alpha: float = 1.0
    gamma_n: float = 1.0
    gamma_sv: float = 0.0
    gamma_P: float = 1.0

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.t_min < 1.0:
            raise ValueError(f"t_min must be >= 1.0, got {self.t_min}")
        if self.t_max <= self.t_min:
            raise ValueError(f"t_max ({self.t_max}) must be > t_min ({self.t_min})")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.n_steps < 100:
            raise ValueError(f"n_steps must be >= 100, got {self.n_steps}")


# Predefined scenarios
SCENARIOS: dict[str, dict[str, float]] = {
    "mild_dilution": {
        "gamma_n": 1.0,
        "gamma_P": 1.0,
        "gamma_sv": 0.0,
    },
    "strong_dilution": {
        "gamma_n": 2.0,
        "gamma_P": 2.0,
        "gamma_sv": 0.0,
    },
    "desitter_like_no_free_power": {
        "gamma_n": 0.0,
        "gamma_P": 5.0,
        "gamma_sv": 0.0,
    },
    "persistent_power_nonterminal": {
        "gamma_n": 1.0,
        "gamma_P": 0.0,
        "gamma_sv": 0.0,
    },
}


def get_scenario_params(scenario_name: str, base_params: Optional[TSEParameters] = None) -> TSEParameters:
    """Get parameters for a named scenario.

    Args:
        scenario_name: Name of the scenario (must be in SCENARIOS)
        base_params: Optional base parameters to modify

    Returns:
        TSEParameters with scenario-specific exponents

    Raises:
        ValueError: If scenario_name is not recognized
    """
    if scenario_name not in SCENARIOS:
        valid = ", ".join(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Valid: {valid}")

    if base_params is None:
        base_params = TSEParameters()

    scenario = SCENARIOS[scenario_name]
    return TSEParameters(
        t_min=base_params.t_min,
        t_max=base_params.t_max,
        n_steps=base_params.n_steps,
        eps=base_params.eps,
        n0=base_params.n0,
        sv0=base_params.sv0,
        P0=base_params.P0,
        alpha=base_params.alpha,
        gamma_n=scenario["gamma_n"],
        gamma_sv=scenario["gamma_sv"],
        gamma_P=scenario["gamma_P"],
    )


def w(t: NDArray[np.float64], eps: float) -> NDArray[np.float64]:
    """Temporal weighting function.

    w(t) = 1 / (1 + t)^(1 + ε)

    For numerical stability at large t, compute in log space.

    Args:
        t: Time values (array)
        eps: Exponent offset (ε > 0)

    Returns:
        w(t) values
    """
    exponent = 1.0 + eps
    # Use log-space computation to avoid overflow at large t
    log_w = -exponent * np.log1p(t)
    return np.exp(log_w)


def n(t: NDArray[np.float64], n0: float, gamma_n: float) -> NDArray[np.float64]:
    """Number density as function of time.

    n(t) = n0 / (1 + t)^(γ_n)

    Args:
        t: Time values
        n0: Initial normalization
        gamma_n: Power-law exponent

    Returns:
        n(t) values
    """
    if gamma_n == 0:
        return np.full_like(t, n0)
    log_n = np.log(n0) - gamma_n * np.log1p(t)
    return np.exp(log_n)


def sv(t: NDArray[np.float64], sv0: float, gamma_sv: float) -> NDArray[np.float64]:
    """Velocity-averaged cross section <σv>(t).

    <σv>(t) = sv0 / (1 + t)^(γ_sv)

    Args:
        t: Time values
        sv0: Initial normalization
        gamma_sv: Power-law exponent

    Returns:
        <σv>(t) values
    """
    if gamma_sv == 0:
        return np.full_like(t, sv0)
    log_sv = np.log(sv0) - gamma_sv * np.log1p(t)
    return np.exp(log_sv)


def I(t: NDArray[np.float64], n0: float, sv0: float,
      gamma_n: float, gamma_sv: float) -> NDArray[np.float64]:
    """Interaction activity density.

    I(t) = n(t) * <σv>(t)

    Args:
        t: Time values
        n0: Number density normalization
        sv0: Cross-section normalization
        gamma_n: n(t) exponent
        gamma_sv: <σv>(t) exponent

    Returns:
        I(t) values
    """
    return n(t, n0, gamma_n) * sv(t, sv0, gamma_sv)


def P_free(t: NDArray[np.float64], P0: float, gamma_P: float) -> NDArray[np.float64]:
    """Free power available for complexity growth.

    P_free(t) = P0 / (1 + t)^(γ_P)

    Args:
        t: Time values
        P0: Initial power normalization
        gamma_P: Power-law exponent

    Returns:
        P_free(t) values
    """
    if gamma_P == 0:
        return np.full_like(t, P0)
    log_P = np.log(P0) - gamma_P * np.log1p(t)
    return np.exp(log_P)


def dK_dt(t: NDArray[np.float64], alpha: float, P0: float,
          gamma_P: float) -> NDArray[np.float64]:
    """Accessible complexity growth rate.

    dK_acc/dt = α * P_free(t)

    Args:
        t: Time values
        alpha: Growth coefficient
        P0: Power normalization
        gamma_P: Power exponent

    Returns:
        dK/dt values (always >= 0 due to P_free definition)
    """
    return alpha * P_free(t, P0, gamma_P)


def integrand(t: NDArray[np.float64], params: TSEParameters) -> NDArray[np.float64]:
    """Compute the integrand for operational capacity.

    integrand(t) = w(t) * I(t) * max(0, dK_acc/dt)

    Since dK/dt = α * P_free >= 0 for our model, max(0, ...) is always dK/dt.

    For numerical stability, we compute in log space where possible.

    Args:
        t: Time values
        params: TSE model parameters

    Returns:
        integrand(t) values
    """
    # Compute each component
    w_vals = w(t, params.eps)
    I_vals = I(t, params.n0, params.sv0, params.gamma_n, params.gamma_sv)
    dK_vals = dK_dt(t, params.alpha, params.P0, params.gamma_P)

    # max(0, dK/dt) - since dK >= 0 in our model, this is just dK
    dK_clamped = np.maximum(0.0, dK_vals)

    return w_vals * I_vals * dK_clamped


def integrand_log(t: NDArray[np.float64], params: TSEParameters) -> NDArray[np.float64]:
    """Compute log of the integrand for numerical stability.

    log(integrand) = log(w) + log(I) + log(max(eps, dK/dt))

    This avoids underflow for very small integrand values.

    Args:
        t: Time values
        params: TSE model parameters

    Returns:
        log(integrand(t)) values, or -inf where integrand is 0
    """
    # Total exponent for (1+t) term:
    # w: -(1+eps), n: -gamma_n, sv: -gamma_sv, P: -gamma_P
    total_exp = (1 + params.eps) + params.gamma_n + params.gamma_sv + params.gamma_P

    # Log of normalization constants
    log_norm = np.log(params.n0) + np.log(params.sv0) + np.log(params.alpha) + np.log(params.P0)

    # Compute log integrand
    log_integrand = log_norm - total_exp * np.log1p(t)

    return log_integrand


def total_exponent(params: TSEParameters) -> float:
    """Compute the total power-law exponent of the integrand.

    The integrand scales as (1+t)^(-β) where:
    β = (1 + ε) + γ_n + γ_sv + γ_P

    For convergence of the integral at large t, we need β > 1.

    Args:
        params: TSE model parameters

    Returns:
        Total exponent β
    """
    return (1 + params.eps) + params.gamma_n + params.gamma_sv + params.gamma_P


def convergence_criterion(params: TSEParameters) -> tuple[float, bool]:
    """Check if the integral converges at large t.

    The integral converges if total_exponent > 1.

    Args:
        params: TSE model parameters

    Returns:
        Tuple of (total_exponent, converges)
    """
    beta = total_exponent(params)
    return beta, beta > 1.0
