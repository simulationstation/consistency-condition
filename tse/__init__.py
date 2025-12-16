"""
TSE - Terminal-State Exclusion / Operational Capacity

A research-grade prototype for computing the Operational Capacity functional
and studying terminal vs nonterminal scenarios under different scaling assumptions.

Usage:
    python -m tse --help
    python -m tse --scenario mild_dilution
    python -m tse --sweep --outdir outputs

Main components:
    - model: Mathematical definitions (w, n, sv, I, P, dK, integrand)
    - integrate: Numerical integration routines
    - experiments: Parameter sweeps and scenario studies
    - plot: Visualization utilities
    - report: Report generation
"""

__version__ = "0.1.0"
__author__ = "TSE Research Team"

from .model import (
    TSEParameters,
    SCENARIOS,
    get_scenario_params,
    w,
    n,
    sv,
    I,
    P_free,
    dK_dt,
    integrand,
    total_exponent,
    convergence_criterion,
)

from .integrate import (
    compute_capacity,
    IntegrationResult,
    make_log_grid,
    trapezoid_log_grid,
    convergence_study,
)

__all__ = [
    "TSEParameters",
    "SCENARIOS",
    "get_scenario_params",
    "w",
    "n",
    "sv",
    "I",
    "P_free",
    "dK_dt",
    "integrand",
    "total_exponent",
    "convergence_criterion",
    "compute_capacity",
    "IntegrationResult",
    "make_log_grid",
    "trapezoid_log_grid",
    "convergence_study",
]
