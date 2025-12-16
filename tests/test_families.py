import numpy as np

from tse.integrate import make_log_grid
from tse.experiments_families import (
    evaluate_family,
    PERSISTENT_STATUS,
    TERMINAL_STATUS,
)
from tse.analysis import compute_window_capacity, fit_tail_slope
from tse.families import make_activity


def test_power_law_slope_recovery():
    ts = make_log_grid(1.0, 1e4, 2000)
    p = 2.0
    f_vals = make_activity(ts, "power_law", {"A": 1.0, "t0": 1.0, "p": p})
    T_vals = np.logspace(0, 3.5, 20)
    C_window = compute_window_capacity(ts, f_vals, T_vals)
    slope = fit_tail_slope(T_vals, C_window, K_slope=6)
    assert slope.valid
    expected = 1.0 - p
    assert abs(slope.b - expected) < 0.2


def test_exponential_is_terminal():
    ts = make_log_grid(1.0, 1e5, 1500)
    diag = evaluate_family(
        ts,
        "exponential",
        None,
        nT=12,
        K_last=4,
        K_slope=4,
        C_ZERO_WIN=1e-8,
        B_MIN_PERSIST=0.1,
    )
    assert diag.status == TERMINAL_STATUS
    finite_caps = diag.C_window[np.isfinite(diag.C_window)]
    assert finite_caps[-1] < finite_caps[0] * 1e-3


def test_persistent_plateau_persistent():
    ts = make_log_grid(1.0, 1e4, 1200)
    diag = evaluate_family(
        ts,
        "persistent_plateau",
        {"A": 0.5},
        nT=10,
        K_last=3,
        K_slope=4,
        C_ZERO_WIN=1e-6,
        B_MIN_PERSIST=0.1,
    )
    assert diag.status == PERSISTENT_STATUS
    assert diag.slope.valid
    assert diag.slope.b > 0.5


def test_pulse_train_not_terminal():
    ts = make_log_grid(1.0, 1e5, 3000)
    diag = evaluate_family(
        ts,
        "persistent_pulse_train",
        None,
        nT=15,
        K_last=4,
        K_slope=5,
        C_ZERO_WIN=1e-6,
        B_MIN_PERSIST=0.1,
    )
    assert diag.status != TERMINAL_STATUS
    finite_caps = diag.C_window[np.isfinite(diag.C_window)]
    assert finite_caps[-1] / max(finite_caps[0], 1e-12) > 0.1
