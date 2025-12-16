import numpy as np

from tse.analysis import compute_window_capacity, fit_tail_slope
from tse.experiments_families import (
    COMBINED_WINDOW_PERSIST_INSTANT_TERM,
    INSTANT_ACTIVE,
    INSTANT_TERMINAL,
    LONG_TAIL_STATUS,
    PERSISTENT_STATUS,
    TERMINAL_STATUS,
    evaluate_family,
)
from tse.families import make_activity
from tse.integrate import make_log_grid


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


def test_exponential_is_terminal_in_both_axes():
    ts = make_log_grid(1.0, 1e5, 1500)
    diag = evaluate_family(
        ts,
        "exponential",
        None,
        nT=16,
        K_last=4,
        K_slope=4,
        C_ZERO_WIN=1e-10,
        B_MIN_PERSIST=0.1,
        F_ZERO=1e-12,
        M_tail=100,
    )
    assert diag.instantaneous_status == INSTANT_TERMINAL
    assert diag.window_status == TERMINAL_STATUS
    finite_caps = diag.C_window[np.isfinite(diag.C_window)]
    assert finite_caps[-1] < finite_caps[0] * 1e-3


def test_persistent_plateau_active_in_both_axes():
    ts = make_log_grid(1.0, 1e4, 1200)
    diag = evaluate_family(
        ts,
        "persistent_plateau",
        {"A": 0.5},
        nT=12,
        K_last=4,
        K_slope=4,
        C_ZERO_WIN=1e-6,
        B_MIN_PERSIST=0.1,
        F_ZERO=1e-3,
        M_tail=50,
    )
    assert diag.instantaneous_status == INSTANT_ACTIVE
    assert diag.window_status == PERSISTENT_STATUS
    assert diag.combined_status == PERSISTENT_STATUS
    assert diag.slope.valid
    assert diag.slope.b > 0.5


def test_logarithmic_instantaneous_terminal_but_window_persistent():
    ts = make_log_grid(1.0, 1e6, 5000)
    diag = evaluate_family(
        ts,
        "logarithmic",
        {"q": 3.0},
        nT=18,
        K_last=4,
        K_slope=5,
        C_ZERO_WIN=1e-30,
        B_MIN_PERSIST=0.2,
        F_ZERO=1e-3,
        M_tail=200,
    )
    assert diag.instantaneous_status == INSTANT_TERMINAL
    assert diag.window_status in {PERSISTENT_STATUS, LONG_TAIL_STATUS}
    if diag.window_status == PERSISTENT_STATUS:
        assert diag.combined_status == COMBINED_WINDOW_PERSIST_INSTANT_TERM
    else:
        assert diag.combined_status == LONG_TAIL_STATUS
