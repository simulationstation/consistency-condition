"""
Family-based experiments for late-time activity diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
import json
import numpy as np
from numpy.typing import NDArray

from .integrate import make_log_grid, select_T_values
from .families import get_family_names, make_activity, default_family_params
from .analysis import (
    compute_window_capacity,
    fit_tail_slope,
    tail_median,
    tail_median_f,
    SlopeResult,
)


@dataclass
class FamilyDiagnostics:
    family: str
    params: Dict
    ts: NDArray[np.float64]
    f_values: NDArray[np.float64]
    T_values: NDArray[np.float64]
    C_window: NDArray[np.float64]
    C_win_last_median: float
    slope: SlopeResult
    window_status: str
    instantaneous_status: str
    combined_status: str
    f_tail_ratio: float
    f_tail_median_lastM: float


TERMINAL_STATUS = "Terminal"
LONG_TAIL_STATUS = "LongTailTerminal"
PERSISTENT_STATUS = "Persistent"
INSTANT_TERMINAL = "InstantaneousTerminal"
INSTANT_ACTIVE = "InstantaneousActive"

COMBINED_WINDOW_PERSIST_INSTANT_TERM = "WindowPersistent_InstantaneousTerminal"


def classify_family(C_win_last_median: float,
                    slope: SlopeResult,
                    *,
                    C_ZERO_WIN: float,
                    B_MIN_PERSIST: float) -> str:
    if not np.isfinite(C_win_last_median):
        return TERMINAL_STATUS

    if C_win_last_median < C_ZERO_WIN:
        return TERMINAL_STATUS

    if slope.valid and slope.b < -B_MIN_PERSIST:
        return LONG_TAIL_STATUS

    return PERSISTENT_STATUS


def evaluate_family(ts: NDArray[np.float64],
                    family: str,
                    params: Optional[Dict],
                    *,
                    nT: int,
                    K_last: int,
                    K_slope: int,
                    C_ZERO_WIN: float,
                    B_MIN_PERSIST: float,
                    F_ZERO: float,
                    M_tail: int) -> FamilyDiagnostics:
    f_vals = make_activity(ts, family, params)

    effective_M_tail = max(1, min(M_tail, max(1, int(len(ts) // 10))))

    T_values = select_T_values(float(ts.min()), float(ts.max()), nT, factor=2.0)
    C_window = compute_window_capacity(ts, f_vals, T_values)

    C_win_last_median = tail_median(C_window, K_last)
    slope = fit_tail_slope(T_values, C_window, K_slope)

    window_status = classify_family(C_win_last_median, slope,
                                    C_ZERO_WIN=C_ZERO_WIN,
                                    B_MIN_PERSIST=B_MIN_PERSIST)

    f_tail_ratio = float(f_vals[-1] / max(f_vals[0], 1e-30))
    f_tail_median_lastM = tail_median_f(f_vals, effective_M_tail)

    if np.isfinite(f_tail_median_lastM) and f_tail_median_lastM < F_ZERO:
        instantaneous_status = INSTANT_TERMINAL
    else:
        instantaneous_status = INSTANT_ACTIVE

    if window_status == PERSISTENT_STATUS and instantaneous_status == INSTANT_TERMINAL:
        combined_status = COMBINED_WINDOW_PERSIST_INSTANT_TERM
    elif window_status == PERSISTENT_STATUS and instantaneous_status == INSTANT_ACTIVE:
        combined_status = PERSISTENT_STATUS
    elif window_status == LONG_TAIL_STATUS:
        combined_status = LONG_TAIL_STATUS
    else:
        combined_status = TERMINAL_STATUS

    return FamilyDiagnostics(
        family=family,
        params=params or default_family_params(family),
        ts=ts,
        f_values=f_vals,
        T_values=T_values,
        C_window=C_window,
        C_win_last_median=C_win_last_median,
        slope=slope,
        window_status=window_status,
        instantaneous_status=instantaneous_status,
        combined_status=combined_status,
        f_tail_ratio=f_tail_ratio,
        f_tail_median_lastM=f_tail_median_lastM,
    )


def diagnostics_to_json(diag: FamilyDiagnostics) -> Dict:
    return {
        "family": diag.family,
        "params": diag.params,
        "C_win_last_median": diag.C_win_last_median,
        "slope_b": diag.slope.b,
        "slope_n_used": diag.slope.n_used,
        "slope_valid": diag.slope.valid,
        "status": diag.window_status,  # backward compatibility
        "window_status": diag.window_status,
        "instantaneous_status": diag.instantaneous_status,
        "combined_status": diag.combined_status,
        "f_tail_ratio": diag.f_tail_ratio,
        "f_tail_median_lastM": diag.f_tail_median_lastM,
        "T_values": diag.T_values.tolist(),
        "C_window": diag.C_window.tolist(),
    }


def run_family_suite(*,
                     tmin: float,
                     tmax: float,
                     nsteps: int,
                     nT: int,
                     K_last: int,
                     K_slope: int,
                     C_ZERO_WIN: float,
                     B_MIN_PERSIST: float,
                     F_ZERO: float,
                     M_tail: int,
                     outdir: Path,
                     family: Optional[str] = None,
                     family_params: Optional[Dict] = None,
                     quiet: bool = False) -> list[FamilyDiagnostics]:
    """Run diagnostics for a suite of activity families."""

    ts = make_log_grid(tmin, tmax, nsteps)
    families: Iterable[str] = [family] if family else get_family_names()

    results: list[FamilyDiagnostics] = []
    for fam in families:
        params = default_family_params(fam)
        if fam == family and family_params:
            params.update(family_params)
        diag = evaluate_family(
            ts,
            fam,
            params,
            nT=nT,
            K_last=K_last,
            K_slope=K_slope,
            C_ZERO_WIN=C_ZERO_WIN,
            B_MIN_PERSIST=B_MIN_PERSIST,
            F_ZERO=F_ZERO,
            M_tail=M_tail,
        )
        results.append(diag)
        if not quiet:
            print(
                "Family {fam}: window_status={status}, inst_status={instant}, "
                "C_win_median={cwin:.3e}, slope={slope:.3f}".format(
                    fam=fam,
                    status=diag.window_status,
                    instant=diag.instantaneous_status,
                    cwin=diag.C_win_last_median,
                    slope=diag.slope.b,
                )
            )

    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results_families.json"
    json_results = {
        "tmin": tmin,
        "tmax": tmax,
        "nsteps": nsteps,
        "nT": nT,
        "K_last": K_last,
        "K_slope": K_slope,
        "C_ZERO_WIN": C_ZERO_WIN,
        "B_MIN_PERSIST": B_MIN_PERSIST,
        "F_ZERO": F_ZERO,
        "M_tail": M_tail,
        "families": [diagnostics_to_json(r) for r in results],
    }
    results_path.write_text(json.dumps(json_results, indent=2))

    return results
