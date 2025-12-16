"""
Unit tests for the TSE module.

Tests cover:
- Model function correctness
- Integration convergence with n_steps
- Scenario ordering (persistent_power > strong_dilution)
- Parameter validation
"""

import pytest
import numpy as np

from tse.model import (
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
from tse.integrate import (
    compute_capacity,
    make_log_grid,
    trapezoid_log_grid,
    check_nsteps_convergence,
)
from tse.experiments import (
    run_tmax_convergence,
    run_single_scenario,
    compute_scenario_diagnostics,
)


class TestParameters:
    """Tests for TSEParameters validation."""

    def test_default_parameters(self):
        """Test default parameter creation."""
        params = TSEParameters()
        assert params.t_min == 1.0
        assert params.t_max == 1e8
        assert params.eps == 0.2
        assert params.n_steps == 10000

    def test_invalid_tmin(self):
        """Test that t_min < 1 raises error."""
        with pytest.raises(ValueError, match="t_min must be >= 1.0"):
            TSEParameters(t_min=0.5)

    def test_invalid_eps(self):
        """Test that eps <= 0 raises error."""
        with pytest.raises(ValueError, match="eps must be > 0"):
            TSEParameters(eps=0)

        with pytest.raises(ValueError, match="eps must be > 0"):
            TSEParameters(eps=-0.1)

    def test_invalid_tmax(self):
        """Test that t_max <= t_min raises error."""
        with pytest.raises(ValueError, match="t_max .* must be > t_min"):
            TSEParameters(t_min=10, t_max=5)


class TestModelFunctions:
    """Tests for model function correctness."""

    def test_w_at_t0(self):
        """Test weighting function at t=0."""
        t = np.array([0.0])
        # w(0) = 1 / (1+0)^(1+eps) = 1
        result = w(t, eps=0.2)
        np.testing.assert_allclose(result, [1.0])

    def test_w_decay(self):
        """Test that w(t) decays as expected."""
        t = np.array([0.0, 1.0, 10.0, 100.0])
        eps = 0.2
        result = w(t, eps)

        # Should decay as (1+t)^-(1+eps)
        expected = (1 + t) ** (-(1 + eps))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_n_constant_when_gamma_zero(self):
        """Test n(t) is constant when gamma_n = 0."""
        t = np.array([1.0, 10.0, 100.0, 1000.0])
        result = n(t, n0=5.0, gamma_n=0.0)
        np.testing.assert_allclose(result, 5.0 * np.ones_like(t))

    def test_n_decay(self):
        """Test n(t) decays correctly."""
        t = np.array([0.0, 1.0, 9.0])
        n0, gamma_n = 2.0, 1.5
        result = n(t, n0, gamma_n)

        expected = n0 / (1 + t) ** gamma_n
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_I_product(self):
        """Test that I = n * sv."""
        t = np.array([1.0, 10.0, 100.0])
        n0, sv0 = 2.0, 3.0
        gamma_n, gamma_sv = 1.0, 0.5

        result = I(t, n0, sv0, gamma_n, gamma_sv)
        expected = n(t, n0, gamma_n) * sv(t, sv0, gamma_sv)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_dK_dt_positive(self):
        """Test dK/dt is always positive for positive alpha, P0."""
        t = np.logspace(0, 8, 100)
        result = dK_dt(t, alpha=1.0, P0=1.0, gamma_P=2.0)
        assert np.all(result >= 0)

    def test_integrand_positive(self):
        """Test integrand is non-negative."""
        params = TSEParameters()
        t = make_log_grid(params.t_min, params.t_max, params.n_steps)
        result = integrand(t, params)
        assert np.all(result >= 0)

    def test_total_exponent(self):
        """Test total exponent calculation."""
        params = TSEParameters(eps=0.2, gamma_n=1.0, gamma_sv=0.5, gamma_P=2.0)
        beta = total_exponent(params)
        expected = 1 + 0.2 + 1.0 + 0.5 + 2.0
        assert beta == expected


class TestIntegration:
    """Tests for numerical integration."""

    def test_log_grid_endpoints(self):
        """Test log grid has correct endpoints."""
        grid = make_log_grid(1.0, 1000.0, 100)
        np.testing.assert_allclose(grid[0], 1.0, rtol=1e-10)
        np.testing.assert_allclose(grid[-1], 1000.0, rtol=1e-10)
        assert len(grid) == 100

    def test_log_grid_spacing(self):
        """Test log grid is logarithmically spaced."""
        grid = make_log_grid(1.0, 1e6, 7)
        log_grid = np.log10(grid)
        diffs = np.diff(log_grid)
        # Should be equally spaced in log
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10)

    def test_trapezoid_simple(self):
        """Test trapezoidal integration on simple function."""
        # Integrate y = 1 from t=1 to t=10, should give 9
        t = np.linspace(1, 10, 1000)
        y = np.ones_like(t)
        result = trapezoid_log_grid(y, t)
        np.testing.assert_allclose(result, 9.0, rtol=1e-3)

    def test_trapezoid_power_law(self):
        """Test trapezoidal integration on power law."""
        # Integrate t^2 from 1 to 10
        # Exact: (10^3 - 1^3)/3 = 333
        t = np.linspace(1, 10, 10000)
        y = t ** 2
        result = trapezoid_log_grid(y, t)
        expected = (10 ** 3 - 1 ** 3) / 3
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_nsteps_convergence(self):
        """Test that increasing n_steps changes C by < 1%."""
        params = TSEParameters(
            t_min=1.0,
            t_max=1e6,
            n_steps=5000,
            gamma_n=1.0,
            gamma_P=1.0,
        )

        result1 = compute_capacity(TSEParameters(
            **{**params.__dict__, "n_steps": 5000}
        ))
        result2 = compute_capacity(TSEParameters(
            **{**params.__dict__, "n_steps": 10000}
        ))

        rel_change = abs(result2.capacity - result1.capacity) / result1.capacity
        assert rel_change < 0.01, f"Relative change {rel_change:.4f} >= 1%"


class TestScenarios:
    """Tests for scenario behavior."""

    def test_scenario_definitions_exist(self):
        """Test all required scenarios are defined."""
        required = [
            "mild_dilution",
            "strong_dilution",
            "desitter_like_no_free_power",
            "persistent_power_nonterminal",
        ]
        for name in required:
            assert name in SCENARIOS

    def test_get_scenario_params(self):
        """Test scenario parameter retrieval."""
        params = get_scenario_params("mild_dilution")
        assert params.gamma_n == 1.0
        assert params.gamma_P == 1.0
        assert params.gamma_sv == 0.0

    def test_unknown_scenario(self):
        """Test error on unknown scenario."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario_params("nonexistent_scenario")

    def test_persistent_power_larger_than_strong_dilution(self):
        """Test that persistent_power gives larger C than strong_dilution."""
        # Use same t_max for fair comparison
        base = TSEParameters(t_min=1.0, t_max=1e6, n_steps=5000)

        persistent = get_scenario_params("persistent_power_nonterminal", base)
        strong = get_scenario_params("strong_dilution", base)

        C_persistent = compute_capacity(persistent).capacity
        C_strong = compute_capacity(strong).capacity

        assert C_persistent > C_strong, \
            f"Expected C_persistent ({C_persistent:.3e}) > C_strong ({C_strong:.3e})"

    def test_convergence_criterion(self):
        """Test theoretical convergence criterion."""
        # With eps=0.2, gamma_n=1, gamma_sv=0, gamma_P=1
        # total exp = 1 + 0.2 + 1 + 0 + 1 = 3.2 > 1, should converge
        params = get_scenario_params("mild_dilution")
        beta, converges = convergence_criterion(params)
        assert converges
        assert beta > 1


class TestExperiments:
    """Tests for experiment functions."""

    def test_run_single_scenario(self):
        """Test running a single scenario."""
        result = run_single_scenario("mild_dilution", TSEParameters(
            t_max=1e4, n_steps=1000
        ))
        assert result.capacity_total > 0
        assert len(result.integration.t_grid) == 1000

    def test_tmax_convergence_returns_all_values(self):
        """Test t_max convergence returns values for all t_max."""
        result = run_tmax_convergence(
            "mild_dilution",
            TSEParameters(t_min=1.0, n_steps=1000),
            t_max_values=[1e3, 1e4, 1e5],
        )
        assert len(result.t_max_values) == 3
        assert len(result.capacity_values) == 3
        assert result.scenario_name == "mild_dilution"

    def test_window_capacity_classification(self):
        """Strong dilution should have vanishing window capacity at late times."""
        params = TSEParameters(t_max=1e6, n_steps=1500)
        diag = compute_scenario_diagnostics(
            "strong_dilution", params, nT=20, K_last=5, C_ZERO_WIN=1e-12
        )

        valid = diag.capacity_window_values[~np.isnan(diag.capacity_window_values)]
        assert np.all(np.diff(valid) <= 0), "Window capacities should decay with T"
        assert diag.capacity_window_last_median < 1e-12
        assert diag.status == "Terminal"

    def test_persistent_power_nonterminal_window(self):
        """Persistent power should remain nonterminal via window capacity."""
        params = TSEParameters(t_max=1e6, n_steps=1500)
        diag = compute_scenario_diagnostics(
            "persistent_power_nonterminal", params, nT=20, K_last=5, C_ZERO_WIN=1e-12
        )

        assert diag.capacity_window_last_median > 1e-10
        assert diag.status == "Nonterminal"


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_no_nan_in_integrand(self):
        """Test integrand has no NaN values."""
        params = TSEParameters(t_max=1e10, n_steps=10000)
        t = make_log_grid(params.t_min, params.t_max, params.n_steps)
        vals = integrand(t, params)
        assert not np.any(np.isnan(vals))

    def test_no_inf_in_integrand(self):
        """Test integrand has no inf values."""
        params = TSEParameters(t_max=1e10, n_steps=10000)
        t = make_log_grid(params.t_min, params.t_max, params.n_steps)
        vals = integrand(t, params)
        assert not np.any(np.isinf(vals))

    def test_capacity_finite(self):
        """Test computed capacity is finite."""
        for scenario_name in SCENARIOS:
            params = get_scenario_params(scenario_name, TSEParameters(
                t_max=1e6, n_steps=2000
            ))
            result = compute_capacity(params)
            assert np.isfinite(result.capacity), f"Non-finite C for {scenario_name}"

    def test_extreme_exponents(self):
        """Test with extreme exponent values."""
        # Very high decay - should give very small C
        params = TSEParameters(
            gamma_n=5.0, gamma_P=5.0, t_max=1e6, n_steps=2000
        )
        result = compute_capacity(params)
        assert np.isfinite(result.capacity)
        assert result.capacity >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
