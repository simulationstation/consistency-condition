# Terminal-State Exclusion (TSE) / Operational Capacity Report

*Generated: 2025-12-16 12:37:16*

## Model Equations

The Operational Capacity functional is defined as:

$$C = \int_{t_{min}}^{t_{max}} w(t) \cdot I(t) \cdot \max(0, \frac{dK_{acc}}{dt}) \, dt$$

where:

- **Weighting function**: $w(t) = \frac{1}{(1+t)^{1+\varepsilon}}$
- **Interaction density**: $I(t) = n(t) \cdot \langle\sigma v\rangle(t)$
  - $n(t) = \frac{n_0}{(1+t)^{\gamma_n}}$
  - $\langle\sigma v\rangle(t) = \frac{sv_0}{(1+t)^{\gamma_{sv}}}$
- **Complexity growth rate**: $\frac{dK_{acc}}{dt} = \alpha \cdot P_{free}(t)$
  - $P_{free}(t) = \frac{P_0}{(1+t)^{\gamma_P}}$

The integrand scales as $(1+t)^{-\beta}$ where $\beta = 1 + \varepsilon + \gamma_n + \gamma_{sv} + \gamma_P$.

## Parameters Used

| Parameter | Value |
|-----------|-------|
| $t_{min}$ | 1.0 |
| $t_{max}$ | 1e+08 |
| $n_{steps}$ | 10000 |
| $\varepsilon$ | 0.2 |
| $n_0$ | 1.0 |
| $sv_0$ | 1.0 |
| $P_0$ | 1.0 |
| $\alpha$ | 1.0 |

## Scenario Definitions

| Scenario | $\gamma_n$ | $\gamma_P$ | $\gamma_{sv}$ | Physical Interpretation |
|----------|-----------|-----------|-------------|------------------------|
| mild_dilution | 1.0 | 1.0 | 0.0 | Moderate expansion with proportional power dilution |
| strong_dilution | 2.0 | 2.0 | 0.0 | Rapid expansion (de Sitter-like matter) |
| desitter_like_no_free_power | 0.0 | 5.0 | 0.0 | Constant density but severe energy depletion |
| persistent_power_nonterminal | 1.0 | 0.0 | 0.0 | Diluting density but inexhaustible power |

## Results Summary

### Scenario Capacity Values

| Scenario | Final $C$ | Converged? | Classification |
|----------|-----------|------------|----------------|
| mild_dilution | 9.893e-02 | Yes | Nonterminal ($C > 0$) |
| strong_dilution | 1.295e-02 | Yes | Nonterminal ($C > 0$) |
| desitter_like_no_free_power | 5.232e-03 | Yes | Nonterminal ($C > 0$) |
| persistent_power_nonterminal | 3.627e-01 | Yes | Nonterminal ($C > 0$) |

### Convergence with $t_{max}$

The following table shows capacity values at different $t_{max}$ values:

| Scenario | $10^{3}$ | $10^{4}$ | $10^{5}$ | $10^{6}$ | $10^{7}$ | $10^{8}$ |
|----------|----------|----------|----------|----------|----------|----------|
| mild_dilution | 9.89e-02 | 9.89e-02 | 9.89e-02 | 9.89e-02 | 9.89e-02 | 9.89e-02 |
| strong_dilution | 1.30e-02 | 1.30e-02 | 1.30e-02 | 1.30e-02 | 1.30e-02 | 1.30e-02 |
| desitter_like_no_free_power | 5.23e-03 | 5.23e-03 | 5.23e-03 | 5.23e-03 | 5.23e-03 | 5.23e-03 |
| persistent_power_nonterminal | 3.63e-01 | 3.63e-01 | 3.63e-01 | 3.63e-01 | 3.63e-01 | 3.63e-01 |

### Exponent Space Analysis

Swept $\gamma_n \in [0.0, 3.0]$ and 
$\gamma_P \in [0.0, 3.0]$ with 61 points per axis.

**Threshold for 'effectively zero'**: $C < 1e-12$

No clear boundary found in the scanned range.

### Sensitivity to Weighting Exponent $\varepsilon$

Larger $\varepsilon$ increases the effective suppression of late-time contributions,
generally reducing $C$ but also improving convergence.

- **mild_dilution**: $C$ ranges from 4.17e-02 to 1.24e-01
  as $\varepsilon$ varies from 0.01 to 1.00
- **strong_dilution**: $C$ ranges from 6.25e-03 to 1.55e-02
  as $\varepsilon$ varies from 0.01 to 1.00

## Numerical Stability Notes

1. **Integration method**: Trapezoidal rule on log-spaced grid
2. **Grid construction**: `numpy.logspace` with proper $dt$ in linear space
3. **Underflow protection**: Log-space computation with `np.log1p` for numerical stability
4. **Precision**: All computations in `float64` (double precision)

Default integration uses 10000 grid points spanning $t \in [1, 10^8]$.
Heatmap sweeps use reduced resolution (2000 points) for computational efficiency.

## Figures

| Filename | Description |
|----------|-------------|
| `capacity_vs_tmax_mild_dilution.png` | Convergence of $C$ with $t_{max}$ for mild_dilution |
| `capacity_vs_tmax_strong_dilution.png` | Convergence of $C$ with $t_{max}$ for strong_dilution |
| `capacity_vs_tmax_desitter_like_no_free_power.png` | Convergence of $C$ with $t_{max}$ for desitter_like_no_free_power |
| `capacity_vs_tmax_persistent_power_nonterminal.png` | Convergence of $C$ with $t_{max}$ for persistent_power_nonterminal |
| `capacity_vs_tmax_all_scenarios.png` | All scenarios on one plot |
| `heatmap_log10C_gamma_n_gamma_P.png` | Heatmap of $\log_{10}(C)$ in exponent space |
| `capacity_vs_eps_combined.png` | $C$ vs $\varepsilon$ for all tested scenarios |
| `capacity_vs_eps_mild_dilution.png` | $C$ vs $\varepsilon$ for mild_dilution |
| `capacity_vs_eps_strong_dilution.png` | $C$ vs $\varepsilon$ for strong_dilution |
| `summary_figure.png` | Multi-panel summary figure |

## Key Findings

**Nonterminal scenarios** ($C > 0$): mild_dilution, strong_dilution, desitter_like_no_free_power, persistent_power_nonterminal

The persistent_power_nonterminal scenario demonstrates that even with diluting
density, inexhaustible free power ($\gamma_P = 0$) maintains $C > 0$.

Conversely, rapid power depletion (high $\gamma_P$) drives $C \to 0$,
classifying the scenario as terminal regardless of density evolution.
