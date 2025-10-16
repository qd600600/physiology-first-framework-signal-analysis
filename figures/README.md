# W(t) Framework Figures

This directory contains the 9 publication-quality figures for the W(t) Stress Accumulation Framework whitepaper.

## Figure List

### Figure 1: WAIC Comparison (Continuous vs Discrete Models)
- **File**: `fig1_waic_comparison.png`
- **Content**: Bar chart comparing WAIC values for continuous vs discrete models across 6 datasets
- **Key Finding**: All datasets show decisive preference for continuous models (WAIC Δ < -10)
- **Resolution**: 300 DPI, publication quality

### Figure 2: Stress Stratification Effects
- **File**: `fig2_stratification_boxplot.png`
- **Content**: Box plots showing α and β parameter differences between high/low stress groups
- **Key Finding**: Extreme effect sizes (d = 16.80 for α, d = 9.06 for β)
- **Resolution**: 300 DPI, publication quality

### Figure 3: W(t) Time Series Trajectories
- **File**: `fig3_trajectories_timeseries.png`
- **Content**: Dynamic W(t) trajectories across 6 datasets showing continuous accumulation
- **Key Finding**: Smooth, continuous stress accumulation patterns
- **Resolution**: 300 DPI, publication quality

### Figure 4: Context-Specific Alpha Benchmarks
- **File**: `fig4_context_alpha_barplot.png`
- **Content**: Bar chart showing α parameter values across different environmental contexts
- **Key Finding**: Workplace shows highest risk (α = 5.01), driving shows lowest (α = 0.30)
- **Resolution**: 300 DPI, publication quality

### Figure 5: LRI-Alpha Correlation
- **File**: `fig5_lri_alpha_scatter.png`
- **Content**: Scatter plot with regression line showing LRI-α coupling
- **Key Finding**: Strong negative correlation (r = -0.65)
- **Resolution**: 300 DPI, publication quality

### Figure 6: Risk Stratification Decision Tree
- **File**: `fig6_risk_decision_tree.png`
- **Content**: Flowchart showing clinical decision tree for risk stratification
- **Key Finding**: Clear thresholds for intervention prioritization
- **Resolution**: 300 DPI, publication quality

### Figure 7: Timescale Separation Diagram
- **File**: `fig7_timescale_separation.png`
- **Content**: Schematic showing α (minutes) vs β (hours-days) timescales
- **Key Finding**: Distinct temporal dynamics for accumulation vs recovery
- **Resolution**: 300 DPI, publication quality

### Figure 8: Beta-Boost Intervention Simulation
- **File**: `fig8_intervention_simulation.png`
- **Content**: W(t) trajectory showing intervention effects on stress reduction
- **Key Finding**: β-boost strategies show 35% stress reduction
- **Resolution**: 300 DPI, publication quality

### Figure 9: Multi-modal Fusion Performance Heatmap
- **File**: `fig9_performance_heatmap.png`
- **Content**: Heatmap showing R² performance across 11 datasets
- **Key Finding**: Exceptional performance (R² = 0.9987 ± 0.0003)
- **Resolution**: 300 DPI, publication quality

## Figure Generation Notes

- All figures are generated using Python matplotlib/seaborn with 300 DPI resolution
- Color schemes optimized for both color and grayscale printing
- Font sizes and line weights adjusted for publication standards
- Statistical annotations include p-values, confidence intervals, and effect sizes
- Figures are designed to be self-contained with clear legends and axis labels

## Data Sources

- **L1 Complete Inventory**: Performance metrics from 11 publicly available datasets (1,184,135 samples)
- **L2 Multimodal Fusion**: Statistical results from hierarchical Bayesian validation across all datasets
- **L3 Stress Stratification**: Clinical thresholds from empirical risk stratification analysis (6 datasets, 35,497 samples)
- **L4 Context Benchmarking**: Environmental risk thresholds from context-specific analysis (5 environments)
- **L5 Clinical Translation**: Intervention effects from simulated intervention studies

## Citation

When using these figures, please cite the W(t) framework whitepaper:

Li, P. (2025). W(t) Stress Accumulation Framework: Bayesian Continuity Validation (BF > 10³¹) and Extreme Physiological Stratification (d > 10). Technical Whitepaper. DOI: 10.2139/ssrn.5589750

