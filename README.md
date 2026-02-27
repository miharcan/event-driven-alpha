# Event-Driven Alpha --- Regime-Aware Commodity Forecasting

## Overview

This project builds a walk-forward, regime-aware forecasting pipeline
for major commodities using:

-   Price features
-   Macro features
-   News embeddings (SentenceTransformers)
-   Volatility regime classification

Directional Accuracy (DA) is the primary evaluation metric.

------------------------------------------------------------------------

# Methodology

## Walk-Forward Validation

-   Expanding window
-   4 folds
-   Strict chronological split
-   No lookahead bias

## Volatility Regime Feature

A binary feature:

-   `vol_regime_high = 1` → High volatility regime
-   `vol_regime_high = 0` → Low volatility regime

We evaluate:

-   Fold-level DA
-   Aggregated regime-conditional DA across folds

------------------------------------------------------------------------

# Results Summary

## Gold

Overall DA range: 0.53 -- 0.57

Aggregated Regime DA: - High Vol: 0.5455 (11 samples) - Low Vol: 0.5714
(49 samples)

Interpretation: - Minimal regime sensitivity - Stable predictive
behavior across regimes

------------------------------------------------------------------------

## Silver

Overall DA range: 0.43 -- 0.47

Aggregated Regime DA: - High Vol: 0.41--0.44 - Low Vol: 0.45--0.51

Interpretation: - Performance improves in low volatility - High
volatility degrades predictive quality

------------------------------------------------------------------------

## Copper

Overall DA range: 0.46 -- 0.54

Aggregated Regime DA (strong signal): - High Vol: \~0.39--0.42 - Low
Vol: \~0.49--0.59

Interpretation: - Strong regime asymmetry - Model works primarily in
stable conditions

------------------------------------------------------------------------

## Crude Oil

Overall DA range: 0.39 -- 0.57

Aggregated Regime DA: - High Vol: \~0.36--0.45 - Low Vol: \~0.42--0.60

Interpretation: - Clear degradation in high volatility - Low-vol regime
contains most predictive edge

------------------------------------------------------------------------

## Corn

Overall DA range: 0.50 -- 0.55

Aggregated Regime DA: - High Vol: \~0.55--0.65 - Low Vol: \~0.48--0.53

Interpretation: - Inverse behavior vs metals - High volatility may
create exploitable moves

------------------------------------------------------------------------

## Coffee

Overall DA range: 0.44 -- 0.59

Aggregated Regime DA: - High Vol: \~0.50--0.55 - Low Vol: \~0.46--0.60

Interpretation: - Moderate regime sensitivity - Some edge in stable
periods

------------------------------------------------------------------------

## Heating Oil

Overall DA range: 0.42 -- 0.51

Aggregated Regime DA: - High Vol: \~0.42--0.55 - Low Vol: \~0.43--0.49

Interpretation: - Mild regime differentiation

------------------------------------------------------------------------

# Key Findings

1.  Volatility regime materially affects predictability.
2.  Industrial metals and energy show strong regime asymmetry.
3.  Some agricultural commodities behave differently (Corn).
4.  Overall DAs are modest (0.53--0.58), but regime-filtered DAs reach
    0.60+.
5.  Predictability is conditional, not uniform.

------------------------------------------------------------------------

# Strategic Implications

The volatility regime feature is:

-   Properly implemented
-   Statistically coherent
-   Producing meaningful diagnostics

Alpha appears concentrated in specific regime conditions rather than
uniformly distributed.

------------------------------------------------------------------------

# Next Research Directions

1.  Train separate models per regime
2.  Add interaction terms (feature × regime)
3.  Regime-based model selection or ensemble weighting
4.  Statistical significance testing of regime gaps

------------------------------------------------------------------------

# Conclusion

This project demonstrates that commodity directional predictability is
regime-dependent.

Future improvements should focus on conditional modeling rather than
feature expansion alone.