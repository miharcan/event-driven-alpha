# Event-Driven Alpha: Modeling Framework

## Overview

This repository implements a research-grade modeling framework for
evaluating event-driven directional prediction in commodity markets
(metals, energy, agricultural products).

The system integrates:

-   Price-based technical features\
-   Macro features\
-   News embedding features\
-   Event intensity conditioning\
-   Volatility regime modeling\
-   Walk-forward cross-validation\
-   Linear and tree-based model families

The goal is to evaluate whether predictive signal is:

-   Linear vs nonlinear\
-   Regime-dependent\
-   Event-intensity dependent\
-   Horizon-sensitive

------------------------------------------------------------------------

## Architecture

### Data Inputs

-   Historical price data
-   Macro time series data
-   News headline dataset (embedded using transformer-based sentence
    embeddings)

### Feature Engineering

-   Log returns and lagged returns
-   Rolling volatility measures
-   Volatility regime classification
-   News embeddings (dimension-reduced)
-   Event intensity filtering (percentile-based)
-   Optional interaction features

All feature engineering is performed prior to model training to prevent
leakage.

------------------------------------------------------------------------

## Modeling Families

The framework supports multiple model types via configuration:

### 1. Ridge Regression (Linear Baseline)

-   Expanding-window walk-forward CV
-   Directional prediction via regression sign
-   Serves as linear benchmark

### 2. XGBoost (Tree-Based)

-   Binary classification objective
-   Captures nonlinear interactions
-   Expanding-window CV

### 3. LightGBM (Tree-Based)

-   Gradient boosting framework
-   Fast and efficient tree construction
-   Suppressed training verbosity for clean experimentation

All models share:

-   Identical CV geometry\
-   Identical target construction\
-   Identical evaluation metric

This ensures fair cross-model comparison.

------------------------------------------------------------------------

## Evaluation Methodology

### Walk-Forward Cross-Validation

-   Expanding training window
-   Strict temporal ordering
-   No look-ahead bias

### Primary Metric

Directional Accuracy (DA):

Predicted direction vs realized forward return sign.

Results are stored as:

-   Full per-configuration results
-   Per-model summary results
-   Final cross-model comparison table

------------------------------------------------------------------------

## Event-Intensity Gradient

The framework evaluates performance across percentile-based event
intensity buckets:

-   Full sample
-   Top percentiles (e.g., Top 50, 60, 70, 80, 90)

This allows testing whether predictive power concentrates during
high-information regimes.

------------------------------------------------------------------------

## Regime Modeling

Optional volatility regime interaction modeling is supported:

-   Regime feature interaction
-   Regime-specific training
-   Conditional performance evaluation

------------------------------------------------------------------------

## Output Structure

Each model run produces:

-   results\_`<model>`\_full.csv\
-   results\_`<model>`\_summary.csv

A final comparison script merges model families into:

-   final_model_comparison.csv

------------------------------------------------------------------------

## Key Questions

1.  Is predictive structure linear or nonlinear?
2.  Does signal concentrate in high-intensity event regimes?
3.  Does volatility regime alter predictive strength?
4.  Are improvements consistent across forecast horizons?

------------------------------------------------------------------------

## Key Findings

1.  Event intensity unlocks the signal.
Full-sample models average ~52–58% directional accuracy. Conditioning on high-information regimes (Top 60–90%) pushes performance to 65–78% DA across metals, energy, and broader commodities.

2.  Filtering beats complexity.
Moving to high-intensity subsets delivers larger gains than switching model families. Signal concentration drives performance more than algorithm sophistication.

3.  Nonlinearity matters — but only where it should.
Tree-based models add +3–8% DA in short-horizon, high-intensity environments. Longer horizons remain largely linear and macro-driven.

3.  Horizon changes the game.
Short-term forecasts (3–10 periods) show sharper nonlinear effects (~68–72% DA). Longer horizons (≈20 periods) reach 70–79% DA primarily via structured linear drift.

4.  Regime-aware modeling improves selectively.
Volatility interaction terms produce +2–9% DA improvements in targeted contexts, while hard regime segmentation reduces statistical stability.

------------------------------------------------------------------------

## Final Best Model per Asset

| Asset          | Model Family | Horizon | Best Model                           | Directional Accuracy |
|----------------|--------------|----------|-------------------------------------|----------------------|
| Energy_1       | Ridge        | 20       | Top60_Macro                         | 0.7867 |
| Metal_1        | Ridge        | 20       | Top90_All+RegimeInteraction         | 0.7778 |
| Metal_2        | LightGBM     | 5        | Top90_News                          | 0.7111 |
| Metal_3        | LightGBM     | 10       | Top80_Price                         | 0.7000 |
| Metal_4        | Ridge        | 10       | Top80_RegimeSpecific                | 0.7000 |
| Energy_2       | LightGBM     | 3        | Top90_News                          | 0.6889 |
| Commodities_1  | LightGBM     | 10       | Top90_Price                         | 0.6667 |
| Commodities_2  | Ridge        | 20       | Top50_Macro                         | 0.6520 |


-----------------------------------------------------------

## How To Run

Run with a specific configuration:

``` bash
python -m eda.cli --config configs/<model_config>.yaml
```

Available model types:

-   ridge
-   xgboost
-   lightgbm

After running all model families:

``` bash
python analysis/final_model_comparison.py
```

------------------------------------------------------------------------

## Conclusion

This repository provides a clean experimental framework for testing
whether directional signals in commodity markets are primarily:

-   Linear
-   Nonlinear
-   Conditional on event intensity
-   Regime-dependent

It is structured for clarity, reproducibility, and extensibility.