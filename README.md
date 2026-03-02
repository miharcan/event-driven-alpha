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

## Key Research Questions

1.  Is predictive structure linear or nonlinear?
2.  Does signal concentrate in high-intensity event regimes?
3.  Does volatility regime alter predictive strength?
4.  Are improvements consistent across forecast horizons?

------------------------------------------------------------------------

## Design Principles

-   No data leakage
-   Reproducible configuration-based modeling
-   Model-family modularity
-   Consistent evaluation geometry
-   Transparent result tracking

------------------------------------------------------------------------

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