# Event-Driven Alpha: Regime-Aware Multi-Modal Forecasting

## Overview

This repository implements an event-driven forecasting framework
combining:

-   Time-series price features
-   Macroeconomic indicators
-   NLP-derived news embeddings
-   Volatility regime detection
-   Walk-forward validation

The system evaluates directional forecasting performance across multiple
assets and feature configurations.

The goal is exploratory research into whether structured news and macro
information improve short-horizon financial forecasting under regime
shifts.

------------------------------------------------------------------------

## Selected Experimental Highlights

Across multiple assets and configurations, the framework demonstrates
consistent directional performance above random baseline (50%) in
several cases:

-   Directional Accuracy (DA) reaching **\~0.58--0.59** using price-only
    models
-   Multi-modal combinations (Price + Macro) achieving **\~0.57 DA**
-   Macro-only models reaching **\~0.57 DA** in certain assets
-   News-enhanced models achieving **\~0.55 DA** in selected
    configurations
-   Regime-conditioned performance showing improved stability in
    low-volatility environments

These results indicate that structured feature integration and regime
awareness can produce meaningful predictive signal under strict
walk-forward validation.

------------------------------------------------------------------------

## Data Inputs

The pipeline expects the following data types:

### 1. Price Data

-   Daily OHLC or returns
-   Rolling features (e.g., lagged returns)
-   Volatility estimation inputs

### 2. Macroeconomic Features

-   Interest rates
-   Inflation metrics
-   Yield spreads
-   Other macro indicators aligned by date

### 3. News Data

-   Timestamped headlines
-   Optional asset tagging
-   Text embeddings generated using sentence-transformer models
-   Daily aggregation of embeddings
-   Article count features

------------------------------------------------------------------------

## Feature Engineering

-   Rolling return windows
-   Volatility regime classification (high/low via rolling std)
-   News embedding PCA compression
-   High-attention filtering (based on article volume)

------------------------------------------------------------------------

## Models

-   Linear Regression
-   Ridge Regression
-   Walk-forward time-series cross-validation
-   Regime-specific training (separate models per volatility state)

------------------------------------------------------------------------

## Evaluation Metrics

-   Directional Accuracy (primary metric)
-   Fold-level DA
-   Aggregated regime DA
-   MSE and R²

All evaluation is performed using strict chronological splits (no
leakage).

------------------------------------------------------------------------

## Questions Explored

-   Do macro features improve directional forecasts?
-   Does structured news embedding add incremental signal?
-   Does asset-specific news outperform global news?
-   Does volatility regime separation improve predictive performance?
-   Are multi-modal models more robust than single-source inputs?

------------------------------------------------------------------------

## Running the Pipeline

``` bash
python -m eda.cli --config configs/default.yaml
```

------------------------------------------------------------------------

## Experimental Design

-   Multi-asset evaluation
-   Walk-forward folds
-   Parallel unified vs regime-specific models
-   Embedding compression via PCA
-   High-attention news subset analysis

------------------------------------------------------------------------

## Notes

This repository is designed for exploratory quantitative research and
methodological investigation.

It is not intended for live trading deployment.