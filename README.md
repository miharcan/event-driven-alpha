# Event-Driven Alpha

## A Framework for Semantic News Signal Extraction

------------------------------------------------------------------------

## Summary

This project evaluates whether structured and semantic news signals
contain actionable information for forecasting next-day asset returns
under strict out-of-sample validation.

The framework is designed with institutional research standards in mind:

-   Chronological walk-forward validation\
-   Explicit leakage controls\
-   Fold-level performance transparency\
-   Controlled model comparison

The objective is not to maximize backtest performance, but to determine
whether news-derived signals exhibit stable, regime-aware predictive
behavior.

------------------------------------------------------------------------

## Challenge

Can daily macro-news information improve directional forecasting of gold
returns beyond autoregressive price structure?

Specifically:

-   Do simple category counts add value?
-   Do semantic embeddings capture latent macro regime shifts?
-   Is the signal persistent or regime-dependent?

------------------------------------------------------------------------

## Data & Feature Architecture

### Price Features

-   Log returns\
-   21-day rolling volatility\
-   Autoregressive lags

### Structured News Features

-   Daily article count\
-   Category frequency aggregation

### Semantic News Features

-   Sentence-transformer headline embeddings (`all-MiniLM-L6-v2`)
-   384-dimensional representation per headline\
-   Daily mean embedding aggregation\
-   PCA dimensionality reduction applied *inside training folds only*

All transformations are strictly time-aligned.

------------------------------------------------------------------------

## Validation Framework

Evaluation uses expanding-window walk-forward validation:

-   Initial 60% training period\
-   Sequential forward testing folds\
-   PCA fit exclusively on training data\
-   No future information leakage

Metrics reported:

-   Directional Accuracy (DA)
-   MSE
-   R²
-   Fold-level DA stability

This framework mirrors a live deployment retraining cycle.

------------------------------------------------------------------------

## Empirical Findings (Leakage-Safe Walk-Forward)

  Model                Directional Accuracy
  -------------------- ----------------------
  Price-only           \~0.48
  Structured-only      \~0.52
  Embeddings-only      \~0.52
  Price + Embeddings   \~0.52

### Observations

-   Price-only models underperform random threshold.
-   Structured news categories provide modest lift.
-   Semantic embeddings show **regime-dependent performance**.
-   Average predictive edge remains modest (\~52% DA).
-   Stronger performance appears concentrated in later macro-stress
    regimes.

------------------------------------------------------------------------

## Interpretation for Portfolio Context

The findings suggest:

1.  Daily semantic signals are not universally predictive.
2.  Predictive power increases during periods of elevated macro
    instability.
3.  Embeddings may function as a **macro regime detector** rather than a
    stable daily alpha source.
4.  Strict leakage controls materially reduce overstated backtest
    performance.

From a portfolio perspective:

-   This is not yet a standalone alpha strategy.
-   It may serve as a conditional overlay.
-   It may improve risk allocation during macro stress periods.

------------------------------------------------------------------------

## Potential Applications

-   Volatility-conditioned signal activation\
-   Macro stress regime classification\
-   Risk budget adjustment overlay\
-   Multi-asset validation (commodities, FX, indices)\
-   Ensemble integration with macro models

------------------------------------------------------------------------

## Project Status

Current stage:\
Leakage-controlled, walk-forward validated research prototype.

Next steps:

-   Regime segmentation analysis\
-   Multi-asset cross-validation\
-   Rolling Sharpe analysis\
-   Statistical significance testing

------------------------------------------------------------------------

## Disclaimer

This material is for informational purposes only.\
It does not constitute investment advice or a recommendation to trade.