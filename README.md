# Event-Driven Alpha

A research framework for modeling asset price dynamics using event-driven signals and time-series learning.

---

## 🎯 Objective

This project explores whether structured news events contain predictive information for next-day asset returns.

The workflow implements:

- Clean data ingestion
- Time-series feature engineering
- Event aggregation
- Leakage-free alignment
- Baseline regression modeling
- Controlled model comparison

---

## 📂 Project Structure

```bash
event-driven-alpha/
├── src/
│ ├── data/ # Data loading and alignment
│ ├── features/ # Feature engineering
│ ├── models/ # Baseline models
├── configs/
└── data/ # Ignored (raw datasets)
```

---

## 🔬 Methodology

### 1️⃣ Price Features
- Log returns
- Rolling volatility (21-day)
- Autoregressive lags

### 2️⃣ News Features
- Daily article count
- Daily category frequency matrix
- Aggregated to trading-day resolution

### 3️⃣ Alignment
- Inner join on overlapping dates
- Target defined as next-day log return
- Strict avoidance of look-ahead bias

### 4️⃣ Modeling
Linear regression baseline:

- Price-only model
- News-only model
- Combined model

---

## 📊 Current Findings

Using daily data (2012–2022):

- Price-only model: ~53–54% directional accuracy
- News-only model: ~52%
- Combined model: ~52%

Raw category counts do not add incremental predictive power beyond autoregressive price structure.

This highlights:
- The difficulty of daily return prediction
- The importance of feature quality
- The risk of high-dimensional noise

---

## 🚀 Next Research Directions

- Regularized regression (Ridge/Lasso)
- Dimensionality reduction (PCA)
- Sentiment-based features
- Walk-forward validation
- Statistical significance testing

---

## ⚠️ Disclaimer

This project is for research and educational purposes only.
No trading or investment advice is implied.