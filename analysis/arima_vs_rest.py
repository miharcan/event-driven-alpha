import pandas as pd

# ----------------------------
# Load data
# ----------------------------

df_ml = pd.read_csv("outputs/final_model_comparison.csv")
df_arima = pd.read_csv("outputs/results_arima_full.csv")

# ----------------------------
# Clean ARIMA table
# ----------------------------

df_arima_clean = (
    df_arima[df_arima["model"] == "ARIMA_Full"]
    .groupby(["asset", "horizon"], as_index=False)["mean_da"]
    .mean()
    .rename(columns={"mean_da": "arima_da"})
)

# ----------------------------
# Merge ML best with ARIMA
# ----------------------------

df_compare = df_ml.merge(
    df_arima_clean,
    on=["asset", "horizon"],
    how="left"
)

# ----------------------------
# Compute uplift
# ----------------------------

df_compare["delta_abs"] = df_compare["best_da"] - df_compare["arima_da"]
df_compare["delta_pct"] = (
    df_compare["delta_abs"] / df_compare["arima_da"] * 100
)

# ----------------------------
# Sort by absolute uplift
# ----------------------------

df_compare = df_compare.sort_values(
    "delta_abs",
    ascending=False
)

print("\n=== ARIMA vs ML UPLIFT ===")
print(df_compare[[
    "asset",
    "horizon",
    "model_family",
    "best_model",
    "best_da",
    "arima_da",
    "delta_abs",
    "delta_pct"
]])

# Optional: save
df_compare.to_csv("outputs/arima_vs_ml_comparison.csv", index=False)