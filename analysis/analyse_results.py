import pandas as pd

df = pd.read_csv("outputs/results_full.csv")

# ----------------------------
# 1️⃣ Best horizon per asset
# ----------------------------
best = (
    df.groupby(["asset", "horizon"])
      .mean(numeric_only=True)
      .reset_index()
)

best_overall = (
    best.loc[best.groupby("asset")["mean_da"].idxmax()]
      .sort_values("mean_da", ascending=False)
)

print("\n=== BEST HORIZON PER ASSET ===")
print(best_overall[["asset", "horizon", "mean_da"]])


# ----------------------------
# 2️⃣ EVENT INTENSITY GRADIENT
# ----------------------------
# df["is_high"] = df["model"].str.contains("HighAttention")
df["attention_bucket"] = df["model"].str.extract(r"(Full|Top\d+)")
gradient = (
    df.groupby(["asset", "horizon", "attention_bucket"])["mean_da"]
      .max()
      .unstack()
)

print("\n=== EVENT INTENSITY GRADIENT ===")
print(gradient)

# ----------------------------
# 3️⃣ Regime Interaction uplift
# ----------------------------
df["is_regime_inter"] = df["model"].str.contains("RegimeInteraction")

regime_uplift = (
    df.groupby(["asset", "horizon", "is_regime_inter"])["mean_da"]
      .max()
      .unstack()
)

regime_uplift["delta_regime"] = regime_uplift[True] - regime_uplift[False]

print("\n=== REGIME INTERACTION UPLIFT ===")
print(regime_uplift.sort_values("delta_regime", ascending=False))