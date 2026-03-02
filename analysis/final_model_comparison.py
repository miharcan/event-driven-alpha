import pandas as pd

ridge = pd.read_csv("outputs/results_ridge_summary.csv")
xgb = pd.read_csv("outputs/results_xgboost_summary.csv")
lgb = pd.read_csv("outputs/results_lightgbm_summary.csv")

ridge["model_family"] = "Ridge"
xgb["model_family"] = "XGBoost"
lgb["model_family"] = "LightGBM"

df = pd.concat([ridge, xgb, lgb], ignore_index=True)

print("Columns:", df.columns)

# Best model per asset (across families)
best_overall = (
    df.loc[df.groupby("asset")["best_da"].idxmax()]
    .sort_values("best_da", ascending=False)
    .reset_index(drop=True)
)

print("\n=== FINAL BEST MODEL PER ASSET ===")
print(best_overall[[
    "asset",
    "model_family",
    "horizon",
    "best_model",
    "best_da"
]])

best_overall.to_csv("outputs/final_model_comparison.csv", index=False)

print("\nSaved final_model_comparison.csv")