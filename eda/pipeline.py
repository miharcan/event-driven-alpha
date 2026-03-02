import logging
from pathlib import Path
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
import re
from sklearn.decomposition import PCA
from collections import defaultdict

from src.features.price_features import (
    compute_log_returns,
    compute_rolling_volatility,
    add_lag_features,
    add_forward_return,
    add_volatility_regime
)
from src.data.news_loader import load_news_json
from src.features.news_features import build_daily_news_features
from src.data.alignment import align_price_and_news
from src.models.registry import get_model_trainer
from src.features.embeddings import NewsEmbedder
from src.data.macro_loader import load_macro_features
from src.models.ridge import train_regime_specific
from src.models.baseline_regression import (
    train_baseline_regression,
    directional_accuracy,
    run_regime_gated_model
)


logger = logging.getLogger(__name__)


def run_pipeline(config: dict):

    logger.info("=== STARTING PIPELINE ===")

    # ---------------------------
    # 1. Configuration
    # ---------------------------

    macro_categories = [
        "POLITICS",
        "BUSINESS",
        "WORLD NEWS",
        "U.S. NEWS",
        "THE WORLDPOST",
        "WORLDPOST",
    ]

    macro_keywords = [
        # Monetary
        "inflation", "federal reserve", "fed", "interest rate",
        "rate hike", "rate cut", "yield", "bond", "real rate",
        # FX
        "dollar", "usd", "currency", "forex",
        # Energy
        "oil", "crude", "opec", "refinery", "pipeline",
        "energy", "natural gas",
        # Geopolitics
        "sanction", "russia", "china", "ukraine",
        "war", "military", "conflict", "trade",
        # Macro risk
        "recession", "debt", "default", "gdp",
        "supply", "shortage", "export", "import",
    ]

    pattern = re.compile("|".join(macro_keywords), re.IGNORECASE)

    asset_map = {
        "Gold": ("Date Gold", "Value Gold"),
        "Silver": ("Date Silver", "Value Silver"),
        "Platinum": ("Date Platinum", "Value Platinum"),
        "Copper": ("Date Copper", "Value Copper"),
        "Crude": ("Date Crude Oil", "Value Crude Oil"),
        "HeatingOil": ("Date Heating Oil", "Value Heating Oil"),
        "Corn": ("Date Corn", "Value Corn"),
        "Coffee": ("Date Coffee", "Value Coffee"),
    }

    trainer = get_model_trainer(config["model"]["model_type"], config)

    # ---------------------------
    # 2. Load Price File (Raw)
    # ---------------------------

    price_path = config["price_path"]

    if price_path.endswith(".csv"):
        df_raw = pd.read_csv(price_path)
    else:
        df_raw = pd.read_excel(price_path)

    # ---------------------------
    # 3. Load & Prepare News
    # ---------------------------

    logger.info("Loading news data...")

    df_news_raw = load_news_json(config["news_path"])

    # Category filter
    df_news_raw = df_news_raw[
        df_news_raw["category"].isin(macro_categories)
    ]

    logger.info(f"Headlines before keyword filter: {len(df_news_raw)}")

    # Keyword filter
    df_news_raw = df_news_raw[
        df_news_raw["headline"].str.contains(pattern, na=False)
    ]

    logger.info(f"Headlines after keyword filter: {len(df_news_raw)}")

    if len(df_news_raw) < 1000:
        logger.warning("Very few headlines after filtering.")

    # Structured news features
    df_news_structured = build_daily_news_features(df_news_raw)

    # Embeddings
    embedder = NewsEmbedder(
        model_name=config.get("embedding_model", "all-MiniLM-L6-v2")
    )

    headline_embeddings = embedder.embed_headlines(df_news_raw)
    daily_embeddings = embedder.aggregate_daily(
        df_news_raw,
        headline_embeddings,
    )

    daily_embeddings.columns = [
        f"emb_{i}" for i in range(daily_embeddings.shape[1])
    ]

    df_news = df_news_structured.join(
        daily_embeddings,
        how="left"
    ).fillna(0)

    # ---------------------------
    # 4. Load Macro Features
    # ---------------------------

    logger.info("Loading macro features...")
    df_macro = load_macro_features(config["macro_path"])

    # ---------------------------
    # 5. Multi-Asset Loop
    # ---------------------------
    tracker = ResultTracker()
    for asset, (date_col, price_col) in asset_map.items():

        logger.info(f"\n--- Running {asset} ---")

        if date_col not in df_raw.columns:
            logger.info(f"{asset}: Missing columns in raw file.")
            continue

        # --- Extract asset series
        df_price = df_raw[[date_col, price_col]].copy()
        df_price.columns = ["date", "price"]

        df_price["date"] = pd.to_datetime(df_price["date"], errors="coerce")
        df_price = df_price.dropna()
        df_price = df_price.set_index("date").sort_index()

        # --- Price features
        df_price = compute_log_returns(df_price)
        df_price = compute_rolling_volatility(df_price)
        df_price = add_lag_features(df_price, "log_return", 3)
        
        horizons = config.get("horizons", [5])
        for horizon in horizons:

            logger.info(f"\n===== Testing Horizon: {horizon} days =====")

            df_price_h = df_price.copy()

            df_price_h = add_forward_return(df_price_h, horizon=horizon)
            df_price_h = add_volatility_regime(df_price_h)


            # --- Merge macro (CRITICAL FIX)
            df_price_h = df_price_h.join(df_macro, how="left")
            df_price_h = df_price_h.ffill()

            asset_kw = config.get("asset_keywords", {}).get(asset, [])

            if asset_kw:
                pattern_asset = re.compile("|".join(asset_kw), re.IGNORECASE)

                df_asset_news_raw = df_news_raw[
                    df_news_raw["headline"].str.contains(pattern_asset, na=False)
                ].copy()

                logger.info(f"{asset}: Asset-specific headlines = {len(df_asset_news_raw)}")

            else:
                df_asset_news_raw = df_news_raw.copy()

            # Structured features
            df_asset_structured = build_daily_news_features(df_asset_news_raw)

            # Embeddings
            headline_emb_asset = embedder.embed_headlines(df_asset_news_raw)
            daily_emb_asset = embedder.aggregate_daily(
                df_asset_news_raw,
                headline_emb_asset,
            )

            daily_emb_asset.columns = [
                f"emb_{i}" for i in range(daily_emb_asset.shape[1])
            ]

            df_news_asset = df_asset_structured.join(
                daily_emb_asset,
                how="left"
            ).fillna(0)

            df_model_global = align_price_and_news(df_price_h, df_news)

            df_model_asset = align_price_and_news(df_price_h, df_news_asset)

            logger.info(f"{asset}: Global daily news rows = {len(df_news)}")
            logger.info(f"{asset}: Asset daily news rows = {len(df_news_asset)}")

            for name, df_tmp in [("global", df_model_global), ("asset", df_model_asset)]:
                df_tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
                df_tmp.dropna(inplace=True)

            for news_version, df_model in [
                ("GlobalNews", df_model_global),
                ("AssetNews", df_model_asset)
            ]:
                if len(df_model) < 200:
                    logger.info(f"{asset}: Not enough overlapping data.")
                    continue

                datasets = [("Full", df_model)]

                if "article_count" in df_model.columns:

                    percentiles = [0.5, 0.6, 0.7, 0.8, 0.9]

                    for q in percentiles:
                        threshold = df_model["article_count"].quantile(q)

                        df_filtered = df_model[df_model["article_count"] > threshold].copy()

                        if len(df_filtered) > 100:
                            label = f"Top{int(q*100)}"
                            datasets.append((label, df_filtered))

                # ---------------------------
                # Dataset loop: Full vs HighAttention
                # ---------------------------
                for dataset_name, df_use in datasets:

                    df_use = df_use.copy()

                    # ---------------------------
                    # Identify Embeddings
                    # ---------------------------
                    raw_embedding_cols = [c for c in df_use.columns if c.startswith("emb_")]
                    if not raw_embedding_cols:
                        logger.warning(f"{asset} | H{horizon} | {news_version} | {dataset_name}: No raw embedding cols.")
                        continue

                    # PCA -> emb_pca_0..9
                    pca = PCA(n_components=10)
                    emb_pca = pd.DataFrame(
                        pca.fit_transform(df_use[raw_embedding_cols]),
                        index=df_use.index,
                        columns=[f"emb_pca_{i}" for i in range(10)]
                    )
                    df_use = df_use.join(emb_pca)

                    pca_cols = [f"emb_pca_{i}" for i in range(10)]
                    df_use["emb_mean"] = df_use[pca_cols].mean(axis=1)

                    # ---------------------------
                    # Identify Macro Features (from df_macro columns)
                    # ---------------------------
                    macro_cols = [c for c in df_use.columns if c in df_macro.columns]
                    

                    # ---------------------------
                    # Interaction Terms (emb_mean x macro)
                    # ---------------------------
                    interaction_cols = []
                    for macro in macro_cols:
                        col_name = f"emb_mean_x_{macro}"
                        df_use[col_name] = df_use["emb_mean"] * df_use[macro]
                        interaction_cols.append(col_name)

                    # ---------------------------
                    # Feature Blocks
                    # ---------------------------
                    price_cols = config["price_features"]

                    news_features = pca_cols
                    macro_features = macro_cols
                    all_features = price_cols + macro_cols + pca_cols + interaction_cols

                    # ---------------------------
                    # Experiments
                    # ---------------------------
                    results = {}

                    if price_cols:
                        res_price = normalize_result(trainer(df_use, feature_cols=price_cols))
                        tracker.add(asset, horizon, f"{dataset_name}_Price", res_price["mean_da"], res_price["fold_das"])
                        results["Price"] = res_price

                    if macro_features:
                        res_macro = normalize_result(trainer(df_use, feature_cols=macro_features))
                        tracker.add(asset, horizon, f"{dataset_name}_Macro", res_macro["mean_da"], res_macro["fold_das"])
                        results["Macro"] = res_macro

                    if news_features:
                        res_news = normalize_result(trainer(df_use, feature_cols=news_features))
                        tracker.add(asset, horizon, f"{dataset_name}_News", res_news["mean_da"], res_news["fold_das"])
                        results["News"] = res_news

                    if price_cols and macro_features:
                        res_price_macro = normalize_result(trainer(df_use, feature_cols=price_cols + macro_features))
                        tracker.add(asset, horizon, f"{dataset_name}_Price+Macro", res_price_macro["mean_da"], res_price_macro["fold_das"])
                        results["Price+Macro"] = res_price_macro

                    if price_cols and news_features:
                        res_price_news = normalize_result(trainer(df_use, feature_cols=price_cols + news_features))
                        tracker.add(asset, horizon, f"{dataset_name}_Price+News", res_price_news["mean_da"], res_price_news["fold_das"])
                        results["Price+News"] = res_price_news

                    if macro_features and news_features:
                        res_macro_news = normalize_result(trainer(df_use, feature_cols=macro_features + news_features))
                        tracker.add(asset, horizon, f"{dataset_name}_Macro+News", res_macro_news["mean_da"], res_macro_news["fold_das"])
                        results["Macro+News"] = res_macro_news

                    if all_features:
                        res_all = normalize_result(trainer(df_use, feature_cols=all_features))
                        tracker.add(asset, horizon, f"{dataset_name}_All", res_all["mean_da"], res_all["fold_das"])
                        results["All"] = res_all

                    # Regime-specific (if your function returns mean_da/fold_das; if not, wrap with normalize_result or adjust)
                    if "vol_regime_high" in df_use.columns and all_features:
                        res_regime = train_regime_specific(
                            df_use,
                            config,
                            feature_cols=all_features + ["vol_regime_high"],
                            alpha=config.get("ridge_alpha", 1.0)
                        )
                        # If train_regime_specific doesn't return mean_da, normalize it or fix the function return
                        res_regime = normalize_result(res_regime)
                        tracker.add(asset, horizon, f"{dataset_name}_RegimeSpecific", res_regime["mean_da"], res_regime["fold_das"])

                    # All + RegimeInteraction
                    if all_features:
                        res_all_inter = normalize_result(run_all_with_regime_interaction(
                            df=df_use,
                            base_feature_cols=all_features,
                            config=config
                        ))
                        tracker.add(asset, horizon, f"{dataset_name}_All+RegimeInteraction", res_all_inter["mean_da"], res_all_inter["fold_das"])

                        if "All" in results:
                            delta = res_all_inter["mean_da"] - results["All"]["mean_da"]
                            logger.info(f"{asset} | H{horizon} | {news_version} | {dataset_name} | RegimeInteraction Δ vs All: {delta:+.4f}")

                    # Regime gated
                    if all_features:
                        res_gated = normalize_result(run_regime_gated_model(
                            df=df_use,
                            target_col="fwd_return",
                            feature_cols=all_features,
                            config=config,
                        ))
                        tracker.add(asset, horizon, f"{dataset_name}_All+RegimeGated", res_gated["mean_da"], res_gated["fold_das"])

                    # Best for this asset+horizon across everything recorded so far
                    best_row = tracker.best_for(asset, horizon)
                    logger.info(f"{asset} | H{horizon} | {news_version} | {dataset_name} | Best={best_row['model']} ({best_row['mean_da']:.4f})")
           
    model_type = config["model"]["model_type"]

    full_path = f"outputs/results_{model_type}_full.csv"
    summary_path = f"outputs/results_{model_type}_summary.csv"

    tracker.save(full_path)
    tracker.save_summary(summary_path)

    logger.info(f"Results saved to {full_path} and {summary_path}")


# ---------------------------
# Logging Results
# ---------------------------

def log_result(name, res, asset):
    da = directional_accuracy(
        res["y_test"],
        res["predictions"]
    )
    logger.info(f"{asset} {name} DA: {da:.4f}")

    if "fold_das" in res:
        logger.info(f"{asset} {name} Fold DAs: {res['fold_das']}")
        logger.info(
            f"{asset} {name} Mean Fold DA: "
            f"{res['mean_da']:.4f}"
        )


def collect_metrics(res, experiment_name, feature_set):
    return {
        "experiment_name": experiment_name,
        "feature_set": feature_set,
        "metrics": {
            "mse": res["mse"],
            "r2": res["r2"],
            "directional_accuracy": directional_accuracy(
                res["y_test"], res["predictions"]
            ),
        },
    }


def persist_outputs(results, config):

    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_version": config.get("model_version"),
        "model_type": config.get("model_type"),
        "timestamp": datetime.utcnow().isoformat(),
        "experiments": results,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(artifact, f, indent=4)

    return output_dir


def run_regime_specific_model(
    X_train, y_train, regimes_train,
    X_test, y_test, regimes_test,
    model_class
):
    """
    Trains separate models for high-vol and low-vol regimes.
    Returns:
        high_da, low_da
    """

    # Split training data
    X_train_high = X_train[regimes_train == 1]
    y_train_high = y_train[regimes_train == 1]

    X_train_low = X_train[regimes_train == 0]
    y_train_low = y_train[regimes_train == 0]

    # Initialize models
    model_high = model_class()
    model_low = model_class()

    high_da = None
    low_da = None

    # Train + test high regime
    if len(X_train_high) > 5:
        model_high.fit(X_train_high, y_train_high)
        X_test_high = X_test[regimes_test == 1]
        y_test_high = y_test[regimes_test == 1]

        if len(X_test_high) > 0:
            preds_high = model_high.predict(X_test_high)
            high_da = (preds_high == y_test_high).mean()

    # Train + test low regime
    if len(X_train_low) > 5:
        model_low.fit(X_train_low, y_train_low)
        X_test_low = X_test[regimes_test == 0]
        y_test_low = y_test[regimes_test == 0]

        if len(X_test_low) > 0:
            preds_low = model_low.predict(X_test_low)
            low_da = (preds_low == y_test_low).mean()

    return high_da, low_da


def run_all_with_regime_interaction(df, base_feature_cols, config):
    """
    Unified model with continuous volatility interaction.
    Computes volatility proxy from log returns to avoid missing column issues.
    """

    df_int = df.copy()

    if "log_return" not in df_int.columns:
        raise ValueError("log_return column required for volatility interaction.")

    # --- compute rolling volatility locally (safe)
    vol = df_int["log_return"].rolling(10).std()

    vol = vol.bfill()

    # normalize
    vol_z = (vol - vol.mean()) / (vol.std() + 1e-8)

    interaction_cols = []

    for col in base_feature_cols:
        if col != "vol_regime_high":
            new_col = f"{col}_x_vol"
            df_int[new_col] = df_int[col] * vol_z
            interaction_cols.append(new_col)

    feature_cols_extended = base_feature_cols + interaction_cols

    return train_baseline_regression(
        df=df_int,
        target_col="fwd_return",
        feature_cols=feature_cols_extended,
        config=config
    )



class ResultTracker:
    def __init__(self):
        self.records = []

    def add(
        self,
        asset,
        horizon,
        model_name,
        mean_da,
        fold_das=None,
        regime_variant=None,
    ):
        self.records.append({
            "asset": asset,
            "horizon": horizon,
            "model": model_name,
            "mean_da": float(mean_da),
            "fold_das": fold_das,
            "regime_variant": regime_variant,
        })

    def best_for(self, asset, horizon):
        df = pd.DataFrame(self.records)
        df = df[(df["asset"] == asset) & (df["horizon"] == horizon)]
        return df.loc[df["mean_da"].idxmax()]

    def summary_by_asset(self):
        df = pd.DataFrame(self.records)

        summary_rows = []

        for asset in df["asset"].unique():
            df_asset = df[df["asset"] == asset]

            for horizon in sorted(df_asset["horizon"].unique()):
                df_h = df_asset[df_asset["horizon"] == horizon]

                best_row = df_h.loc[df_h["mean_da"].idxmax()]

                summary_rows.append({
                    "asset": asset,
                    "horizon": horizon,
                    "best_model": best_row["model"],
                    "best_da": best_row["mean_da"],
                })

        return pd.DataFrame(summary_rows)

    def save(self, path="results_full.csv"):
        df = pd.DataFrame(self.records)
        df.to_csv(path, index=False)

    def save_summary(self, path="results_summary.csv"):
        summary_df = self.summary_by_asset()
        summary_df.to_csv(path, index=False)



def normalize_result(res):
    if "mean_da" in res:
        return res
    elif "mean_fold_da" in res:
        res["mean_da"] = res["mean_fold_da"]
        return res
    else:
        raise ValueError("Trainer result missing mean_da / mean_fold_da")