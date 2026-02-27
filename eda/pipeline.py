import logging
from pathlib import Path
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
import re
from sklearn.decomposition import PCA

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
from src.models.baseline_regression import directional_accuracy
from src.features.embeddings import NewsEmbedder
from src.data.macro_loader import load_macro_features


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

    trainer = get_model_trainer(config["model_type"], config)

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
        df_price = add_forward_return(df_price, horizon=5)
        df_price = add_volatility_regime(df_price)

        # --- Merge macro (CRITICAL FIX)
        df_price = df_price.join(df_macro, how="left")
        df_price = df_price.ffill()

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

        # --- Align with news
        # df_model = align_price_and_news(df_price, df_news)
        # --- Align with GLOBAL news (existing behaviour)
        df_model_global = align_price_and_news(df_price, df_news)

        # --- Align with ASSET-SPECIFIC news (new branch)
        # df_model_asset  = align_price_and_news(df_price, df_asset_news)
        # df_model_asset  = align_price_and_news(df_price, df_news_asset)
        df_model_asset  = align_price_and_news(df_price, df_news_asset)


        # df_model = df_model.replace([np.inf, -np.inf], np.nan)
        # df_model = df_model.dropna()

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

            # ---- High news intensity regime
            if "article_count" in df_model.columns:
                threshold = df_model["article_count"].quantile(0.75)
                df_model = df_model[df_model["article_count"] > threshold]

            # ---------------------------
            # Identify Embeddings
            # ---------------------------

            embedding_cols = [
                c for c in df_model.columns if c.startswith("emb_")
            ]

            if not embedding_cols:
                logger.warning(f"{asset}: No embedding columns found.")
                continue

            # ---- PCA compression of embeddings
            pca = PCA(n_components=10)

            emb_pca = pd.DataFrame(
                pca.fit_transform(df_model[embedding_cols]),
                index=df_model.index,
                columns=[f"emb_pca_{i}" for i in range(10)]
            )

            df_model = df_model.join(emb_pca)

            embedding_cols = [f"emb_pca_{i}" for i in range(10)]

            # Create aggregate embedding signal
            df_model["emb_mean"] = df_model[embedding_cols].mean(axis=1)

            # ---------------------------
            # Identify Macro Features
            # ---------------------------

            macro_cols = [
                c for c in df_model.columns
                if c in df_macro.columns
            ]

            # ---------------------------
            # Add Interaction Terms
            # ---------------------------

            interaction_cols = []

            for macro in macro_cols:
                col_name = f"emb_mean_x_{macro}"
                df_model[col_name] = df_model["emb_mean"] * df_model[macro]
                interaction_cols.append(col_name)

            # ---------------------------
            # Feature Blocks
            # ---------------------------

            price_cols = config["price_features"]

            news_features = embedding_cols
            macro_features = macro_cols
            all_features = (
                price_cols
                + macro_cols
                + embedding_cols
                + interaction_cols
            )

            # ---------------------------
            # Experiments
            # ---------------------------

            results = {}

            # --- Price only
            if price_cols:
                res_price = trainer(df_model, feature_cols=price_cols)
                # log_result("Price_5day", res_price, asset)
                log_result(f"{news_version}_Price_5day", res_price, asset)
                results["Price_5day"] = res_price

            # --- Macro only (already computed above but ensure consistent block)
            if macro_features:
                res_macro = trainer(df_model, feature_cols=macro_features)
                log_result("Macro", res_macro, asset)
                results["Macro"] = res_macro

            # --- News only
            if news_features:
                res_news = trainer(df_model, feature_cols=news_features)
                log_result("News_5day_HighAttention", res_news, asset)
                results["News_5day_HighAttention"] = res_news

            # --- Price + Macro
            if price_cols and macro_features:
                res_price_macro = trainer(
                    df_model,
                    feature_cols=price_cols + macro_features
                )
                log_result("Price+Macro", res_price_macro, asset)
                results["Price+Macro"] = res_price_macro

            # --- Price + News
            if price_cols and news_features:
                res_price_news = trainer(
                    df_model,
                    feature_cols=price_cols + news_features
                )
                log_result("Price+News", res_price_news, asset)
                results["Price+News"] = res_price_news

            # --- Macro + News
            if macro_features and news_features:
                res_macro_news = trainer(
                    df_model,
                    feature_cols=macro_features + news_features
                )
                log_result("Macro+News", res_macro_news, asset)
                results["Macro+News"] = res_macro_news

            # --- All features (including interactions)
            if all_features:
                res_all = trainer(
                    df_model,
                    feature_cols=all_features
                )
                log_result("All", res_all, asset)
                results["All"] = res_all

    logger.info("=== PIPELINE COMPLETE ===")


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
            f"{res['mean_fold_da']:.4f}"
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

