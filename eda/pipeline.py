import logging
from pathlib import Path
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

# from src.data.loader import load_auronum_series
from src.features.price_features import (
    compute_log_returns,
    compute_rolling_volatility,
    add_lag_features,
)
from src.data.news_loader import load_news_json
from src.features.news_features import build_daily_news_features
from src.data.alignment import align_price_and_news
from src.models.registry import get_model_trainer
from src.models.baseline_regression import directional_accuracy
from src.features.embeddings import NewsEmbedder


logger = logging.getLogger(__name__)


def run_pipeline(config: dict):

    macro_categories = [
        "POLITICS",
        "BUSINESS",
        "WORLD NEWS",
        "U.S. NEWS",
        "THE WORLDPOST",
        "WORLDPOST",
        "MONEY",
        "CRIME",
        "ENVIRONMENT",
    ]

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
    
    price_path = config["price_path"]

    if price_path.endswith(".csv"):
        df_raw = pd.read_csv(price_path)
    else:
        df_raw = pd.read_excel(price_path)

    # ----- NEWS -----
    df_news_raw = load_news_json(config["news_path"])
    df_news_raw = df_news_raw[
        df_news_raw["category"].isin(macro_categories)
    ]

    df_news_structured = build_daily_news_features(df_news_raw)

    embedder = NewsEmbedder(
        model_name=config.get("embedding_model", "all-MiniLM-L6-v2")
    )

    headline_embeddings = embedder.embed_headlines(df_news_raw)
    daily_embeddings = embedder.aggregate_daily(
        df_news_raw,
        headline_embeddings
    )

    daily_embeddings.columns = [
        f"emb_{i}" for i in range(daily_embeddings.shape[1])
    ]

    df_news = df_news_structured.join(
        daily_embeddings,
        how="left"
    ).fillna(0)

    trainer = get_model_trainer(config["model_type"], config)

    # ----- MULTI-ASSET LOOP -----
    for asset, (date_col, price_col) in asset_map.items():

        logger.info(f"--- Running {asset} ---")

        if date_col not in df_raw.columns:
            logger.info(f"{asset}: Missing columns.")
            continue

        df_price = df_raw[[date_col, price_col]].copy()
        df_price.columns = ["date", "price"]

        df_price["date"] = pd.to_datetime(df_price["date"], errors="coerce")
        df_price = df_price.dropna()
        df_price = df_price.set_index("date").sort_index()

        df_price = compute_log_returns(df_price)
        df_price = compute_rolling_volatility(df_price)
        df_price = add_lag_features(df_price, "log_return", 3)

        df_model = align_price_and_news(df_price, df_news)
        df_model = df_model.dropna()

        if len(df_model) < 200:
            logger.info(f"{asset}: Not enough overlapping data.")
            continue

        embedding_cols = [
            c for c in df_model.columns if c.startswith("emb_")
        ]

        res = trainer(df_model, feature_cols=embedding_cols)

        da = directional_accuracy(
            res["y_test"],
            res["predictions"]
        )

        logger.info(f"{asset} DA: {da:.4f}")

        if "fold_das" in res:
            logger.info(f"{asset} Fold DAs: {res['fold_das']}")
            logger.info(f"{asset} Mean Fold DA: {res['mean_fold_da']:.4f}")


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

