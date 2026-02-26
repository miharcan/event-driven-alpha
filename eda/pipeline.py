import logging
from pathlib import Path
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.data.loader import load_auronum_series
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

    logger.info("Loading price data...")
    df_price = load_auronum_series(
        config["price_path"],
        date_col=config["price_date_col"],
        price_col=config["price_value_col"],
    )

    df_price = compute_log_returns(df_price)
    df_price = compute_rolling_volatility(df_price)
    df_price = add_lag_features(df_price, "log_return", 3)

    logger.info("Loading news data...")
    
    # df_news_raw = load_news_json(config["news_path"])
    # df_news = build_daily_news_features(df_news_raw)

    df_news_raw = load_news_json(config["news_path"])
    # print("NEWS COLUMNS:", df_news_raw.columns.tolist())
    # exit()

    # ---- Structured category features (existing)
    df_news_structured = build_daily_news_features(df_news_raw)

    embedder = NewsEmbedder(
        model_name=config.get("embedding_model", "all-MiniLM-L6-v2")
    )

    headline_embeddings = embedder.embed_headlines(
        df_news_raw,
        text_col=config.get("news_text_col", "headline")
    )

    daily_embeddings = embedder.aggregate_daily(
        df_news_raw,
        headline_embeddings
    )

    # Rename columns for detection inside trainer
    daily_embeddings.columns = [
        f"emb_{i}" for i in range(daily_embeddings.shape[1])
    ]

    # ---- Merge structured + embeddings
    df_news = df_news_structured.join(daily_embeddings, how="left").fillna(0)

    logger.info("Aligning datasets...")
    df_model = align_price_and_news(df_price, df_news)
    df_model = df_model.dropna()

    logger.info(f"Dataset shape: {df_model.shape}")

    price_cols = config["price_features"]
    # news_cols = [
    #     col for col in df_model.columns
    #     if col.startswith("cat_") or col == "article_count"
    # ]
    # news_cols = [
    #     col for col in df_model.columns
    #     if col.startswith("cat_")
    #     or col == "article_count"
    #     or col.startswith("news_pca_")
    # ]

    trainer = get_model_trainer(config["model_type"], config)

    # --- Feature blocks
    structured_cols = [
        col for col in df_model.columns
        if col.startswith("cat_") or col == "article_count"
    ]

    embedding_cols = [
        col for col in df_model.columns
        if col.startswith("emb_")
    ]

    # --- Experiments
    res_price = trainer(df_model, feature_cols=price_cols)
    res_structured = trainer(df_model, feature_cols=structured_cols)
    res_embeddings = trainer(df_model, feature_cols=embedding_cols)
    res_price_embeddings = trainer(
        df_model,
        feature_cols=price_cols + embedding_cols
    )

    # --- Directional Accuracy
    da_price = directional_accuracy(
        res_price["y_test"], res_price["predictions"]
    )

    da_structured = directional_accuracy(
        res_structured["y_test"], res_structured["predictions"]
    )

    da_embeddings = directional_accuracy(
        res_embeddings["y_test"], res_embeddings["predictions"]
    )

    da_price_embeddings = directional_accuracy(
        res_price_embeddings["y_test"],
        res_price_embeddings["predictions"]
    )

    logger.info(f"Price-only DA: {da_price:.4f}")
    logger.info(f"Structured-only DA: {da_structured:.4f}")
    logger.info(f"Embeddings-only DA: {da_embeddings:.4f}")
    logger.info(f"Price + Embeddings DA: {da_price_embeddings:.4f}")

    logger.info(f"Embeddings fold DAs: {res_embeddings['fold_das']}")
    logger.info(f"Embeddings mean fold DA: {res_embeddings['mean_fold_da']:.4f}")


    results = {
        "price_only": collect_metrics(
            res_price,
            experiment_name="price_only",
            feature_set="price_features",
        ),
        "structured_only": collect_metrics(
            res_structured,
            experiment_name="structured_only",
            feature_set="structured_news_features",
        ),
        "embeddings_only": collect_metrics(
            res_embeddings,
            experiment_name="embeddings_only",
            feature_set="semantic_news_features",
        ),
        "price_plus_embeddings": collect_metrics(
            res_price_embeddings,
            experiment_name="price_plus_embeddings",
            feature_set="price+semantic_news_features",
        ),
    }

    persist_outputs(results, config)

    return results


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

