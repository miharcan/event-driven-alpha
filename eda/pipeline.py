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
    df_news_raw = load_news_json(config["news_path"])
    df_news = build_daily_news_features(df_news_raw)

    logger.info("Aligning datasets...")
    df_model = align_price_and_news(df_price, df_news)
    df_model = df_model.dropna()

    logger.info(f"Dataset shape: {df_model.shape}")

    price_cols = config["price_features"]
    news_cols = [
        col for col in df_model.columns
        if col.startswith("cat_") or col == "article_count"
    ]

    trainer = get_model_trainer(config["model_type"], config)


    res_price = trainer(df_model, feature_cols=price_cols)
    res_news = trainer(df_model, feature_cols=news_cols)
    all_features = [
        col for col in df_model.columns
        if col != "target"
    ]

    res_combined = trainer(df_model, feature_cols=all_features)

    da_price = directional_accuracy(res_price["y_test"], res_price["predictions"])
    da_news = directional_accuracy(res_news["y_test"], res_news["predictions"])
    da_combined = directional_accuracy(
        res_combined["y_test"], res_combined["predictions"]
    )

    output_dir = Path(config.get("output_dir", "outputs"))
    pred_df = pd.DataFrame({
        "y_true": res_combined["y_test"],
        "y_pred": res_combined["predictions"],
    })
    pred_df.to_csv(output_dir / "predictions.csv")

    logger.info(f"Price-only DA: {da_price:.4f}")
    logger.info(f"News-only DA: {da_news:.4f}")
    logger.info(f"Combined DA: {da_combined:.4f}")

    results = {
        "price_only": collect_metrics(
            res_price,
            experiment_name="price_only",
            feature_set="price_features",
        ),
        "news_only": collect_metrics(
            res_news,
            experiment_name="news_only",
            feature_set="news_features",
        ),
        "combined": collect_metrics(
            res_combined,
            experiment_name="combined",
            feature_set="price+news_features",
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

