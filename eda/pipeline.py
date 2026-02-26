import logging
from pathlib import Path

from src.data.loader import load_auronum_series
from src.features.price_features import (
    compute_log_returns,
    compute_rolling_volatility,
    add_lag_features,
)
from src.data.news_loader import load_news_json
from src.features.news_features import build_daily_news_features
from src.data.alignment import align_price_and_news
from src.models.baseline_regression import (
    train_baseline_regression,
    directional_accuracy,
)


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

    res_price = train_baseline_regression(df_model, feature_cols=price_cols)
    da_price = directional_accuracy(res_price["y_test"], res_price["predictions"])

    res_news = train_baseline_regression(df_model, feature_cols=news_cols)
    da_news = directional_accuracy(res_news["y_test"], res_news["predictions"])

    res_combined = train_baseline_regression(df_model)
    da_combined = directional_accuracy(
        res_combined["y_test"], res_combined["predictions"]
    )

    logger.info(f"Price-only DA: {da_price:.4f}")
    logger.info(f"News-only DA: {da_news:.4f}")
    logger.info(f"Combined DA: {da_combined:.4f}")

    return {
        "price_da": da_price,
        "news_da": da_news,
        "combined_da": da_combined,
    }