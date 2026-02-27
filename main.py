import pandas as pd

from src.data.loader import load_auronum_series
from src.features.price_features import (
    compute_log_returns,
    compute_rolling_volatility,
    add_lag_features,
    add_volatility_regime
)
from src.data.news_loader import load_news_json
from src.features.news_features import build_daily_news_features
from src.data.alignment import align_price_and_news
from src.models.baseline_regression import (
    train_baseline_regression,
    directional_accuracy,
)


def build_dataset():
    print("Loading price data...")
    df_price = load_auronum_series(
        "data/raw/HistoricPriceData.xlsx",
        date_col=4,
        price_col=5,
    )

    df_price = compute_log_returns(df_price)
    df_price = compute_rolling_volatility(df_price)
    df_price = add_lag_features(df_price, "log_return", 3)
    df_price = add_volatility_regime(df_price)

    print("Loading news data...")
    df_news_raw = load_news_json("data/raw/News_Category_Dataset_v3.json")
    df_news = build_daily_news_features(df_news_raw)

    print("Aligning datasets...")
    df_model = align_price_and_news(df_price, df_news)
    df_model = df_model.dropna()

    print("Final dataset shape:", df_model.shape)
    print("Date range:", df_model.index.min(), "→", df_model.index.max())

    return df_model


def run_experiments(df_model):
    print("\nRunning model comparison...")

    price_cols = [
        "log_return",
        "rolling_vol_21",
        "log_return_lag1",
        "log_return_lag2",
        "log_return_lag3",
    ]

    news_cols = [
        col
        for col in df_model.columns
        if col.startswith("cat_") or col == "article_count"
    ]

    # Price-only
    res_price = train_baseline_regression(df_model, feature_cols=price_cols)
    da_price = directional_accuracy(res_price["y_test"], res_price["predictions"])

    # News-only
    res_news = train_baseline_regression(df_model, feature_cols=news_cols)
    da_news = directional_accuracy(res_news["y_test"], res_news["predictions"])

    # Combined
    res_combined = train_baseline_regression(df_model)
    da_combined = directional_accuracy(
        res_combined["y_test"], res_combined["predictions"]
    )

    print("\nResults:")
    print("Price-only DA:  ", round(da_price, 4))
    print("News-only DA:   ", round(da_news, 4))
    print("Combined DA:    ", round(da_combined, 4))


if __name__ == "__main__":
    df_model = build_dataset()
    run_experiments(df_model)