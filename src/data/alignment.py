import pandas as pd


def align_price_and_news(
    df_price: pd.DataFrame,
    df_news: pd.DataFrame,
    return_col: str = "log_return"
) -> pd.DataFrame:
    """
    Align daily price features and news features.
    Uses News_t to predict Return_t+1.
    """

    # Inner join on dates (keeps only overlapping range)
    df = df_price.join(df_news, how="inner")

    # Create prediction target (next-day return)
    df["target"] = df[return_col].shift(-1)

    # Drop last row (no target)
    df = df.dropna(subset=["target"])

    return df