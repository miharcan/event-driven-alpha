import pandas as pd


def aggregate_daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total article count per day.
    """

    daily_counts = (
        df.groupby(df.index)
        .size()
        .to_frame(name="article_count")
    )

    return daily_counts


def aggregate_category_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create daily category count matrix.
    """

    category_counts = (
        df.groupby([df.index, "category"])
        .size()
        .unstack(fill_value=0)
    )

    # Optional: normalize category names
    category_counts.columns = [
        f"cat_{str(col).lower().replace(' ', '_')}"
        for col in category_counts.columns
    ]

    return category_counts


def build_daily_news_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine total counts and category counts into one daily feature matrix.
    """

    daily_counts = aggregate_daily_counts(df)
    category_counts = aggregate_category_counts(df)

    features = daily_counts.join(category_counts, how="left")

    return features