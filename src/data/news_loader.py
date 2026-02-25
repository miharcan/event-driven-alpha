import pandas as pd
from pathlib import Path


def load_news_json(path: str) -> pd.DataFrame:
    """
    Load news dataset from JSON Lines (one JSON object per line).

    Returns:
        DataFrame indexed by datetime with selected fields.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # JSON Lines loader (memory efficient)
    df = pd.read_json(path, lines=True)

    required_columns = ["date", "headline", "short_description", "category"]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[required_columns].copy()

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df.sort_values("date")
    df = df.set_index("date")

    return df