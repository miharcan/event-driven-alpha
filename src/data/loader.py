import pandas as pd
from pathlib import Path
import openpyxl


def load_auronum_series(path: str, date_col: int, price_col: int) -> pd.DataFrame:
    """
    Parse Auronum multi-asset Excel sheet and extract a specific series
    based on column indices.

    Parameters:
        path (str): Path to Excel file
        date_col (int): Column index for date
        price_col (int): Column index for price

    Returns:
        pd.DataFrame indexed by datetime with single 'price' column
    """

    df = pd.read_excel(path, header=None)

    series = df.iloc[1:, [date_col, price_col]].copy()
    series.columns = ["date", "price"]

    series = series.dropna(subset=["date"])

    series["date"] = pd.to_datetime(series["date"], errors="coerce")
    series["price"] = pd.to_numeric(series["price"], errors="coerce")

    series = series.dropna()

    series = series.sort_values("date")
    series = series.set_index("date")

    return series
