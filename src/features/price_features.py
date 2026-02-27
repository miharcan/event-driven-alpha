import numpy as np
import pandas as pd


def add_forward_return(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Add forward log return over given horizon.
    """
    df = df.copy()

    if "price" not in df.columns:
        raise ValueError("price column not found")

    df["fwd_return"] = (
        np.log(df["price"].shift(-horizon))
        - np.log(df["price"])
    )

    return df

def compute_log_returns(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """
    Compute log returns from price series safely.
    """
    df = df.copy()

    if price_col not in df.columns:
        raise ValueError(f"{price_col} not found in DataFrame.")

    # Ensure numeric
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # Replace non-positive values with NaN
    df.loc[df[price_col] <= 0, price_col] = np.nan

    # Compute log return
    df["log_return"] = np.log(df[price_col]).diff()

    return df


def compute_rolling_volatility(
    df: pd.DataFrame,
    return_col: str = "log_return",
    window: int = 21
) -> pd.DataFrame:
    """
    Compute rolling volatility (standard deviation of returns).

    Default window=21 ~ 1 trading month.
    """
    df = df.copy()

    if return_col not in df.columns:
        raise ValueError(f"{return_col} not found in DataFrame.")

    df[f"rolling_vol_{window}"] = (
        df[return_col]
        .rolling(window=window)
        .std()
    )

    return df


def add_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: int = 3
) -> pd.DataFrame:
    """
    Add lagged versions of a given column.

    Example:
        column='log_return', lags=3
        → log_return_lag1, log_return_lag2, log_return_lag3
    """
    df = df.copy()

    if column not in df.columns:
        raise ValueError(f"{column} not found in DataFrame.")

    for lag in range(1, lags + 1):
        df[f"{column}_lag{lag}"] = df[column].shift(lag)

    return df


def add_volatility_regime(
    df: pd.DataFrame,
    vol_col: str = "rolling_vol_21",
    long_window: int = 252
) -> pd.DataFrame:
    """
    Add volatility regime feature:
    1 if current volatility > long-term average volatility,
    else 0.
    """
    df = df.copy()

    if vol_col not in df.columns:
        raise ValueError(f"{vol_col} not found in DataFrame.")

    # Long-term rolling mean of volatility
    df[f"{vol_col}_mean_{long_window}"] = (
        df[vol_col]
        .rolling(window=long_window)
        .mean()
    )

    # High-vol regime indicator
    df["vol_regime_high"] = (
        df[vol_col] > df[f"{vol_col}_mean_{long_window}"]
    ).astype(int)

    return df