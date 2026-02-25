import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def train_test_split_time_series(df: pd.DataFrame, test_size: float = 0.2):
    """
    Time-series split (no shuffling).
    """
    split_index = int(len(df) * (1 - test_size))
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test


def train_baseline_regression(
    df: pd.DataFrame,
    target_col: str = "target",
    feature_cols=None
):

    train, test = train_test_split_time_series(df)

    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]

    X_train = train[feature_cols]
    y_train = train[target_col]

    X_test = test[feature_cols]
    y_test = test[target_col]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "model": model,
        "mse": mse,
        "r2": r2,
        "y_test": y_test,
        "predictions": predictions,
    }