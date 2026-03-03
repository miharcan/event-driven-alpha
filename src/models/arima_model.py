from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning
import numpy as np
import pandas as pd
import warnings


def train_arima(df, feature_cols, config):

    warnings.filterwarnings("ignore", category=ValueWarning)

    target_col = "fwd_return"
    n_splits = config["model"]["n_splits"]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    y = df[target_col].astype(float).copy()
    y.index = pd.to_datetime(y.index)
    y = y.sort_index()

    fold_das = []
    all_preds = []
    all_y = []

    for train_idx, test_idx in tscv.split(y):

        train_y = y.iloc[train_idx]
        test_y = y.iloc[test_idx]

        if len(test_y) == 0:
            continue

        try:
            model = ARIMA(train_y, order=(1, 0, 0))
            res = model.fit()
            fc = res.forecast(steps=len(test_y))
        except Exception:
            continue

        preds = (fc.values > 0).astype(int)
        y_true = (test_y.values > 0).astype(int)

        da = (preds == y_true).mean()

        fold_das.append(float(da))
        all_preds.extend(preds)
        all_y.extend(y_true)

    mean_da = float(np.mean(fold_das)) if fold_das else None

    return {
        "predictions": np.array(all_preds),
        "y_test": np.array(all_y),
        "fold_das": fold_das,
        "mean_fold_da": mean_da,
    }