from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning
import numpy as np
import pandas as pd
import warnings
from src.models.utils import expanding_window_slices


def train_arima(df, feature_cols, config):

    warnings.filterwarnings("ignore", category=ValueWarning)

    target_col = "fwd_return"
    n_splits = config["model"]["n_splits"]
    arima_order = tuple(config["model"].get("arima_order", [1, 0, 0]))

    splits = expanding_window_slices(len(df), n_splits, train_fraction=0.6)
    if not splits:
        raise ValueError("Not enough data for walk-forward split.")

    y = df[target_col].astype(float).copy()
    # Keep temporal ordering when a datetime-like index exists.
    try:
        idx_dt = pd.to_datetime(y.index, errors="coerce")
        if not idx_dt.isna().any():
            y.index = idx_dt
            y = y.sort_index()
    except Exception:
        pass
    # Use a supported integer index for statsmodels forecasting APIs.
    y = pd.Series(y.to_numpy(), index=pd.RangeIndex(len(y)), name=target_col)

    fold_das = []
    all_preds = []
    all_y = []
    fold_sizes = []

    for train_end, test_end in splits:

        train_y = y.iloc[:train_end]
        test_y = y.iloc[train_end:test_end]

        if len(test_y) == 0:
            continue
        fold_sizes.append(len(test_y))

        try:
            model = ARIMA(train_y, order=arima_order)
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
        "fold_sizes": fold_sizes,
        "n_test_obs": int(sum(fold_sizes)),
    }
