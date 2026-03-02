# src/models/xgboost_model.py

import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from src.models.baseline_regression import directional_accuracy


def train_xgboost(df, feature_cols, config):

    target_col = "fwd_return"
    n_splits = config["model"]["n_splits"]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_das = []
    all_preds = []
    all_y = []

    for train_idx, test_idx in tscv.split(df):

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        da = directional_accuracy(y_test, preds)

        fold_das.append(da)
        all_preds.extend(preds)
        all_y.extend(y_test)

    return {
        "predictions": np.array(all_preds),
        "y_test": np.array(all_y),
        "fold_das": fold_das,
        "mean_fold_da": np.mean(fold_das)
    }