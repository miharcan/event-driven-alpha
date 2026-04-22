import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import Ridge
from src.models.utils import apply_fold_pca, expanding_window_slices


def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))


def run_regime_gated_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols,
    config=None,
):
    """
    Regime-specific training:
    - Train one model on high-vol regime
    - Train one model on low-vol regime
    - Predict using correct regime model
    """

    if "vol_regime_high" not in df.columns:
        raise ValueError("vol_regime_high column missing.")

    X = df[feature_cols].copy()
    y = df[target_col]

    folds = config["model"]["n_splits"]
    splits = expanding_window_slices(len(df), folds, train_fraction=0.6)
    if not splits:
        raise ValueError("Not enough data for walk-forward split.")

    all_predictions = []
    all_y_test = []
    fold_das = []
    fold_sizes = []

    for train_end, test_end in splits:

        X_train = X.iloc[:train_end].copy()
        y_train = y.iloc[:train_end]

        X_test = X.iloc[train_end:test_end].copy()
        y_test = y.iloc[train_end:test_end]
        fold_sizes.append(len(y_test))

        pca_components = config.get("publication", {}).get("embedding_pca_components", 10)
        X_train, X_test = apply_fold_pca(
            X_train,
            X_test,
            pca_components=pca_components,
        )

        # Split by regime
        high_mask = df.iloc[:train_end]["vol_regime_high"] == 1
        low_mask = df.iloc[:train_end]["vol_regime_high"] == 0

        if high_mask.sum() < 20 or low_mask.sum() < 20:
            # Fallback to single model if regime too small
            model = Ridge(alpha=config.get("ridge_alpha", 1.0))
            model.fit(X_train, y_train)
            feature_names = X_train.columns
            X_test = pd.DataFrame(X_test, columns=feature_names, index=X_test.index)
            preds = model.predict(X_test)
        else:
            model_high = Ridge(alpha=config.get("ridge_alpha", 1.0))
            model_low = Ridge(alpha=config.get("ridge_alpha", 1.0))

            model_high.fit(X_train[high_mask], y_train[high_mask])
            model_low.fit(X_train[low_mask], y_train[low_mask])

            preds = []

            for idx in X_test.index:
                row = X_test.loc[[idx]]  # always DataFrame

                if df.loc[idx, "vol_regime_high"] == 1:
                    preds.append(model_high.predict(row)[0])
                else:
                    preds.append(model_low.predict(row)[0])

            preds = np.array(preds)

        all_predictions.extend(preds)
        all_y_test.extend(y_test)

        fold_da = directional_accuracy(y_test, preds)
        fold_das.append(fold_da)

    return {
        "mean_fold_da": np.mean(fold_das),
        "fold_das": fold_das,
        "predictions": np.array(all_predictions),
        "y_test": np.array(all_y_test),
        "fold_sizes": fold_sizes,
        "n_test_obs": int(sum(fold_sizes)),
    }


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
    feature_cols=None,
    config=None
):

    X = df[feature_cols].copy()
    y = df[target_col]

    folds = config["model"]["n_splits"]
    train_start = 0
    splits = expanding_window_slices(len(df), folds, train_fraction=0.6)
    if not splits:
        raise ValueError("Not enough data for walk-forward split.")

    all_predictions = []
    all_y_test = []
    fold_das = []
    fold_sizes = []

    for train_end, test_end in splits:

        X_train = X.iloc[train_start:train_end].copy()
        y_train = y.iloc[train_start:train_end]

        X_test = X.iloc[train_end:test_end].copy()
        y_test = y.iloc[train_end:test_end]
        fold_sizes.append(len(y_test))

        pca_components = config.get("publication", {}).get("embedding_pca_components", 10)
        X_train, X_test = apply_fold_pca(
            X_train,
            X_test,
            pca_components=pca_components,
        )

        model = Ridge(alpha=config.get("ridge_alpha", 1.0))
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # store fold results
        all_predictions.extend(preds)
        all_y_test.extend(y_test)

        fold_da = directional_accuracy(y_test, preds)
        fold_das.append(fold_da)

    mse = mean_squared_error(all_y_test, all_predictions)
    r2 = r2_score(all_y_test, all_predictions)

    return {
        "model": model,
        "mse": mse,
        "r2": r2,
        "y_test": np.array(all_y_test),
        "predictions": np.array(all_predictions),
        "fold_das": fold_das,
        "mean_fold_da": np.mean(fold_das),
        "fold_sizes": fold_sizes,
        "n_test_obs": int(sum(fold_sizes)),
    }


def train_ridge_regression(
    df: pd.DataFrame,
    target_col: str = "target",
    feature_cols=None,
    alpha: float = 1.0,
):

    train, test = train_test_split_time_series(df)

    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]

    X_train = train[feature_cols]
    y_train = train[target_col]

    X_test = test[feature_cols]
    y_test = test[target_col]

    model = Ridge(alpha=alpha)
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
