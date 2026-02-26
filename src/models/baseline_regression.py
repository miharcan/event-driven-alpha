import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA


def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))


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

    n = len(df)
    folds = 4
    train_start = 0
    initial_train_size = int(n * 0.6)
    fold_size = int((n - initial_train_size) / folds)

    all_predictions = []
    all_y_test = []
    fold_das = []

    for i in range(folds):

        train_end = initial_train_size + i * fold_size
        test_end = train_end + fold_size

        X_train = X.iloc[train_start:train_end].copy()
        y_train = y.iloc[train_start:train_end]

        X_test = X.iloc[train_end:test_end].copy()
        y_test = y.iloc[train_end:test_end]

        # ---- Detect embedding columns
        embedding_cols = [c for c in X.columns if c.startswith("emb_")]

        if embedding_cols:

            n_components = config.get("embedding_pca_components", 75)
            pca = PCA(n_components=n_components)

            X_train_emb = pca.fit_transform(X_train[embedding_cols])
            X_test_emb = pca.transform(X_test[embedding_cols])

            X_train = X_train.drop(columns=embedding_cols)
            X_test = X_test.drop(columns=embedding_cols)

            for j in range(n_components):
                X_train[f"emb_pca_{j}"] = X_train_emb[:, j]
                X_test[f"emb_pca_{j}"] = X_test_emb[:, j]

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