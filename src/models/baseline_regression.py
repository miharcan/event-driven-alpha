import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA


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

    n = len(df)
    folds = 4
    initial_train_size = int(n * 0.6)
    fold_size = int((n - initial_train_size) / folds)

    all_predictions = []
    all_y_test = []
    fold_das = []

    for i in range(folds):

        train_end = initial_train_size + i * fold_size
        test_end = train_end + fold_size

        X_train = X.iloc[:train_end].copy()
        y_train = y.iloc[:train_end]

        X_test = X.iloc[train_end:test_end].copy()
        y_test = y.iloc[train_end:test_end]

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

            max_components = min(
                X_train[embedding_cols].shape[0],
                X_train[embedding_cols].shape[1]
            )

            n_components = min(
                config.get("embedding_pca_components", 75),
                max_components
            )
            pca = PCA(n_components=n_components)

            X_train_emb = pca.fit_transform(X_train[embedding_cols])
            X_test_emb = pca.transform(X_test[embedding_cols])

            # Drop raw embedding columns
            X_train = X_train.drop(columns=embedding_cols)
            X_test = X_test.drop(columns=embedding_cols)

            # Create PCA DataFrames
            pca_columns = [f"emb_pca_{j}" for j in range(n_components)]

            pca_df_train = pd.DataFrame(
                X_train_emb,
                index=X_train.index,
                columns=pca_columns
            )

            pca_df_test = pd.DataFrame(
                X_test_emb,
                index=X_test.index,
                columns=pca_columns
            )

            # Concatenate cleanly
            X_train = pd.concat([X_train, pca_df_train], axis=1)
            X_test = pd.concat([X_test, pca_df_test], axis=1)

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