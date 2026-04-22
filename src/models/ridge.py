from sklearn.linear_model import Ridge
import numpy as np

from src.models.baseline_regression import directional_accuracy
from src.models.utils import apply_fold_pca, expanding_window_slices


def train(df, config, feature_cols, alpha=1.0, target_col="fwd_return"):

    all_test_indices = []

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

    # ---- Aggregate regime tracking
    all_high_y = []
    all_high_preds = []

    all_low_y = []
    all_low_preds = []

    for train_end, test_end in splits:

        X_train = X.iloc[:train_end].copy()
        y_train = y.iloc[:train_end]

        X_test = X.iloc[train_end:test_end].copy()
        y_test = y.iloc[train_end:test_end]
        fold_sizes.append(len(y_test))

        all_test_indices.extend(X_test.index)

        pca_components = config.get("publication", {}).get("embedding_pca_components", 10)
        X_train, X_test = apply_fold_pca(
            X_train,
            X_test,
            pca_components=pca_components,
        )

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        if "vol_regime_high" in X_test.columns:

            high_mask = X_test["vol_regime_high"] == 1
            low_mask = X_test["vol_regime_high"] == 0

            # Append high-vol samples
            if high_mask.sum() > 0:
                all_high_y.extend(y_test[high_mask])
                all_high_preds.extend(preds[high_mask])

            # Append low-vol samples
            if low_mask.sum() > 0:
                all_low_y.extend(y_test[low_mask])
                all_low_preds.extend(preds[low_mask])

        all_predictions.extend(preds)
        all_y_test.extend(y_test)

        fold_da = directional_accuracy(y_test, preds)
        fold_das.append(fold_da)

    if len(all_high_y) > 10:
        directional_accuracy(np.array(all_high_y), np.array(all_high_preds))

    if len(all_low_y) > 10:
        directional_accuracy(np.array(all_low_y), np.array(all_low_preds))

    return {
        "y_test": np.array(all_y_test),
        "predictions": np.array(all_predictions),
        "test_index": all_test_indices,
        "fold_das": fold_das,
        "mean_fold_da": np.mean(fold_das),
        "fold_sizes": fold_sizes,
        "n_test_obs": int(sum(fold_sizes)),
    }


def train_regime_specific(df, config, feature_cols, alpha=1.0, target_col="fwd_return"):

    if "vol_regime_high" not in df.columns:
        raise ValueError("vol_regime_high must exist in df")

    X = df[feature_cols].copy()
    y = df[target_col]
    regimes = df["vol_regime_high"]

    # Remove regime from feature matrix if included
    if "vol_regime_high" in X.columns:
        X = X.drop(columns=["vol_regime_high"])

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
        regime_train = regimes.iloc[:train_end]

        X_test = X.iloc[train_end:test_end].copy()
        y_test = y.iloc[train_end:test_end]
        regime_test = regimes.iloc[train_end:test_end]
        fold_sizes.append(len(y_test))

        pca_components = config.get("publication", {}).get("embedding_pca_components", 10)
        X_train, X_test = apply_fold_pca(
            X_train,
            X_test,
            pca_components=pca_components,
        )

        high_mask_train = regime_train == 1
        low_mask_train = regime_train == 0

        model_high = Ridge(alpha=alpha)
        model_low = Ridge(alpha=alpha)

        if high_mask_train.sum() > 5:
            model_high.fit(
                X_train[high_mask_train],
                y_train[high_mask_train]
            )

        if low_mask_train.sum() > 5:
            model_low.fit(
                X_train[low_mask_train],
                y_train[low_mask_train]
            )

        # ---- Vectorized prediction
        preds = np.zeros(len(X_test))

        high_mask_test = regime_test == 1
        low_mask_test = regime_test == 0

        if high_mask_test.sum() > 0 and high_mask_train.sum() > 5:
            preds[high_mask_test] = model_high.predict(
                X_test[high_mask_test]
            )

        if low_mask_test.sum() > 0 and low_mask_train.sum() > 5:
            preds[low_mask_test] = model_low.predict(
                X_test[low_mask_test]
            )

        all_predictions.extend(preds)
        all_y_test.extend(y_test)

        fold_da = directional_accuracy(y_test, preds)
        fold_das.append(fold_da)

    return {
        "predictions": np.array(all_predictions),
        "y_test": np.array(all_y_test),
        "mean_fold_da": np.mean(fold_das),
        "fold_das": fold_das,
        "fold_sizes": fold_sizes,
        "n_test_obs": int(sum(fold_sizes)),
    }
