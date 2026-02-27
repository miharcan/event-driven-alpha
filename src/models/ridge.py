from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import numpy as np

from src.models.baseline_regression import directional_accuracy


def train(df, feature_cols, alpha=1.0, target_col="fwd_return"):

    all_test_indices = []

    X = df[feature_cols].copy()
    y = df[target_col]

    n = len(df)

    # ---- Walk-forward setup
    folds = 4
    initial_train_size = int(n * 0.6)
    fold_size = int((n - initial_train_size) / folds)

    if fold_size <= 0:
        raise ValueError("Not enough data for walk-forward split.")

    all_predictions = []
    all_y_test = []
    fold_das = []

    # ---- Aggregate regime tracking
    all_high_y = []
    all_high_preds = []

    all_low_y = []
    all_low_preds = []

    for i in range(folds):

        train_end = initial_train_size + i * fold_size
        test_end = train_end + fold_size

        print(
            f"Fold {i+1}: "
            f"Train end {df.index[train_end]} | "
            f"Test start {df.index[train_end]} | "
            f"Test end {df.index[test_end-1]}"
        )

        X_train = X.iloc[:train_end].copy()
        y_train = y.iloc[:train_end]

        X_test = X.iloc[train_end:test_end].copy()
        y_test = y.iloc[train_end:test_end]

        # ---- Regime diagnostics (only if regime column exists)
        if "vol_regime_high" in X_test.columns:

            regime_counts = X_test["vol_regime_high"].value_counts()

            high_count = regime_counts.get(1, 0)
            low_count = regime_counts.get(0, 0)

            print(
                f"Fold {i+1} Regime Distribution | "
                f"High Vol: {high_count} | Low Vol: {low_count}"
            )

        all_test_indices.extend(X_test.index)

        # ---- No embedding logic here anymore
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # ---- Regime-conditional DA
        # if "vol_regime_high" in X_test.columns:

        #     high_mask = X_test["vol_regime_high"] == 1
        #     low_mask = X_test["vol_regime_high"] == 0

        #     if high_mask.sum() > 5:
        #         da_high = directional_accuracy(
        #             y_test[high_mask],
        #             preds[high_mask]
        #         )
        #         print(f"Fold {i+1} High-Vol DA: {da_high:.4f}")

        #     if low_mask.sum() > 5:
        #         da_low = directional_accuracy(
        #             y_test[low_mask],
        #             preds[low_mask]
        #         )
        #         print(f"Fold {i+1} Low-Vol DA: {da_low:.4f}")
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

    
    # ---- Aggregated Regime DA
    if len(all_high_y) > 10:
        high_da = directional_accuracy(
            np.array(all_high_y),
            np.array(all_high_preds)
        )
        print(f"Aggregated High-Vol DA: {high_da:.4f} | Samples: {len(all_high_y)}")

    if len(all_low_y) > 10:
        low_da = directional_accuracy(
            np.array(all_low_y),
            np.array(all_low_preds)
        )
        print(f"Aggregated Low-Vol DA: {low_da:.4f} | Samples: {len(all_low_y)}")
    
    mse = mean_squared_error(all_y_test, all_predictions)
    r2 = r2_score(all_y_test, all_predictions)

    return {
        "model": model,
        "mse": mse,
        "r2": r2,
        "y_test": np.array(all_y_test),
        "predictions": np.array(all_predictions),
        "test_index": all_test_indices,
        "fold_das": fold_das,
        "mean_fold_da": np.mean(fold_das),
    }