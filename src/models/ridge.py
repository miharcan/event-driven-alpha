from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import numpy as np

from src.models.baseline_regression import directional_accuracy


def train(df, feature_cols, alpha=1.0, target_col="target"):

    all_test_indices = []

    X = df[feature_cols].copy()
    y = df[target_col]

    n = len(df)

    # ---- Walk-forward setup
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

        all_test_indices.extend(X_test.index)

        # ---- Detect embedding columns
        embedding_cols = [c for c in X.columns if c.startswith("emb_")]

        if embedding_cols:

            n_components = 75  # or pass via config later
            pca = PCA(n_components=n_components)

            # Fit PCA ONLY on training embeddings
            X_train_emb = pca.fit_transform(X_train[embedding_cols])
            X_test_emb = pca.transform(X_test[embedding_cols])

            # Drop raw embedding columns
            X_train = X_train.drop(columns=embedding_cols)
            X_test = X_test.drop(columns=embedding_cols)

            # Add PCA components
            for j in range(n_components):
                X_train[f"emb_pca_{j}"] = X_train_emb[:, j]
                X_test[f"emb_pca_{j}"] = X_test_emb[:, j]

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

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
        "test_index": all_test_indices,
        "fold_das": fold_das,
        "mean_fold_da": np.mean(fold_das),
    }