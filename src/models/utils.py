import pandas as pd
from sklearn.decomposition import PCA


def train_test_split_time_series(df: pd.DataFrame, test_size: float = 0.2):
    split_index = int(len(df) * (1 - test_size))
    return df.iloc[:split_index], df.iloc[split_index:]


def expanding_window_slices(n_rows: int, n_splits: int, train_fraction: float = 0.6):
    """
    Generate contiguous expanding-window train/test index bounds.
    Returns a list of (train_end, test_end) tuples where:
    - train rows are [0:train_end]
    - test rows are [train_end:test_end]
    """
    if n_rows <= 0:
        return []
    initial_train_size = int(n_rows * train_fraction)
    fold_size = int((n_rows - initial_train_size) / n_splits)
    if fold_size <= 0:
        return []

    splits = []
    for i in range(n_splits):
        train_end = initial_train_size + i * fold_size
        test_end = min(train_end + fold_size, n_rows)
        if test_end > train_end:
            splits.append((train_end, test_end))
    return splits


def apply_fold_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    pca_components: int = 10,
):
    """
    Fit PCA on training embedding columns only and transform both train/test.
    This prevents test-period information leaking into embedding representation.
    """
    embedding_cols = [c for c in X_train.columns if c.startswith("emb_")]
    if not embedding_cols:
        return X_train, X_test

    max_components = min(len(X_train), len(embedding_cols))
    n_components = min(pca_components, max_components)
    if n_components <= 0:
        return X_train.drop(columns=embedding_cols), X_test.drop(columns=embedding_cols)

    pca = PCA(n_components=n_components)
    X_train_emb = pca.fit_transform(X_train[embedding_cols])
    X_test_emb = pca.transform(X_test[embedding_cols])

    X_train_out = X_train.drop(columns=embedding_cols).copy()
    X_test_out = X_test.drop(columns=embedding_cols).copy()

    pca_columns = [f"emb_pca_{j}" for j in range(n_components)]
    X_train_pca = pd.DataFrame(X_train_emb, index=X_train.index, columns=pca_columns)
    X_test_pca = pd.DataFrame(X_test_emb, index=X_test.index, columns=pca_columns)

    X_train_out = pd.concat([X_train_out, X_train_pca], axis=1)
    X_test_out = pd.concat([X_test_out, X_test_pca], axis=1)
    return X_train_out, X_test_out
