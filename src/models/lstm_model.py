import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


def train_lstm(df, feature_cols, config):

    target_col = "fwd_return"
    n_splits = config["model"]["n_splits"]
    lookback = config["model"].get("lookback", 20)

    series = df["log_return"].values
    future = df[target_col].values

    X_all, y_all = [], []

    for i in range(len(series) - lookback):
        X_all.append(series[i:i+lookback])
        y_all.append(future[i+lookback-1])

    X_all = np.array(X_all).reshape(-1, lookback, 1)
    y_all = np.array(y_all)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_das = []

    for train_idx, test_idx in tscv.split(X_all):

        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        # -------------------------
        # 🔹 SCALE INPUTS HERE
        # -------------------------
        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train.reshape(-1, 1)
        ).reshape(X_train.shape)

        X_test = scaler.transform(
            X_test.reshape(-1, 1)
        ).reshape(X_test.shape)
        # -------------------------

        model = Sequential([
            Input(shape=(lookback, 1)),
            LSTM(16),
            Dense(1, activation="tanh")
        ])

        model.compile(
            optimizer=Adam(0.001),
            loss="mse"
        )

        model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=32,
            verbose=0
        )

        preds = model.predict(X_test, verbose=0).flatten()

        preds_dir = (preds > 0).astype(int)
        y_true = (y_test > 0).astype(int)

        da = (preds_dir == y_true).mean()

        # # ---- DIAGNOSTIC ----
        # pos_ratio = y_true.mean()
        # naive_da = max(pos_ratio, 1 - pos_ratio)

        # print(
        #     f"Fold naive: {naive_da:.4f} | "
        #     f"LSTM: {da:.4f} | "
        #     f"Pos ratio: {pos_ratio:.4f}"
        # )
        # # --------------------

        fold_das.append(float(da))

    mean_da = float(np.mean(fold_das)) if fold_das else None

    return {
        "predictions": None,
        "y_test": None,
        "fold_das": fold_das,
        "mean_fold_da": mean_da,
    }