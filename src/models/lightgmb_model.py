import numpy as np
import lightgbm as lgb
from src.models.baseline_regression import directional_accuracy


def train_lightgbm(df, feature_cols, config):

    target_col = "fwd_return"
    folds = config["model"]["n_splits"]

    n = len(df)
    initial_train_size = int(n * 0.6)
    fold_size = int((n - initial_train_size) / folds)

    fold_das = []
    all_preds = []
    all_y = []

    for i in range(folds):

        train_end = initial_train_size + i * fold_size
        test_end = train_end + fold_size

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train = train_df[feature_cols]
        y_train = (train_df[target_col] > 0).astype(int)

        X_test = test_df[feature_cols]
        y_test = (test_df[target_col] > 0).astype(int)

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

        da = (preds == y_test).mean()

        fold_das.append(da)
        all_preds.extend(preds)
        all_y.extend(y_test)

    return {
        "predictions": np.array(all_preds),
        "y_test": np.array(all_y),
        "fold_das": fold_das,
        "mean_fold_da": np.mean(fold_das)
    }