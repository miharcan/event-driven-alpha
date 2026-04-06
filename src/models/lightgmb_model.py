import logging
from time import perf_counter
import numpy as np
import lightgbm as lgb

logger = logging.getLogger(__name__)


def _resolve_lgb_device(config):
    requested = str(config["model"].get("compute_device", "auto")).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        requested = "auto"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "gpu"
    return "cpu"


def train_lightgbm(df, feature_cols, config):

    target_col = "fwd_return"
    folds = config["model"]["n_splits"]

    n = len(df)
    initial_train_size = int(n * 0.6)
    fold_size = int((n - initial_train_size) / folds)

    fold_das = []
    all_preds = []
    all_y = []

    device_type = _resolve_lgb_device(config)
    logger.info(
        "LightGBM trainer: device=%s rows=%d features=%d folds=%d",
        device_type,
        len(df),
        len(feature_cols),
        folds,
    )

    for i in range(folds):
        fold_start = perf_counter()

        train_end = initial_train_size + i * fold_size
        test_end = train_end + fold_size

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train = train_df[feature_cols]
        y_train = (train_df[target_col] > 0).astype(int)

        X_test = test_df[feature_cols]
        y_test = (test_df[target_col] > 0).astype(int)

        model_kwargs = dict(
            n_estimators=config["model"].get("lgb_n_estimators", 200),
            max_depth=config["model"].get("lgb_max_depth", -1),
            learning_rate=config["model"].get("lgb_learning_rate", 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            device_type=device_type,
        )
        model = lgb.LGBMClassifier(**model_kwargs)

        try:
            model.fit(X_train, y_train)
        except Exception as e:
            if device_type == "gpu":
                logger.warning("LightGBM GPU execution failed (%s). Retrying on CPU.", e)
                model_kwargs["device_type"] = "cpu"
                model = lgb.LGBMClassifier(**model_kwargs)
                model.fit(X_train, y_train)
            else:
                raise

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

        da = (preds == y_test).mean()

        fold_das.append(da)
        all_preds.extend(preds)
        all_y.extend(y_test)
        logger.info(
            "LightGBM fold %d/%d done in %.2fs (DA=%.4f)",
            i + 1,
            folds,
            perf_counter() - fold_start,
            da,
        )

    return {
        "predictions": np.array(all_preds),
        "y_test": np.array(all_y),
        "fold_das": fold_das,
        "mean_fold_da": np.mean(fold_das)
    }
