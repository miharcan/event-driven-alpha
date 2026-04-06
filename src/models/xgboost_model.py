import logging
from time import perf_counter
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def _resolve_xgb_device(config):
    requested = str(config["model"].get("compute_device", "auto")).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        requested = "auto"

    has_cuda = False
    try:
        build_info = xgb.build_info()
        has_cuda = str(build_info.get("USE_CUDA", "False")).lower() in {"1", "true", "yes"}
    except Exception:
        has_cuda = False

    if requested == "cuda":
        return "cuda", has_cuda
    if requested == "cpu":
        return "cpu", has_cuda
    return ("cuda" if has_cuda else "cpu"), has_cuda


def train_xgboost(df, feature_cols, config):

    target_col = "fwd_return"
    n_splits = config["model"]["n_splits"]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_das = []
    all_preds = []
    all_y = []

    device, has_cuda = _resolve_xgb_device(config)
    if device == "cuda" and not has_cuda:
        logger.warning("XGBoost CUDA requested but this build has no CUDA support. Falling back to CPU.")
        device = "cpu"

    logger.info(
        "XGBoost trainer: device=%s rows=%d features=%d folds=%d",
        device,
        len(df),
        len(feature_cols),
        n_splits,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        fold_start = perf_counter()

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train = train_df[feature_cols]
        y_train = (train_df[target_col] > 0).astype(int)

        X_test = test_df[feature_cols]
        y_test = (test_df[target_col] > 0).astype(int)

        model_kwargs = dict(
            objective="binary:logistic",
            n_estimators=config["model"].get("xgb_n_estimators", 300),
            max_depth=config["model"].get("xgb_max_depth", 4),
            learning_rate=config["model"].get("xgb_learning_rate", 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=config["model"].get("xgb_n_jobs", -1),
            eval_metric="logloss",
            tree_method="hist",
        )
        if device == "cuda":
            model_kwargs["device"] = "cuda"

        model = xgb.XGBClassifier(**model_kwargs)
        try:
            model.fit(X_train, y_train)
        except xgb.core.XGBoostError as e:
            if device == "cuda":
                logger.warning("XGBoost CUDA execution failed (%s). Retrying on CPU.", e)
                model_kwargs.pop("device", None)
                model = xgb.XGBClassifier(**model_kwargs)
                model.fit(X_train, y_train)
            else:
                raise

        # preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

        da = (preds == y_test).mean()

        fold_das.append(da)
        all_preds.extend(preds)
        all_y.extend(y_test)
        logger.info(
            "XGBoost fold %d/%d done in %.2fs (DA=%.4f)",
            fold_idx,
            n_splits,
            perf_counter() - fold_start,
            da,
        )

    return {
        "predictions": np.array(all_preds),
        "y_test": np.array(all_y),
        "fold_das": fold_das,
        "mean_fold_da": np.mean(fold_das)
    }
