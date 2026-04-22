from src.models import linear, ridge


def _optional_import_error(dep_name: str, model_type: str, exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"Model '{model_type}' requires optional dependency '{dep_name}', "
        f"but it is not installed. Install it and retry. Original error: {exc}"
    )


def get_model_trainer(model_type, config):
    if model_type == "linear":
        return linear.train

    elif model_type == "xgboost":
        try:
            from src.models.xgboost_model import train_xgboost
        except Exception as exc:
            raise _optional_import_error("xgboost", "xgboost", exc) from exc
        return lambda df, feature_cols: train_xgboost(
            df,
            feature_cols=feature_cols,
            config=config,
        )

    elif model_type == "lightgbm":
        try:
            from src.models.lightgmb_model import train_lightgbm
        except Exception as exc:
            raise _optional_import_error("lightgbm", "lightgbm", exc) from exc
        return lambda df, feature_cols: train_lightgbm(
            df,
            feature_cols=feature_cols,
            config=config,
        )

    elif model_type == "ridge":
        alpha = config["model"].get("ridge_alpha", 1.0)
        return lambda df, feature_cols: ridge.train(
            df=df,
            config=config,
            feature_cols=feature_cols,
            alpha=alpha,
        )

    elif model_type == "arima":
        try:
            from src.models.arima_model import train_arima
        except Exception as exc:
            raise _optional_import_error("statsmodels", "arima", exc) from exc
        return lambda df, feature_cols: train_arima(
            df=df,
            feature_cols=feature_cols,
            config=config,
        )

    elif model_type == "lstm":
        try:
            from src.models.lstm_model import train_lstm
        except Exception as exc:
            raise _optional_import_error("torch", "lstm", exc) from exc
        return lambda df, feature_cols: train_lstm(
            df,
            feature_cols=feature_cols,
            config=config,
        )

    elif model_type == "tcn":
        try:
            from src.models.tcn_model import train_tcn
        except Exception as exc:
            raise _optional_import_error("torch", "tcn", exc) from exc
        return lambda df, feature_cols: train_tcn(
            df=df,
            feature_cols=feature_cols,
            config=config,
        )

    elif model_type == "patchtst":
        try:
            from src.models.patchtst_model import train_patchtst
        except Exception as exc:
            raise _optional_import_error("torch", "patchtst", exc) from exc
        return lambda df, feature_cols: train_patchtst(
            df=df,
            feature_cols=feature_cols,
            config=config,
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
