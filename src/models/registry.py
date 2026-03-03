from src.models import linear, ridge
from src.models.xgboost_model import train_xgboost
from src.models.lightgmb_model import train_lightgbm
from src.models.arima_model import train_arima
from src.models.lstm_model import train_lstm


def get_model_trainer(model_type, config):

    if model_type == "linear":
        return linear.train
    
    elif model_type == "xgboost":
        return lambda df, feature_cols: train_xgboost(
            df,
            feature_cols=feature_cols,
            config=config,
        )
    
    elif model_type == "lightgbm":
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
        return lambda df, feature_cols: train_arima(
            df=df,
            feature_cols=feature_cols,
            config=config,
        )

    elif model_type == "lstm":
        return lambda df, feature_cols: train_lstm(
            df,
            feature_cols=feature_cols,
            config=config,
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")