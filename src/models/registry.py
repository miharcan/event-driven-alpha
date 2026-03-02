from src.models import linear, ridge
from src.models.xgboost_model import train_xgboost
from src.models.lightgmb_model import train_lightgbm


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

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")