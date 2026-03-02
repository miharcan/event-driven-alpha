from src.models import linear, ridge
from src.models.xgboost_model import train_xgboost


def get_model_trainer(model_type, config):

    if model_type == "linear":
        return linear.train
    
    elif model_type == "xgboost":
        return lambda df, feature_cols: train_xgboost(
            df,
            feature_cols=feature_cols,
            config=config,
        )
    
    elif model_type == "ridge":
        alpha = config.get("ridge_alpha", 1.0)
        return lambda df, feature_cols: ridge.train(
            df,
            feature_cols=feature_cols,
            alpha=alpha,
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")