from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from .utils import train_test_split_time_series


def train(df, feature_cols, alpha=1.0, target_col="target"):

    train_df, test_df = train_test_split_time_series(df)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return {
        "model": model,
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "y_test": y_test,
        "predictions": predictions,
    }