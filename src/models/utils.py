import pandas as pd


def train_test_split_time_series(df: pd.DataFrame, test_size: float = 0.2):
    split_index = int(len(df) * (1 - test_size))
    return df.iloc[:split_index], df.iloc[split_index:]