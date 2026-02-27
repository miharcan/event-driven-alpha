import pandas as pd
from pathlib import Path


def load_macro_features(path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(path)

    macro_map = {
        "2y": ("Date 2Y Yield", "Value 2Y Yield"),
        "eurusd": ("Date EURUSD", "Value EURUSD"),
        "ted": ("Date TED", "Value TED"),
    }

    dfs = []

    for name, (date_col, value_col) in macro_map.items():
        temp = df_raw[[date_col, value_col]].copy()
        temp.columns = ["date", name]

        temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
        temp = temp.dropna()
        temp = temp.set_index("date").sort_index()

        # Use daily change (NOT level)
        temp[f"{name}_chg"] = temp[name].diff()

        dfs.append(temp[[f"{name}_chg"]])

    # Outer join all macro series
    df_macro = pd.concat(dfs, axis=1)

    return df_macro
