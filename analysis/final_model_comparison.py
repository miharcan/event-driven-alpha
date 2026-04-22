import argparse
from pathlib import Path
import pandas as pd


def family_label_from_file(path: Path) -> str:
    name = path.stem.replace("results_", "").replace("_summary", "")
    mapping = {
        "ridge": "Ridge",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "arima": "ARIMA",
        "lstm": "LSTM",
        "tcn": "TCN",
        "patchtst": "PatchTST",
    }
    return mapping.get(name, name.upper())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.outputs_dir)
    summary_files = sorted(out_dir.glob("results_*_summary.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No results_*_summary.csv files in {out_dir}")

    frames = []
    for f in summary_files:
        df = pd.read_csv(f)
        df["model_family"] = family_label_from_file(f)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    print("Columns:", df.columns)

    best_overall = (
        df.loc[df.groupby("asset")["best_da"].idxmax()]
        .sort_values("best_da", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== FINAL BEST MODEL PER ASSET ===")
    print(
        best_overall[
            [
                "asset",
                "model_family",
                "horizon",
                "best_model",
                "best_da",
            ]
        ]
    )

    out_path = out_dir / "final_model_comparison.csv"
    best_overall.to_csv(out_path, index=False)
    print(f"\nSaved {out_path.name}")


if __name__ == "__main__":
    main()
