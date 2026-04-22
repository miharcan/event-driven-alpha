import argparse
from pathlib import Path
import pandas as pd


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path)


def _best_significant(df: pd.DataFrame, min_test_obs: int, p_threshold: float) -> pd.DataFrame:
    df = df.copy()
    if "n_test_obs" in df.columns:
        df = df[df["n_test_obs"] >= min_test_obs]
    if "p_gt_0_5" in df.columns:
        df = df[df["p_gt_0_5"] <= p_threshold]
    if df.empty:
        return df

    best = (
        df.sort_values(["asset", "horizon", "mean_da"], ascending=[True, True, False])
        .groupby(["asset", "horizon"], as_index=False)
        .first()
    )
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--min-test-obs", type=int, default=120)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.outputs_dir)
    tables = []
    for full_path in sorted(out_dir.glob("results_*_full.csv")):
        family = full_path.stem.replace("results_", "").replace("_full", "")
        df = _load_table(full_path)
        df["model_family"] = family
        tables.append(df)

    if not tables:
        raise FileNotFoundError(f"No results_*_full.csv files found in {out_dir}")

    df_all = pd.concat(tables, ignore_index=True)
    best = _best_significant(df_all, args.min_test_obs, args.p_threshold)

    pub_path = out_dir / "publication_best_models.csv"
    if best.empty:
        pd.DataFrame(
            columns=["asset", "horizon", "model", "mean_da", "n_test_obs", "p_gt_0_5", "model_family"]
        ).to_csv(pub_path, index=False)
        print(f"No models passed filters (min_test_obs={args.min_test_obs}, p<={args.p_threshold}).")
        print(f"Saved empty template to {pub_path}")
        return

    keep_cols = [c for c in [
        "asset", "horizon", "model_family", "model", "mean_da",
        "n_test_obs", "ci_low_95", "ci_high_95", "p_gt_0_5"
    ] if c in best.columns]
    best = best[keep_cols].sort_values(["mean_da"], ascending=False)
    best.to_csv(pub_path, index=False)
    print(best.to_string(index=False))
    print(f"\nSaved: {pub_path}")


if __name__ == "__main__":
    main()
