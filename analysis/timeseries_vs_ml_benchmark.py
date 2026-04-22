import argparse
from pathlib import Path
import pandas as pd


def _load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file for {label}: {path}")
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()
    out = Path(args.outputs_dir)

    ml = _load(out / "final_model_comparison.csv", "ml-best")
    arima = _load(out / "results_arima_summary.csv", "arima")
    arima["baseline_family"] = "ARIMA"

    frames = [arima]
    lstm_path = out / "results_lstm_summary.csv"
    if lstm_path.exists():
        lstm = pd.read_csv(lstm_path)
        lstm["baseline_family"] = "LSTM"
        frames.append(lstm)

    baseline = pd.concat(frames, ignore_index=True)
    baseline = baseline.rename(
        columns={
            "best_da": "baseline_da",
            "best_model": "baseline_model",
            "ci_low_95": "baseline_ci_low_95",
            "ci_high_95": "baseline_ci_high_95",
            "p_gt_0_5": "baseline_p_gt_0_5",
            "n_test_obs": "baseline_n_test_obs",
        }
    )

    merged = ml.merge(
        baseline[
            [
                "asset",
                "horizon",
                "baseline_family",
                "baseline_model",
                "baseline_da",
                "baseline_n_test_obs",
                "baseline_ci_low_95",
                "baseline_ci_high_95",
                "baseline_p_gt_0_5",
            ]
        ],
        on=["asset", "horizon"],
        how="inner",
    )
    merged["delta_abs"] = merged["best_da"] - merged["baseline_da"]
    merged["delta_pct"] = 100.0 * merged["delta_abs"] / merged["baseline_da"]
    if {"ci_low_95", "ci_high_95", "baseline_ci_low_95", "baseline_ci_high_95"}.issubset(merged.columns):
        merged["delta_lcb95"] = merged["ci_low_95"] - merged["baseline_ci_high_95"]
        merged["strict_ci_separation"] = merged["delta_lcb95"] > 0
    else:
        merged["delta_lcb95"] = float("nan")
        merged["strict_ci_separation"] = False

    cols = [
        "asset",
        "horizon",
        "model_family",
        "best_model",
        "best_da",
        "n_test_obs",
        "baseline_family",
        "baseline_model",
        "baseline_da",
        "baseline_n_test_obs",
        "delta_abs",
        "delta_pct",
        "delta_lcb95",
        "strict_ci_separation",
    ]
    merged = merged[cols].sort_values(["delta_abs"], ascending=False).reset_index(drop=True)
    merged.to_csv(out / "timeseries_vs_ml_benchmark.csv", index=False)

    summary = (
        merged.groupby("baseline_family", as_index=False)
        .agg(
            n=("delta_abs", "size"),
            mean_delta_abs=("delta_abs", "mean"),
            median_delta_abs=("delta_abs", "median"),
            min_delta_abs=("delta_abs", "min"),
            max_delta_abs=("delta_abs", "max"),
            win_rate=("delta_abs", lambda s: (s > 0).mean()),
            strict_ci_separation_rate=("strict_ci_separation", "mean"),
        )
        .sort_values("baseline_family")
    )
    summary.to_csv(out / "timeseries_vs_ml_summary.csv", index=False)

    print("=== ML vs time-series baseline benchmark ===")
    print(merged.to_string(index=False))
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {out / 'timeseries_vs_ml_benchmark.csv'}")
    print(f"Saved: {out / 'timeseries_vs_ml_summary.csv'}")


if __name__ == "__main__":
    main()
