import argparse
from pathlib import Path
import pandas as pd


def _load_setting(outputs_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(outputs_dir.glob("results_*_full.csv")):
        fam = f.stem.replace("results_", "").replace("_full", "")
        df = pd.read_csv(f)
        df["model_family"] = fam
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _family_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "model_family",
                "configs",
                "mean_da_all",
                "median_da_all",
                "sig_share_all",
                "mean_da_best",
                "median_da_best",
                "sig_share_best",
                "n_asset_h",
            ]
        )
    all_cfg = (
        df.groupby("model_family", as_index=False)
        .agg(
            configs=("mean_da", "size"),
            mean_da_all=("mean_da", "mean"),
            median_da_all=("mean_da", "median"),
            sig_share_all=("p_gt_0_5", lambda s: (s <= 0.05).mean()),
        )
    )

    best_ah = (
        df.sort_values("mean_da", ascending=False)
        .groupby(["asset", "horizon", "model_family"], as_index=False)
        .first()
    )
    best_cfg = (
        best_ah.groupby("model_family", as_index=False)
        .agg(
            mean_da_best=("mean_da", "mean"),
            median_da_best=("mean_da", "median"),
            sig_share_best=("p_gt_0_5", lambda s: (s <= 0.05).mean()),
            n_asset_h=("mean_da", "size"),
        )
    )

    out = all_cfg.merge(best_cfg, on="model_family", how="left")
    return out.sort_values("model_family").reset_index(drop=True)


def _format_family(f: str) -> str:
    mapping = {
        "ridge": "Ridge",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "arima": "ARIMA",
        "lstm": "LSTM",
        "tcn": "TCN",
        "patchtst": "PatchTST",
    }
    return mapping.get(f, f.upper())


def _to_latex(strict_df: pd.DataFrame, max_df: pd.DataFrame) -> str:
    cols = sorted(set(strict_df.get("model_family", [])).union(set(max_df.get("model_family", []))))
    s = strict_df.set_index("model_family")
    m = max_df.set_index("model_family")

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Robustness of model-family performance under strict vs max-data specifications.}")
    lines.append("\\label{tab:robustness_strict_vs_maxdata}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Setting & Family & Configs & Mean DA (all) & Mean DA (best per asset-horizon) & Share($p\\le0.05$) (all) \\\\")
    lines.append("\\midrule")
    for setting_name, df_set in [("Strict", s), ("Max-data", m)]:
        first = True
        for fam in cols:
            if fam not in df_set.index:
                continue
            r = df_set.loc[fam]
            setting_cell = setting_name if first else ""
            first = False
            lines.append(
                f"{setting_cell} & {_format_family(fam)} & {int(r['configs'])} & "
                f"{r['mean_da_all']:.4f} & {r['mean_da_best']:.4f} & {r['sig_share_all']:.3f} \\\\"
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", default="outputs")
    parser.add_argument("--maxdata-dir", default="outputs_maxdata")
    parser.add_argument("--out-tex", default="analysis/robustness_table.tex")
    args = parser.parse_args()

    strict_dir = Path(args.strict_dir)
    maxdata_dir = Path(args.maxdata_dir)
    out_tex = Path(args.out_tex)

    strict = _load_setting(strict_dir)
    maxdata = _load_setting(maxdata_dir)
    if strict.empty and maxdata.empty:
        raise FileNotFoundError("No results_*_full.csv found in either strict or max-data outputs dirs.")

    strict_sum = _family_summary(strict)
    maxdata_sum = _family_summary(maxdata)

    strict_sum["setting"] = "strict"
    maxdata_sum["setting"] = "maxdata"
    combined = pd.concat([strict_sum, maxdata_sum], ignore_index=True)
    combined.to_csv(out_tex.with_suffix(".csv"), index=False)

    latex = _to_latex(strict_sum, maxdata_sum)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(latex)

    print("=== Strict summary ===")
    print(strict_sum.to_string(index=False))
    print("\n=== Max-data summary ===")
    print(maxdata_sum.to_string(index=False))
    print(f"\nSaved: {out_tex}")
    print(f"Saved: {out_tex.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
