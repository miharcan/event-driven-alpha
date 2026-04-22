import logging
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import numpy as np
import re
import math
from time import perf_counter

from src.features.price_features import (
    compute_log_returns,
    compute_rolling_volatility,
    add_lag_features,
    add_forward_return,
    add_volatility_regime
)
from src.data.news_loader import load_news_json
from src.features.news_features import build_daily_news_features
from src.data.alignment import align_price_and_news
from src.models.registry import get_model_trainer
from src.features.embeddings import NewsEmbedder
from src.data.macro_loader import load_macro_features
from src.models.ridge import train_regime_specific
from src.models.baseline_regression import (
    train_baseline_regression,
    directional_accuracy,
    run_regime_gated_model
)


logger = logging.getLogger(__name__)


def run_pipeline(config: dict):

    logger.info("=== STARTING PIPELINE ===")

    # ---------------------------
    # 1. Configuration
    # ---------------------------

    macro_categories = [
        "POLITICS",
        "BUSINESS",
        "WORLD NEWS",
        "U.S. NEWS",
        "THE WORLDPOST",
        "WORLDPOST",
    ]

    macro_keywords = [
        # Monetary
        "inflation", "federal reserve", "fed", "interest rate",
        "rate hike", "rate cut", "yield", "bond", "real rate",
        # FX
        "dollar", "usd", "currency", "forex",
        # Energy
        "oil", "crude", "opec", "refinery", "pipeline",
        "energy", "natural gas",
        # Geopolitics
        "sanction", "russia", "china", "ukraine",
        "war", "military", "conflict", "trade",
        # Macro risk
        "recession", "debt", "default", "gdp",
        "supply", "shortage", "export", "import",
    ]

    pattern = re.compile("|".join(macro_keywords), re.IGNORECASE)

    asset_map = {
        "Gold": ("Date Gold", "Value Gold"),
        "Silver": ("Date Silver", "Value Silver"),
        "Platinum": ("Date Platinum", "Value Platinum"),
        "Copper": ("Date Copper", "Value Copper"),
        "Crude": ("Date Crude Oil", "Value Crude Oil"),
        "HeatingOil": ("Date Heating Oil", "Value Heating Oil"),
        "Corn": ("Date Corn", "Value Corn"),
        "Coffee": ("Date Coffee", "Value Coffee"),
    }

    trainer = get_model_trainer(config["model"]["model_type"], config)

    model_type = config["model"]["model_type"]
    publication_cfg = config.get("publication", {})
    news_filter_cfg = config.get("news_filter", {})


    # ---------------------------
    # 2. Load Price File (Raw)
    # ---------------------------

    price_path = config["price_path"]

    if price_path.endswith(".csv"):
        df_raw = pd.read_csv(price_path)
    else:
        df_raw = pd.read_excel(price_path)

    # ---------------------------
    # 3. Load & Prepare News
    # ---------------------------

    logger.info("Loading news data...")

    df_news_raw = load_news_json(config["news_path"])
    logger.info("Headlines raw: %d", len(df_news_raw))

    use_category_filter = news_filter_cfg.get("use_category_filter", True)
    categories = news_filter_cfg.get("categories", macro_categories)
    if use_category_filter:
        df_news_raw = df_news_raw[df_news_raw["category"].isin(categories)]
        logger.info("Headlines after category filter: %d", len(df_news_raw))
    else:
        logger.info("Category filter disabled.")

    use_keyword_filter = news_filter_cfg.get("use_keyword_filter", True)
    keywords = news_filter_cfg.get("keywords", macro_keywords)
    if use_keyword_filter and keywords:
        pattern = re.compile("|".join(keywords), re.IGNORECASE)
        df_news_raw = df_news_raw[df_news_raw["headline"].str.contains(pattern, na=False)]
        logger.info("Headlines after keyword filter: %d", len(df_news_raw))
    else:
        logger.info("Keyword filter disabled.")

    max_headlines = news_filter_cfg.get("max_headlines")
    if max_headlines is not None:
        max_headlines = int(max_headlines)
        if len(df_news_raw) > max_headlines:
            df_news_raw = df_news_raw.iloc[:max_headlines].copy()
            logger.info("Applied max_headlines=%d; kept %d", max_headlines, len(df_news_raw))

    if len(df_news_raw) < 1000:
        logger.warning("Very few headlines after filtering.")

    # Structured news features
    df_news_structured = build_daily_news_features(df_news_raw)

    # Embeddings
    embedder = NewsEmbedder(
        model_name=config.get("embedding_model", "all-MiniLM-L6-v2")
    )

    embed_start = perf_counter()
    headline_embeddings = embedder.embed_headlines(df_news_raw)
    daily_embeddings = embedder.aggregate_daily(
        df_news_raw,
        headline_embeddings,
    )

    daily_embeddings.columns = [
        f"emb_{i}" for i in range(daily_embeddings.shape[1])
    ]

    df_news = df_news_structured.join(
        daily_embeddings,
        how="left"
    ).fillna(0)
    logger.info(
        "Prepared global news feature matrix with %d rows in %.2fs",
        len(df_news),
        perf_counter() - embed_start,
    )

    # ---------------------------
    # 4. Load Macro Features
    # ---------------------------

    logger.info("Loading macro features...")
    df_macro = load_macro_features(config["macro_path"])

    # ---------------------------
    # 5. Multi-Asset Loop
    # ---------------------------
    tracker = ResultTracker()
    for asset, (date_col, price_col) in asset_map.items():

        logger.info(f"\n--- Running {asset} ---")

        if date_col not in df_raw.columns:
            logger.info(f"{asset}: Missing columns in raw file.")
            continue

        # --- Extract asset series
        df_price = df_raw[[date_col, price_col]].copy()
        df_price.columns = ["date", "price"]

        df_price["date"] = pd.to_datetime(df_price["date"], errors="coerce")
        df_price = df_price.dropna()
        df_price = df_price.set_index("date").sort_index()

        # --- Price features
        df_price = compute_log_returns(df_price)
        df_price = compute_rolling_volatility(df_price)
        df_price = add_lag_features(df_price, "log_return", 3)

        # Asset-specific news is independent of horizon: compute once per asset.
        asset_kw = config.get("asset_keywords", {}).get(asset, [])

        if asset_kw:
            pattern_asset = re.compile("|".join(asset_kw), re.IGNORECASE)
            df_asset_news_raw = df_news_raw[
                df_news_raw["headline"].str.contains(pattern_asset, na=False)
            ].copy()
            logger.info("%s: Asset-specific headlines = %d", asset, len(df_asset_news_raw))
        else:
            df_asset_news_raw = df_news_raw.copy()
            logger.info("%s: No asset keywords configured; using global news set.", asset)

        asset_news_start = perf_counter()
        df_asset_structured = build_daily_news_features(df_asset_news_raw)
        headline_emb_asset = embedder.embed_headlines(df_asset_news_raw)
        daily_emb_asset = embedder.aggregate_daily(
            df_asset_news_raw,
            headline_emb_asset,
        )
        daily_emb_asset.columns = [
            f"emb_{i}" for i in range(daily_emb_asset.shape[1])
        ]
        df_news_asset = df_asset_structured.join(
            daily_emb_asset,
            how="left"
        ).fillna(0)
        logger.info(
            "%s: Prepared asset news feature matrix with %d rows in %.2fs",
            asset,
            len(df_news_asset),
            perf_counter() - asset_news_start,
        )

        horizons = config.get("horizons", [5])
        for horizon in horizons:

            logger.info(f"\n===== Testing Horizon: {horizon} days =====")

            df_price_h = df_price.copy()

            df_price_h = add_forward_return(df_price_h, horizon=horizon)
            df_price_h = add_volatility_regime(df_price_h)

            # --- Merge macro
            df_price_h = df_price_h.join(df_macro, how="left")
            df_price_h = df_price_h.ffill()

            df_model_global = align_price_and_news(df_price_h, df_news)

            df_model_asset = align_price_and_news(df_price_h, df_news_asset)

            logger.info(f"{asset}: Global daily news rows = {len(df_news)}")
            logger.info(f"{asset}: Asset daily news rows = {len(df_news_asset)}")
            logger.info(
                "%s | H%s | aligned rows: global=%d asset=%d",
                asset,
                horizon,
                len(df_model_global),
                len(df_model_asset),
            )

            for name, df_tmp in [("global", df_model_global), ("asset", df_model_asset)]:
                df_tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
                df_tmp.dropna(inplace=True)

            smoke_frac = publication_cfg.get("smoke_sample_fraction")
            smoke_min_rows = publication_cfg.get("smoke_min_rows", 220)
            if smoke_frac is not None:
                df_model_global = downsample_chrono(df_model_global, smoke_frac, smoke_min_rows)
                df_model_asset = downsample_chrono(df_model_asset, smoke_frac, smoke_min_rows)
                logger.info(
                    "%s | H%s | smoke sampling active: frac=%s global=%d asset=%d",
                    asset,
                    horizon,
                    smoke_frac,
                    len(df_model_global),
                    len(df_model_asset),
                )

            # -------------------------------------------------
            # Time-series baseline (ARIMA | LSTM)
            # Run once per asset + horizon
            # -------------------------------------------------
            if model_type in ["arima", "lstm", "tcn", "patchtst"]:

                df_ts = df_model_global.copy()

                if len(df_ts) < 200:
                    logger.info(f"{asset}: Not enough data for {model_type.upper()}.")
                    continue

                res = normalize_result(
                    trainer(df_ts, feature_cols=[])
                )

                model_name = f"{model_type.upper()}_Full"

                tracker.add(
                    asset,
                    horizon,
                    model_name,
                    res["mean_da"],
                    res["fold_das"],
                    n_test_obs=res.get("n_test_obs"),
                    eval_split="cv",
                )

                if res["mean_da"] is not None:
                    logger.info(
                        f"{asset} | H{horizon} | {model_name} = {res['mean_da']:.4f}"
                    )
                else:
                    logger.info(
                        f"{asset} | H{horizon} | {model_name} = None (no valid folds)"
                    )

                continue  # Skip feature/dataset loops entirely

            for news_version, df_model in [
                ("GlobalNews", df_model_global),
                ("AssetNews", df_model_asset)
            ]:
                if len(df_model) < 200:
                    logger.info(f"{asset}: Not enough overlapping data.")
                    continue

                datasets = build_attention_datasets(
                    df_model=df_model,
                    publication_cfg=publication_cfg,
                )

                # ---------------------------
                # Dataset loop: Full vs HighAttention
                # ---------------------------
                for dataset_name, df_use in datasets:
                    dataset_start = perf_counter()

                    df_use = df_use.copy()
                    logger.info(
                        "%s | H%s | %s | %s | starting experiments on %d rows",
                        asset,
                        horizon,
                        news_version,
                        dataset_name,
                        len(df_use),
                    )

                    feature_blocks = build_feature_blocks(
                        df_use=df_use,
                        df_macro=df_macro,
                        config=config,
                        publication_cfg=publication_cfg,
                    )
                    if feature_blocks is None:
                        logger.warning(f"{asset} | H{horizon} | {news_version} | {dataset_name}: No raw embedding cols.")
                        continue
                    df_use = feature_blocks["df_use"]
                    price_cols = feature_blocks["price_cols"]
                    macro_features = feature_blocks["macro_features"]
                    news_features = feature_blocks["news_features"]
                    all_features = feature_blocks["all_features"]

                    # ---------------------------
                    # Experiments
                    # ---------------------------
                    results = {}

                    if price_cols:
                        t0 = perf_counter()
                        res_price = normalize_result(trainer(df_use, feature_cols=price_cols))
                        tracker.add(
                            asset,
                            horizon,
                            f"{news_version}__{dataset_name}_Price",
                            res_price["mean_da"],
                            res_price["fold_das"],
                            n_test_obs=res_price.get("n_test_obs"),
                            eval_split="cv",
                        )
                        results["Price"] = res_price
                        logger.info(
                            "%s | H%s | %s | %s | Price done in %.2fs (DA=%s)",
                            asset,
                            horizon,
                            news_version,
                            dataset_name,
                            perf_counter() - t0,
                            f"{res_price['mean_da']:.4f}" if res_price["mean_da"] is not None else "None",
                        )

                    if macro_features:
                        t0 = perf_counter()
                        res_macro = normalize_result(trainer(df_use, feature_cols=macro_features))
                        tracker.add(
                            asset,
                            horizon,
                            f"{news_version}__{dataset_name}_Macro",
                            res_macro["mean_da"],
                            res_macro["fold_das"],
                            n_test_obs=res_macro.get("n_test_obs"),
                            eval_split="cv",
                        )
                        results["Macro"] = res_macro
                        logger.info(
                            "%s | H%s | %s | %s | Macro done in %.2fs (DA=%s)",
                            asset,
                            horizon,
                            news_version,
                            dataset_name,
                            perf_counter() - t0,
                            f"{res_macro['mean_da']:.4f}" if res_macro["mean_da"] is not None else "None",
                        )

                    if news_features:
                        t0 = perf_counter()
                        res_news = normalize_result(trainer(df_use, feature_cols=news_features))
                        tracker.add(
                            asset,
                            horizon,
                            f"{news_version}__{dataset_name}_News",
                            res_news["mean_da"],
                            res_news["fold_das"],
                            n_test_obs=res_news.get("n_test_obs"),
                            eval_split="cv",
                        )
                        results["News"] = res_news
                        logger.info(
                            "%s | H%s | %s | %s | News done in %.2fs (DA=%s)",
                            asset,
                            horizon,
                            news_version,
                            dataset_name,
                            perf_counter() - t0,
                            f"{res_news['mean_da']:.4f}" if res_news["mean_da"] is not None else "None",
                        )

                    if price_cols and macro_features:
                        t0 = perf_counter()
                        res_price_macro = normalize_result(trainer(df_use, feature_cols=price_cols + macro_features))
                        tracker.add(
                            asset,
                            horizon,
                            f"{news_version}__{dataset_name}_Price+Macro",
                            res_price_macro["mean_da"],
                            res_price_macro["fold_das"],
                            n_test_obs=res_price_macro.get("n_test_obs"),
                            eval_split="cv",
                        )
                        results["Price+Macro"] = res_price_macro
                        logger.info(
                            "%s | H%s | %s | %s | Price+Macro done in %.2fs (DA=%s)",
                            asset,
                            horizon,
                            news_version,
                            dataset_name,
                            perf_counter() - t0,
                            f"{res_price_macro['mean_da']:.4f}" if res_price_macro["mean_da"] is not None else "None",
                        )

                    if price_cols and news_features:
                        t0 = perf_counter()
                        res_price_news = normalize_result(trainer(df_use, feature_cols=price_cols + news_features))
                        tracker.add(
                            asset,
                            horizon,
                            f"{news_version}__{dataset_name}_Price+News",
                            res_price_news["mean_da"],
                            res_price_news["fold_das"],
                            n_test_obs=res_price_news.get("n_test_obs"),
                            eval_split="cv",
                        )
                        results["Price+News"] = res_price_news
                        logger.info(
                            "%s | H%s | %s | %s | Price+News done in %.2fs (DA=%s)",
                            asset,
                            horizon,
                            news_version,
                            dataset_name,
                            perf_counter() - t0,
                            f"{res_price_news['mean_da']:.4f}" if res_price_news["mean_da"] is not None else "None",
                        )

                    if macro_features and news_features:
                        t0 = perf_counter()
                        res_macro_news = normalize_result(trainer(df_use, feature_cols=macro_features + news_features))
                        tracker.add(
                            asset,
                            horizon,
                            f"{news_version}__{dataset_name}_Macro+News",
                            res_macro_news["mean_da"],
                            res_macro_news["fold_das"],
                            n_test_obs=res_macro_news.get("n_test_obs"),
                            eval_split="cv",
                        )
                        results["Macro+News"] = res_macro_news
                        logger.info(
                            "%s | H%s | %s | %s | Macro+News done in %.2fs (DA=%s)",
                            asset,
                            horizon,
                            news_version,
                            dataset_name,
                            perf_counter() - t0,
                            f"{res_macro_news['mean_da']:.4f}" if res_macro_news["mean_da"] is not None else "None",
                        )

                    if all_features:
                        t0 = perf_counter()
                        res_all = normalize_result(trainer(df_use, feature_cols=all_features))
                        tracker.add(
                            asset,
                            horizon,
                            f"{news_version}__{dataset_name}_All",
                            res_all["mean_da"],
                            res_all["fold_das"],
                            n_test_obs=res_all.get("n_test_obs"),
                            eval_split="cv",
                        )
                        results["All"] = res_all
                        logger.info(
                            "%s | H%s | %s | %s | All done in %.2fs (DA=%s)",
                            asset,
                            horizon,
                            news_version,
                            dataset_name,
                            perf_counter() - t0,
                            f"{res_all['mean_da']:.4f}" if res_all["mean_da"] is not None else "None",
                        )

                    # Regime-specific (if your function returns mean_da/fold_das; if not, wrap with normalize_result or adjust)
                    if "vol_regime_high" in df_use.columns and all_features:
                        t0 = perf_counter()
                        res_regime = train_regime_specific(
                            df_use,
                            config,
                            feature_cols=all_features + ["vol_regime_high"],
                            alpha=config.get("ridge_alpha", 1.0)
                        )
                        res_regime = normalize_result(res_regime)
                        tracker.add(
                            asset,
                            horizon,
                            f"{news_version}__{dataset_name}_RegimeSpecific",
                            res_regime["mean_da"],
                            res_regime["fold_das"],
                            n_test_obs=res_regime.get("n_test_obs"),
                            eval_split="cv",
                        )
                        logger.info(
                            "%s | H%s | %s | %s | RegimeSpecific done in %.2fs (DA=%s)",
                            asset,
                            horizon,
                            news_version,
                            dataset_name,
                            perf_counter() - t0,
                            f"{res_regime['mean_da']:.4f}" if res_regime["mean_da"] is not None else "None",
                        )

                    # All + RegimeInteraction
                    if all_features:
                        t0 = perf_counter()
                        res_all_inter = normalize_result(run_all_with_regime_interaction(
                            df=df_use,
                            base_feature_cols=all_features,
                            config=config
                        ))
                        tracker.add(
                            asset,
                            horizon,
                            f"{news_version}__{dataset_name}_All+RegimeInteraction",
                            res_all_inter["mean_da"],
                            res_all_inter["fold_das"],
                            n_test_obs=res_all_inter.get("n_test_obs"),
                            eval_split="cv",
                        )
                        logger.info(
                            "%s | H%s | %s | %s | All+RegimeInteraction done in %.2fs (DA=%s)",
                            asset,
                            horizon,
                            news_version,
                            dataset_name,
                            perf_counter() - t0,
                            f"{res_all_inter['mean_da']:.4f}" if res_all_inter["mean_da"] is not None else "None",
                        )

                        if "All" in results:
                            delta = res_all_inter["mean_da"] - results["All"]["mean_da"]
                            logger.info(f"{asset} | H{horizon} | {news_version} | {dataset_name} | RegimeInteraction Δ vs All: {delta:+.4f}")

                    # Regime gated
                    if all_features:
                        t0 = perf_counter()
                        res_gated = normalize_result(run_regime_gated_model(
                            df=df_use,
                            target_col="fwd_return",
                            feature_cols=all_features,
                            config=config,
                        ))
                        tracker.add(
                            asset,
                            horizon,
                            f"{news_version}__{dataset_name}_All+RegimeGated",
                            res_gated["mean_da"],
                            res_gated["fold_das"],
                            n_test_obs=res_gated.get("n_test_obs"),
                            eval_split="cv",
                        )
                        logger.info(
                            "%s | H%s | %s | %s | All+RegimeGated done in %.2fs (DA=%s)",
                            asset,
                            horizon,
                            news_version,
                            dataset_name,
                            perf_counter() - t0,
                            f"{res_gated['mean_da']:.4f}" if res_gated["mean_da"] is not None else "None",
                        )

                    # Best for this asset+horizon across everything recorded so far
                    best_row = tracker.best_for(asset, horizon)
                    logger.info(f"{asset} | H{horizon} | {news_version} | {dataset_name} | Best={best_row['model']} ({best_row['mean_da']:.4f})")
                    logger.info(
                        "%s | H%s | %s | %s | finished dataset in %.2fs",
                        asset,
                        horizon,
                        news_version,
                        dataset_name,
                        perf_counter() - dataset_start,
                    )
           
    model_type = config["model"]["model_type"]

    output_dir = config.get("output_dir", "outputs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    full_path = str(Path(output_dir) / f"results_{model_type}_full.csv")
    summary_path = str(Path(output_dir) / f"results_{model_type}_summary.csv")

    tracker.save(full_path)
    tracker.save_summary(summary_path)

    logger.info(f"Results saved to {full_path} and {summary_path}")


# ---------------------------
# Logging Results
# ---------------------------

def log_result(name, res, asset):
    da = directional_accuracy(
        res["y_test"],
        res["predictions"]
    )
    logger.info(f"{asset} {name} DA: {da:.4f}")

    if "fold_das" in res:
        logger.info(f"{asset} {name} Fold DAs: {res['fold_das']}")
        logger.info(
            f"{asset} {name} Mean Fold DA: "
            f"{res['mean_da']:.4f}"
        )


def collect_metrics(res, experiment_name, feature_set):
    return {
        "experiment_name": experiment_name,
        "feature_set": feature_set,
        "metrics": {
            "mse": res["mse"],
            "r2": res["r2"],
            "directional_accuracy": directional_accuracy(
                res["y_test"], res["predictions"]
            ),
        },
    }


def persist_outputs(results, config):

    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_version": config.get("model_version"),
        "model_type": config.get("model_type"),
        "timestamp": datetime.utcnow().isoformat(),
        "experiments": results,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(artifact, f, indent=4)

    return output_dir


def build_attention_datasets(df_model: pd.DataFrame, publication_cfg: dict):
    datasets = [("Full", df_model)]
    if "article_count" not in df_model.columns:
        return datasets

    percentiles = publication_cfg.get("attention_percentiles", [0.5, 0.6, 0.7, 0.8, 0.9])
    train_fraction = publication_cfg.get("train_fraction", 0.6)
    min_rows = publication_cfg.get("min_dataset_rows", 100)

    anchor_end = max(1, int(len(df_model) * train_fraction))
    anchor = df_model.iloc[:anchor_end]
    if len(anchor) < 20:
        logger.warning("Attention filtering skipped: not enough anchor rows.")
        return datasets

    for q in percentiles:
        threshold = anchor["article_count"].quantile(q)
        df_filtered = df_model[df_model["article_count"] > threshold].copy()
        if len(df_filtered) > min_rows:
            datasets.append((f"Top{int(q * 100)}", df_filtered))
    return datasets


def downsample_chrono(df: pd.DataFrame, sample_fraction: float, min_rows: int = 220):
    if sample_fraction is None:
        return df
    try:
        frac = float(sample_fraction)
    except (TypeError, ValueError):
        return df
    if frac <= 0 or frac >= 1:
        return df
    n = len(df)
    if n == 0:
        return df

    target_n = max(min_rows, int(n * frac))
    target_n = min(target_n, n)
    if target_n >= n:
        return df

    # Keep chronological coverage while reducing rows.
    idx = np.linspace(0, n - 1, num=target_n, dtype=int)
    return df.iloc[idx].copy()


def build_feature_blocks(df_use: pd.DataFrame, df_macro: pd.DataFrame, config: dict, publication_cfg: dict):
    df_use = df_use.copy()
    news_features = [c for c in df_use.columns if c.startswith("emb_")]
    if not news_features:
        return None

    macro_features = [c for c in df_use.columns if c in df_macro.columns]
    price_cols = [c for c in config["price_features"] if c in df_use.columns]
    interaction_cols = []

    if publication_cfg.get("enable_news_macro_interactions", True) and macro_features:
        df_use["emb_mean_raw"] = df_use[news_features].mean(axis=1)
        for macro in macro_features:
            col_name = f"emb_mean_raw_x_{macro}"
            df_use[col_name] = df_use["emb_mean_raw"] * df_use[macro]
            interaction_cols.append(col_name)

    all_features = price_cols + macro_features + news_features + interaction_cols
    return {
        "df_use": df_use,
        "price_cols": price_cols,
        "macro_features": macro_features,
        "news_features": news_features,
        "all_features": all_features,
    }


def run_regime_specific_model(
    X_train, y_train, regimes_train,
    X_test, y_test, regimes_test,
    model_class
):
    """
    Trains separate models for high-vol and low-vol regimes.
    Returns:
        high_da, low_da
    """

    # Split training data
    X_train_high = X_train[regimes_train == 1]
    y_train_high = y_train[regimes_train == 1]

    X_train_low = X_train[regimes_train == 0]
    y_train_low = y_train[regimes_train == 0]

    # Initialize models
    model_high = model_class()
    model_low = model_class()

    high_da = None
    low_da = None

    # Train + test high regime
    if len(X_train_high) > 5:
        model_high.fit(X_train_high, y_train_high)
        X_test_high = X_test[regimes_test == 1]
        y_test_high = y_test[regimes_test == 1]

        if len(X_test_high) > 0:
            preds_high = model_high.predict(X_test_high)
            high_da = (preds_high == y_test_high).mean()

    # Train + test low regime
    if len(X_train_low) > 5:
        model_low.fit(X_train_low, y_train_low)
        X_test_low = X_test[regimes_test == 0]
        y_test_low = y_test[regimes_test == 0]

        if len(X_test_low) > 0:
            preds_low = model_low.predict(X_test_low)
            low_da = (preds_low == y_test_low).mean()

    return high_da, low_da


def run_all_with_regime_interaction(df, base_feature_cols, config):
    """
    Unified model with continuous volatility interaction.
    Computes volatility proxy from log returns to avoid missing column issues.
    """

    df_int = df.copy()

    if "log_return" not in df_int.columns:
        raise ValueError("log_return column required for volatility interaction.")

    # --- compute rolling volatility locally (safe)
    vol = df_int["log_return"].rolling(10).std()

    vol = vol.bfill()

    # normalize
    vol_z = (vol - vol.mean()) / (vol.std() + 1e-8)

    interaction_cols = []

    for col in base_feature_cols:
        if col != "vol_regime_high":
            new_col = f"{col}_x_vol"
            df_int[new_col] = df_int[col] * vol_z
            interaction_cols.append(new_col)

    feature_cols_extended = base_feature_cols + interaction_cols

    return train_baseline_regression(
        df=df_int,
        target_col="fwd_return",
        feature_cols=feature_cols_extended,
        config=config
    )


def compute_wilson_ci(mean_da, n_test_obs, z=1.96):
    if mean_da is None or n_test_obs is None or n_test_obs <= 0:
        return np.nan, np.nan
    p = float(mean_da)
    n = float(n_test_obs)
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) / n) + ((z ** 2) / (4 * (n ** 2))))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_one_sided_p_value(mean_da, n_test_obs):
    if mean_da is None or n_test_obs is None or n_test_obs <= 0:
        return np.nan
    p = float(mean_da)
    n = float(n_test_obs)
    se = math.sqrt(0.25 / n)
    if se == 0:
        return np.nan
    z = (p - 0.5) / se
    return 0.5 * math.erfc(z / math.sqrt(2.0))


class ResultTracker:
    def __init__(self):
        self.records = []

    def add(
        self,
        asset,
        horizon,
        model_name,
        mean_da,
        fold_das=None,
        regime_variant=None,
        n_test_obs=None,
        eval_split="cv",
    ):
        ci_low, ci_high = compute_wilson_ci(mean_da, n_test_obs)
        p_one_sided = compute_one_sided_p_value(mean_da, n_test_obs)
        self.records.append({
            "asset": asset,
            "horizon": horizon,
            "model": model_name,
            "mean_da": float(mean_da) if mean_da is not None else np.nan,
            "fold_das": fold_das,
            "regime_variant": regime_variant,
            "n_test_obs": int(n_test_obs) if n_test_obs is not None else np.nan,
            "ci_low_95": ci_low,
            "ci_high_95": ci_high,
            "p_gt_0_5": p_one_sided,
            "eval_split": eval_split,
        })

    def best_for(self, asset, horizon):
        df = pd.DataFrame(self.records)
        df = df[(df["asset"] == asset) & (df["horizon"] == horizon)]
        df = df.dropna(subset=["mean_da"])
        return df.loc[df["mean_da"].idxmax()]

    def summary_by_asset(self):
        df = pd.DataFrame(self.records)

        summary_rows = []

        for asset in df["asset"].unique():
            df_asset = df[df["asset"] == asset]

            for horizon in sorted(df_asset["horizon"].unique()):
                df_h = df_asset[df_asset["horizon"] == horizon]

                # best_row = df_h.loc[df_h["mean_da"].idxmax()]
                df_h_valid = df_h.dropna(subset=["mean_da"])

                if df_h_valid.empty:
                    continue

                best_row = df_h_valid.loc[df_h_valid["mean_da"].idxmax()]

                summary_rows.append({
                    "asset": asset,
                    "horizon": horizon,
                    "best_model": best_row["model"],
                    "best_da": best_row["mean_da"],
                    "n_test_obs": best_row.get("n_test_obs"),
                    "ci_low_95": best_row.get("ci_low_95"),
                    "ci_high_95": best_row.get("ci_high_95"),
                    "p_gt_0_5": best_row.get("p_gt_0_5"),
                    "eval_split": best_row.get("eval_split"),
                })

        return pd.DataFrame(summary_rows)

    def save(self, path="results_full.csv"):
        df = pd.DataFrame(self.records)
        df.to_csv(path, index=False)

    def save_summary(self, path="results_summary.csv"):
        summary_df = self.summary_by_asset()
        summary_df.to_csv(path, index=False)



def normalize_result(res):
    if "mean_da" in res:
        out = res
    elif "mean_fold_da" in res:
        res["mean_da"] = res["mean_fold_da"]
        out = res
    else:
        raise ValueError("Trainer result missing mean_da / mean_fold_da")

    if "n_test_obs" not in out:
        if "y_test" in out:
            out["n_test_obs"] = len(out["y_test"])
        elif "fold_sizes" in out:
            out["n_test_obs"] = int(sum(out["fold_sizes"]))
        else:
            out["n_test_obs"] = None
    return out
