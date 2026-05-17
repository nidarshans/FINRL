import pandas as pd
from lib.regime_detection.src.constants import (
    BENCHMARK, WF_TRAIN_DAYS, WF_OOS_DAYS, WF_MIN_TRAIN_DAYS, WF_MODE,
    DIVERGENCE_MULT, CORR_METRIC, CORR_WINDOW,
)
from lib.regime_detection.src.features.signals import build_signals, detect_divergence
from lib.regime_detection.src.features.correlation import (
    compute_metric, build_corr_matrix, run_pca_on_corr, inject_pca_features,
    compute_corr_features,
)
from lib.regime_detection.src.models.registry import train_model, decode_model
from lib.regime_detection.src.execution.backtest import build_weight_matrix

def build_wf_windows(all_trading_dates, train_days, oos_days,
                     min_train_days, mode="rolling"):
    n       = len(all_trading_dates)
    windows = []
    oos_start = min_train_days
    while oos_start < n:
        oos_end = min(oos_start + oos_days, n)
        train_start = 0 if mode == "anchored" else max(0, oos_start - train_days)
        train_end   = oos_start
        if train_end - train_start >= min_train_days:
            windows.append((all_trading_dates[train_start], all_trading_dates[train_end - 1],
                            all_trading_dates[oos_start], all_trading_dates[oos_end - 1]))
        oos_start = oos_end
    return windows


def _run_pca_pipeline(data_dict, metric=CORR_METRIC):
    """Cross-sector PCA pipeline for a single WF window."""
    metric_df = compute_metric(data_dict, metric=metric)
    corr_dict = build_corr_matrix(metric_df, window=CORR_WINDOW)
    eigenvalues_df, _, _ = run_pca_on_corr(corr_dict)
    return eigenvalues_df


def _run_corr_pipeline(data_dict, metric=CORR_METRIC):
    """Cross-sector Correlation pipeline for a single WF window."""
    metric_df = compute_metric(data_dict, metric=metric)
    corr_dict = build_corr_matrix(metric_df, window=CORR_WINDOW)
    eigenvalues_df, _, explained_variance_df = run_pca_on_corr(corr_dict)
    corr_features_df = compute_corr_features(corr_dict, eigenvalues_df, explained_variance_df)
    return corr_features_df


def run_walk_forward(all_data, benchmark_ticker=BENCHMARK,
                     train_days=WF_TRAIN_DAYS, oos_days=WF_OOS_DAYS,
                     min_train_days=WF_MIN_TRAIN_DAYS, mode=WF_MODE,
                     metric=CORR_METRIC):
    from lib.regime_detection.src.constants import VERBOSE
    if VERBOSE:
        print(f"\n=== WALK-FORWARD ({mode.upper()}) ===")
    sector_tickers = [t for t in all_data if t != benchmark_ticker]
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in all_data.values()])))
    windows = build_wf_windows(all_dates, train_days, oos_days, min_train_days, mode)
    
    oos_decoded_chunks  = {t: [] for t in sector_tickers}
    wf_windows_meta     = []

    for i, (tr_s, tr_e, oos_s, oos_e) in enumerate(windows):
        train_data = {t: all_data[t].loc[tr_s:tr_e] for t in sector_tickers if len(all_data[t].loc[tr_s:tr_e]) >= min_train_days}
        if len(train_data) < 2: continue

        # Step 1: Cross-sector Correlation features for this training window
        corr_features_df = _run_corr_pipeline(train_data, metric=metric)

        trained = {}
        for ticker, df_raw in train_data.items():
            from lib.regime_detection.src.constants import FEATURES
            df, _ = build_signals(df_raw, corr_features_df)
            features = df[FEATURES].dropna()
            model, state_map, _, scaler = train_model(features)
            if model is not None: trained[ticker] = (df, model, state_map, scaler)

        # Step 2: Cross-sector Correlation features for OOS window
        oos_data = {t: all_data[t].loc[oos_s:oos_e] for t in sector_tickers if len(all_data[t].loc[oos_s:oos_e]) >= 10}
        oos_corr_df = None
        if oos_data:
            oos_corr_df = _run_corr_pipeline(oos_data, metric=metric)

        for ticker, (_, model, state_map, scaler) in trained.items():
            oos_raw = oos_data.get(ticker, all_data[ticker].loc[oos_s:oos_e])
            if len(oos_raw) < 10: continue

            df_oos, _ = build_signals(oos_raw, oos_corr_df)

            decoded = decode_model(model, state_map, df_oos, scaler)
            if decoded.empty: continue
            df_oos.loc[decoded.index, ["Regime", "P_Bull", "P_Bear", "Rank_Score"]] = decoded[["Regime", "P_Bull", "P_Bear", "Rank_Score"]]
            div = detect_divergence(df_oos["Close"], df_oos["KVO"])
            df_oos.loc[div, "Rank_Score"] *= DIVERGENCE_MULT
            oos_decoded_chunks[ticker].append(df_oos)

        wf_windows_meta.append((tr_s, tr_e, oos_s, oos_e))

    wf_decoded = {t: pd.concat(chunks).drop_duplicates(keep="last").sort_index() for t, chunks in oos_decoded_chunks.items() if chunks}
    wf_weights = build_weight_matrix(wf_decoded)

    oos_start_global, oos_end_global = wf_windows_meta[0][2], wf_windows_meta[-1][3]
    all_close = pd.DataFrame({t: all_data[t]["Close"] for t in wf_decoded}).loc[oos_start_global:oos_end_global].ffill()
    benchmark_df = all_data[benchmark_ticker].loc[oos_start_global:oos_end_global, "Close"].ffill()

    return wf_weights, wf_decoded, wf_windows_meta, all_close, benchmark_df

