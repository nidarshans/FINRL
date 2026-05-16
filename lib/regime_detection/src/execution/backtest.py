import pandas as pd
import bt
from lib.regime_detection.src.constants import REBAL_FREQ, TOP_N, BULL_THRESH, BEAR_EXIT_PROB, DIVERGENCE_MULT
from lib.regime_detection.src.features.signals import build_signals, detect_divergence
from lib.regime_detection.src.models.registry import train_model, decode_model

def train_all_sectors(train_data):
    print("\n=== TRAINING PHASE ===")
    trained = {}
    for ticker, df_raw in train_data.items():
        print(f"  Training {ticker}...", end=" ")
        # Using build_signals here to get features for training
        from lib.regime_detection.src.constants import FEATURES
        df, _ = build_signals(df_raw)
        features = df[FEATURES].dropna()
        model, state_map, _, scaler = train_model(features)
        if model is None:
            print("FAILED (insufficient data)")
            continue
        trained[ticker] = (df, model, state_map, scaler)
        regime_counts = df['Regime'].value_counts().to_dict() if 'Regime' in df.columns else {}
        print(f"OK — regimes: {regime_counts}")
    print(f"  Trained {len(trained)}/{len(train_data)} sectors.")
    return trained


def decode_test_sectors(test_data, trained):
    print("\n=== DECODING TEST PERIOD ===")
    decoded_all = {}
    for ticker, (_, model, state_map, scaler) in trained.items():
        if ticker not in test_data:
            print(f"  {ticker}: no test data, skipping.")
            continue
        df_test = test_data[ticker].copy()
        df_test, avg_vol = build_signals(df_test)

        # Generic decode call
        decoded = decode_model(model, state_map, df_test, scaler)
        if decoded.empty:
            print(f"  {ticker}: decode returned empty.")
            continue

        df_test.loc[decoded.index, 'Regime']     = decoded['Regime']
        df_test.loc[decoded.index, 'P_Bull']     = decoded['P_Bull']
        df_test.loc[decoded.index, 'P_Bear']     = decoded['P_Bear']
        df_test.loc[decoded.index, 'Rank_Score'] = decoded['Rank_Score']

        div = detect_divergence(df_test['Close'], df_test['KVO'])
        df_test.loc[div, 'Rank_Score'] *= DIVERGENCE_MULT

        decoded_all[ticker] = df_test
        print(f"  {ticker}: decoded {len(decoded)} bars.")

    return decoded_all


def build_weight_matrix(decoded_all, rebal_freq=REBAL_FREQ, top_n=TOP_N,
                        bull_thresh=BULL_THRESH, bear_exit=BEAR_EXIT_PROB):
    print("\n=== BUILDING WEIGHT MATRIX ===")
    all_dates = sorted(set().union(*[set(df.index) for df in decoded_all.values()]))
    tickers   = list(decoded_all.keys())
    weights   = pd.DataFrame(0.0, index=all_dates, columns=tickers)
    rebal_dates = [d for i, d in enumerate(all_dates) if i % rebal_freq == 0]

    invested_days = 0
    for date in rebal_dates:
        scores = {}
        for ticker in tickers:
            df = decoded_all[ticker]
            if date not in df.index: continue
            row = df.loc[date]
            if row.get('P_Bull', 0) >= bull_thresh and row.get('P_Bear', 1) < bear_exit:
                scores[ticker] = row.get('Rank_Score', -999)

        if scores:
            invested_days += 1
            ranked   = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected = [t for t, _ in ranked[:top_n]]
            for t in selected: weights.loc[date, t] = 1.0 / len(selected)

    rebal_weights = weights.loc[rebal_dates].copy()
    rebal_weights = rebal_weights.reindex(all_dates).ffill()
    return rebal_weights.fillna(0.0)


def run_bt_backtest(weights_df, close_prices_df, benchmark_series, label="HMM Sector Rotation"):
    print(f"\n=== RUNNING bt BACKTEST ({label}) ===")
    close_prices    = close_prices_df.ffill().dropna(how="all")
    weights_aligned = weights_df.reindex(close_prices.index).ffill().fillna(0.0)
    
    strategy = bt.Strategy(label, [
        bt.algos.RunEveryNPeriods(REBAL_FREQ, offset=0),
        bt.algos.SelectAll(),
        bt.algos.WeighTarget(weights_aligned),
        bt.algos.Rebalance(),
    ])
    backtest_hmm = bt.Backtest(strategy, close_prices, initial_capital=100_000)

    spy_prices = benchmark_series.reindex(close_prices.index).ffill().to_frame("SPY Buy & Hold")
    bench_strategy = bt.Strategy("SPY Buy & Hold", [
        bt.algos.RunOnce(), bt.algos.SelectAll(), bt.algos.WeighEqually(), bt.algos.Rebalance(),
    ])
    backtest_spy = bt.Backtest(bench_strategy, spy_prices, initial_capital=100_000)

    result = bt.run(backtest_hmm, backtest_spy)
    print("  bt run complete.")
    return result, close_prices, weights_aligned
