import numpy as np
import pandas as pd
import yfinance as yf
from lib.regime_detection.src.constants import BENCHMARK

def download_all(tickers, start, end):
    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    dates = pd.bdate_range(start=start, end=end)
    data  = {}

    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
        live_ok = False
        for ticker in tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw.xs(ticker, level=1, axis=1).copy()
                else:
                    df = raw.copy()
                df = df.dropna(subset=['Close'])
                if len(df) > 60:
                    data[ticker] = df
                    live_ok = True
            except Exception:
                pass
        if live_ok:
            print(f"  Loaded {len(data)}/{len(tickers)} tickers from yfinance.")
            return data
    except Exception as e:
        print(f"  yfinance unavailable ({e})")
    return data


def _slice_data(data_dict, start, end, min_bars=60):
    out = {}
    for t, df in data_dict.items():
        sliced = df.loc[start:end]
        if len(sliced) > min_bars:
            out[t] = sliced
    return out
