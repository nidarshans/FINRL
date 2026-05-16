import pandas as pd
import pandas_ta as ta
from lib.regime_detection.src.constants import VOL_WINDOW, KVO_FAST_SPAN, KVO_SLOW_SPAN, DIVERGENCE_LOOKBACK
from lib.regime_detection.src.filters.kalman import apply_kalman_to_vf

def build_signals(df):
    df            = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['Vol']     = df['Returns'].rolling(VOL_WINDOW).std()
    avg_vol       = df['Vol'].dropna().mean()

    kvo_df = ta.kvo(
        df['High'], df['Low'], df['Close'], df['Volume'],
        fast=KVO_FAST_SPAN, slow=KVO_SLOW_SPAN, signal=13
    )
    macd_df  = df.ta.macd(fast=12, slow=26, signal=9, append=False)
    raw_macd = macd_df.iloc[:, 1]

    if kvo_df is None or kvo_df.empty:
        for col in ['VF','Filtered_VF','Innovation_Z','KVO_Fast','KVO_Slow','KVO','MACD']:
            df[col] = 0.0
        return df, avg_vol

    vf_series   = kvo_df.iloc[:, 0]
    vol_aligned = df['Vol'].reindex(vf_series.index).fillna(avg_vol)
    filtered_vf, innovations = apply_kalman_to_vf(vf_series, vol_aligned, avg_vol)

    df['VF']           = vf_series
    df['Filtered_VF']  = pd.Series(filtered_vf, index=vf_series.index)
    df['Innovation_Z'] = pd.Series(innovations,  index=vf_series.index)
    df['KVO_Fast']     = kvo_df.iloc[:, 0]
    df['KVO_Slow']     = kvo_df.iloc[:, 1] if kvo_df.shape[1] > 1 else 0.0
    df['KVO']          = df['KVO_Slow']
    df['MACD']         = raw_macd
    return df, avg_vol


def detect_divergence(price_series, kvo_series, lookback=DIVERGENCE_LOOKBACK):
    price_new_high = price_series >= price_series.rolling(lookback).max()
    kvo_new_high   = kvo_series   >= kvo_series.rolling(lookback).max()
    return price_new_high & ~kvo_new_high
