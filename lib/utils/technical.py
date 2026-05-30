import polars as pl
import pandas as pd
import pandas_ta_classic as pta


def utils_add_klinger_and_macd_signals(df: pl.DataFrame) -> pl.DataFrame:
    """
    Engineers MACD and Klinger Volume Oscillator signals.

    Returns:
        MACD
        MACD_Signal
        KVO
        Klinger_Signal
        Signal_KVO_MACD_Bullish
    """

    feature_frames = []

    for group in df.partition_by("Ticker", maintain_order=True):

        pdf = group.to_pandas()

        # -------------------------
        # MACD
        # -------------------------
        macd = pta.macd(
            pdf["Close"],
            fast=12,
            slow=26,
            signal=9,
        )

        macd_col = macd.filter(regex=r"^MACD_").columns[0]
        signal_col = macd.filter(regex=r"^MACDs_").columns[0]

        pdf["MACD"] = macd[macd_col]
        pdf["MACD_Signal"] = macd[signal_col]

        # -------------------------
        # Klinger Volume Oscillator
        # -------------------------
        kvo = pta.kvo(
            high=pdf["High"],
            low=pdf["Low"],
            close=pdf["Close"],
            volume=pdf["Volume"],
            fast=34,
            slow=55,
            signal=13,
        )

        kvo_col = kvo.filter(regex=r"^KVO_").columns[0]
        kvo_signal_col = kvo.filter(regex=r"^KVOs_").columns[0]

        pdf["KVO"] = kvo[kvo_col]
        pdf["Klinger_Signal"] = kvo[kvo_signal_col]

        # -------------------------
        # Combined Bullish Signal
        # -------------------------
        pdf["Signal_KVO_MACD_Bullish"] = (
            (pdf["MACD"] > pdf["MACD_Signal"])
            & (pdf["KVO"] > pdf["Klinger_Signal"])
        )

        feature_frames.append(
            pl.from_pandas(
                pdf[
                    [
                        "Ticker",
                        "Date",
                        "MACD",
                        "MACD_Signal",
                        "KVO",
                        "Klinger_Signal",
                        "Signal_KVO_MACD_Bullish",
                    ]
                ]
            )
        )

    features = (
        pl.concat(feature_frames)
        .with_columns(pl.col("Date").cast(pl.Date))
        .sort(["Ticker", "Date"])
    )
    df = df.with_columns(pl.col("Date").cast(pl.Date)).sort(["Ticker", "Date"])
    return df.join(
        features,
        on=["Ticker", "Date"],
        how="left",
    )

def append_vwap(
    df: pl.DataFrame,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    volume_col: str = "Volume",
    ticker_col: str = "Ticker",
    date_col: str = "Date",
    vwap_col: str = "VWAP",
) -> pl.DataFrame:
    """
    Append per-ticker cumulative VWAP to OHLCV dataframe.
    """

    typical_price = (
        (pl.col(high_col) + pl.col(low_col) + pl.col(close_col)) / 3
    )

    return (
        df.sort([ticker_col, date_col])
        .with_columns(
            (
                (
                    (typical_price * pl.col(volume_col))
                    .cum_sum()
                    .over(ticker_col)
                )
                /
                (
                    pl.col(volume_col)
                    .cum_sum()
                    .over(ticker_col)
                )
            ).alias(vwap_col)
        )
    )

def append_monthly_vwap(
    df: pl.DataFrame,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    volume_col: str = "Volume",
    ticker_col: str = "Ticker",
    date_col: str = "Date",
    vwap_col: str = "VWAP",
) -> pl.DataFrame:
    """
    Append monthly-reset VWAP to OHLCV dataframe.

    VWAP resets at the beginning of each month per ticker.
    """

    typical_price = (
        (pl.col(high_col) + pl.col(low_col) + pl.col(close_col)) / 3
    )

    return (
        df.sort([ticker_col, date_col])

        # Create month bucket
        .with_columns(
            pl.col(date_col)
            .dt.truncate("6mo")
            .alias("Month")
        )

        # Compute monthly-reset VWAP
        .with_columns(
            (
                (
                    (typical_price * pl.col(volume_col))
                    .cum_sum()
                    .over([ticker_col, "Month"])
                )
                /
                (
                    pl.col(volume_col)
                    .cum_sum()
                    .over([ticker_col, "Month"])
                )
            ).alias(vwap_col)
        )

        # Optional cleanup
        .drop("Month")
    )

def add_ew_volume_roc(
    df: pl.DataFrame,
    volume_col: str = "Volume",
    ticker_col: str = "Ticker",
    span: int = 20,
    roc_period: int = 5,
    output_col: str = "ew_volume_roc",
) -> pl.DataFrame:
    """
    Computes exponentially weighted volume ROC.

    Formula:

        ROC_t = log(V_t) - log(V_{t-k})

        EWROC_t = EMA(ROC_t)

    Returns dataframe with appended column.
    """

    alpha = 2 / (span + 1)

    return (
        df.sort([ticker_col, "Date"])
        .with_columns(
            (
                (
                    pl.col(volume_col)
                    .log()
                    -
                    pl.col(volume_col)
                    .log()
                    .shift(roc_period)
                )
                .over(ticker_col)
            ).alias("__volume_roc")
        )
        .with_columns(
            (
                pl.col("__volume_roc")
                .ewm_mean(alpha=alpha)
                .over(ticker_col)
            ).alias(output_col)
        )
        .drop("__volume_roc")
    )

def append_panel_rolling_amihud(df: pl.DataFrame, window_size: int = 5) -> pl.DataFrame:
    """
    Calculates the rolling Amihud Illiquidity Ratio across a stacked multi-ticker
    panel DataFrame using HL2 dollar volume, safely handling zero-volume anomalies.
    """
    # 1. Calculate absolute returns (Requires .over() because it looks at the previous row)
    abs_return_expr = pl.col("Close").pct_change().abs().over("Ticker")

    # 2. Calculate HL2 Dollar Volume (No .over() needed; purely horizontal arithmetic)
    hl2_price = (pl.col("High") + pl.col("Low")) / 2
    dollar_volume_expr = hl2_price * pl.col("Volume")

    # 3. Combine expressions safely handling zero-volume edge cases
    raw_amihud_expr = (
        pl.when(pl.col("Volume") > 0)
        .then((abs_return_expr / dollar_volume_expr) * 1_000_000)
        .otherwise(None) # Prevents division by zero, casting to Null safely ignored by rolling_mean
        .alias("Amihud_Raw")
    )

    # 4. Append the raw column
    df_with_raw = df.with_columns(raw_amihud_expr)

    # 5. Compute rolling window safely isolated over each Ticker context
    rolling_name = f"Amihud_{window_size}d"
    result_df = df_with_raw.with_columns(
        pl.col("Amihud_Raw")
        .rolling_mean(window_size=window_size)
        .over("Ticker")
        .alias(rolling_name)
    )

    return result_df

def append_log_returns(
    df: pl.DataFrame,
    price_col: str = "Close",
    ticker_col: str = "Ticker",
    output_col: str = "Log_Return",
) -> pl.DataFrame:
    """
    Appends log returns to an OHLCV dataframe.

    Formula:

        r_t = log(P_t) - log(P_{t-1})

    Assumes dataframe contains:
        - Date
        - Ticker
        - OHLCV columns

    Returns:
        Original dataframe with appended log return column.
    """

    return (
        df.sort([ticker_col, "Date"])
        .with_columns(
            (
                pl.col(price_col)
                .log()
                .diff()
                .over(ticker_col)
            ).alias(output_col)
        )
    )

def append_rsi(
    df: pl.DataFrame,
    price_col: str = "Close",
    ticker_col: str = "Ticker",
    date_col: str = "Date",
    period: int = 20,
    output_col: str = "RSI_20",
) -> pl.DataFrame:
    """
    Adds RSI to a multi-ticker OHLCV dataframe.

    Uses Wilder-style exponential smoothing.

    Formula:
        RSI = 100 - 100 / (1 + RS)

    where:
        RS = avg_gain / avg_loss

    Parameters
    ----------
    df : pl.DataFrame
        Long-format OHLCV dataframe.

    price_col : str
        Price column.

    ticker_col : str
        Ticker identifier column.

    date_col : str
        Date column.

    period : int
        RSI lookback period.

    output_col : str
        Output RSI column name.

    Returns
    -------
    pl.DataFrame
    """

    alpha = 1 / period

    return (
        df
        .sort([ticker_col, date_col])

        # ----------------------------------------------------
        # Price Delta
        # ----------------------------------------------------

        .with_columns(
            pl.col(price_col)
            .diff()
            .over(ticker_col)
            .alias("__delta")
        )

        # ----------------------------------------------------
        # Gains / Losses
        # ----------------------------------------------------

        .with_columns([
            (
                pl.when(pl.col("__delta") > 0)
                .then(pl.col("__delta"))
                .otherwise(0.0)
            ).alias("__gain"),

            (
                pl.when(pl.col("__delta") < 0)
                .then(-pl.col("__delta"))
                .otherwise(0.0)
            ).alias("__loss"),
        ])

        # ----------------------------------------------------
        # Wilder EWMA
        # ----------------------------------------------------

        .with_columns([
            (
                pl.col("__gain")
                .ewm_mean(alpha=alpha)
                .over(ticker_col)
            ).alias("__avg_gain"),

            (
                pl.col("__loss")
                .ewm_mean(alpha=alpha)
                .over(ticker_col)
            ).alias("__avg_loss"),
        ])

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        .with_columns(
            (
                100
                -
                (
                    100
                    /
                    (
                        1
                        +
                        (
                            pl.col("__avg_gain")
                            /
                            (pl.col("__avg_loss") + 1e-12)
                        )
                    )
                )
            ).alias(output_col)
        )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        .drop([
            "__delta",
            "__gain",
            "__loss",
            "__avg_gain",
            "__avg_loss",
        ])
    )

def append_panel_rolling_amihud_ratio(
    df: pl.DataFrame,
    short_window: int = 5,
    long_window: int = 20
) -> pl.DataFrame:
    """
    Calculates the short-term and long-term EWMA Amihud Illiquidity Ratios
    across a stacked multi-ticker panel DataFrame, and appends their ratio
    (short_window / long_window) while safely handling zero or null anomalies.
    """
    # 1. Calculate absolute returns (Requires .over() because it looks at the previous row)
    abs_return_expr = pl.col("Close").pct_change().abs().over("Ticker")

    # 2. Calculate HL2 Dollar Volume (No .over() needed; purely horizontal arithmetic)
    hl2_price = (pl.col("High") + pl.col("Low")) / 2
    dollar_volume_expr = hl2_price * pl.col("Volume")

    # 3. Combine expressions safely handling zero-volume edge cases
    raw_amihud_expr = (
        pl.when(pl.col("Volume") > 0)
        .then((abs_return_expr / dollar_volume_expr) * 1_000_000)
        .otherwise(None) # Prevents division by zero
        .alias("Amihud_Raw")
    )

    # 4. Append the raw column
    df_with_raw = df.with_columns(raw_amihud_expr)

    # 5. Compute both EWMA windows safely isolated over each Ticker context
    short_name = f"Amihud_{short_window}d_EWMA"
    long_name = f"Amihud_{long_window}d_EWMA"
    ratio_name = f"Amihud_{short_window}d_{long_window}d_Ratio"

    df_with_windows = df_with_raw.with_columns([
        pl.col("Amihud_Raw")
        .ewm_mean(span=short_window, ignore_nulls=True)
        .over("Ticker")
        .alias(short_name),

        pl.col("Amihud_Raw")
        .ewm_mean(span=long_window, ignore_nulls=True)
        .over("Ticker")
        .alias(long_name),
    ])

    # 6. Safely compute the ratio to prevent division-by-zero if long_window is 0 or Null
    result_df = df_with_windows.with_columns(
        pl.when(pl.col(long_name) > 0)
        .then(pl.col(short_name) / pl.col(long_name))
        .otherwise(None)
        .alias(ratio_name)
    )

    # 7. Add pct change of ratio (Requires .over() for panel compliance)
    result_df = result_df.with_columns(
        pl.col(ratio_name).pct_change().over("Ticker").alias(f"{ratio_name}_Pct_Change")
    )

    return result_df

def append_weekly_macd_klinger_hist(
    df: pl.DataFrame,
    weekly_df: pl.DataFrame,
    ticker_col: str = "Ticker",
    date_col: str = "Date",
    macd_hist_col: str = "W_MACD_HIST",
    klinger_hist_col: str = "W_KLINGER_HIST",
) -> pl.DataFrame:
    """
    Computes MACD histogram and Klinger Volume Oscillator histogram on
    a weekly OHLCV dataframe, then forward-fills those values back onto
    the original (daily) dataframe via an as-of join. Compute using ta library

    Weekly indicators:
      - MACD Histogram  = MACD_Line  - MACD_Signal
        where MACD_Line   = EMA(Close,12) - EMA(Close,26)
              MACD_Signal = EMA(MACD_Line, 9)
      - Klinger Histogram = Klinger_Line - Klinger_Signal
        where Klinger_Line   = EMA(VF,34) - EMA(VF,55)
              Klinger_Signal  = EMA(Klinger_Line, 13)

    Parameters
    ----------
    df : pl.DataFrame
        Daily long-form OHLCV panel sorted by [ticker_col, date_col].

    weekly_df : pl.DataFrame
        Weekly aggregated OHLCV panel (e.g. produced by group_by_dynamic
        with every='1w'), containing at least:
        Ticker, Date, Open, High, Low, Close, Volume.

    ticker_col : str
        Name of the ticker identifier column.

    date_col : str
        Name of the date column.

    macd_hist_col : str
        Output column name for the weekly MACD histogram feature.

    klinger_hist_col : str
        Output column name for the weekly Klinger histogram feature.

    Returns
    -------
    pl.DataFrame
        Original df with two new columns appended:
        W_MACD_HIST and W_KLINGER_HIST.
    """

    df = df.sort([ticker_col, date_col])
    weekly_df = weekly_df.sort([ticker_col, date_col])

    features = []

    for group in weekly_df.partition_by(ticker_col, maintain_order=True):

        pdf = group.to_pandas()

        # MACD Histogram
        macd = pta.macd(
            pdf["Close"],
            fast=12,
            slow=26,
            signal=9,
        )

        pdf[macd_hist_col] = (
            macd.filter(like="MACDh")
            .iloc[:, 0]
        )

        # Klinger Histogram
        kvo = pta.kvo(
            high=pdf["High"],
            low=pdf["Low"],
            close=pdf["Close"],
            volume=pdf["Volume"],
            fast=34,
            slow=55,
            signal=13,
        )

        kvo_line = (
            kvo.filter(regex=r"^KVO_")
            .iloc[:, 0]
        )

        kvo_signal = (
            kvo.filter(regex=r"^KVOs_")
            .iloc[:, 0]
        )

        pdf[klinger_hist_col] = (
            kvo_line - kvo_signal
        )

        features.append(
            pl.from_pandas(
                pdf[
                    [
                        ticker_col,
                        date_col,
                        macd_hist_col,
                        klinger_hist_col,
                    ]
                ]
            )
        )

    weekly_features = (
        pl.concat(features)
        .with_columns(
            pl.col(date_col).cast(pl.Date)
        )
        .sort([ticker_col, date_col])
    )

    return df.join_asof(
        weekly_features,
        on=date_col,
        by=ticker_col,
        strategy="backward",
    )

    