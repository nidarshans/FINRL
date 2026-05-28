import polars as pl

def utils_add_klinger_and_macd_signals(df: pl.DataFrame) -> pl.DataFrame:

    """
    Engineers Klinger Volume Oscillator (KVO) and MACD signals,
    returning a primary Boolean flag column: 'Signal_KVO_MACD_Bullish'.

    Assumes df is a long-form panel sorted by ['Ticker', 'Date']
    containing columns: 'Open', 'High', 'Low', 'Close', 'Volume'.
    """

    # -------------------------------------------------------------
    # LAYER 1: MACD CALCULATION
    # -------------------------------------------------------------
    # MACD Line = EMA(Close, 12) - EMA(Close, 26)
    # Signal Line = EMA(MACD Line, 9)
    df_macd = df.with_columns([
        pl.col("Close").ewm_mean(span=12, adjust=False).over("Ticker").alias("EMA_12"),
        pl.col("Close").ewm_mean(span=26, adjust=False).over("Ticker").alias("EMA_26")
    ]).with_columns([
        (pl.col("EMA_12") - pl.col("EMA_26")).alias("MACD_Line")
    ]).with_columns([
        pl.col("MACD_Line").ewm_mean(span=9, adjust=False).over("Ticker").alias("MACD_Signal")
    ])

    # -------------------------------------------------------------
    # LAYER 2: KLINGER VOLUME OSCILLATOR (KVO) CALCULATION
    # -------------------------------------------------------------
    # 1. Typical Price (TP) = (High + Low + Close) / 3
    # 2. Daily Force (VF) = Volume * Trend * 100, where Trend is based on TP movement
    df_kvo = df_macd.with_columns([
        ((pl.col("High") + pl.col("Low") + pl.col("Close")) / 3.0).alias("Typical_Price"),
        (pl.col("High") - pl.col("Low")).alias("Daily_Range")
    ]).with_columns([
        # Determine DM (Daily Trend Direction)
        pl.when(pl.col("Typical_Price") >= pl.col("Typical_Price").shift(1).over("Ticker"))
          .then(1)
          .otherwise(-1)
          .alias("Trend_Direction")
    ]).with_columns([
        # Cumulative Range (cm): rolling sum of daily ranges as a robust
        # approximation of Klinger's original cumulative range variable
        pl.col("Daily_Range").rolling_sum(window_size=10).over("Ticker").alias("Cumulative_Range")
    ]).with_columns([
        # Compute Volume Force (VF) using Klinger's dm/cm scaling
        pl.when(pl.col("Daily_Range") == 0)
          .then(0.0)
          .when(pl.col("Cumulative_Range") == 0)
          .then(0.0)
          .otherwise(
              pl.col("Volume") * pl.col("Trend_Direction") * 100.0 *
              ((2.0 * ((pl.col("High") - pl.col("Low")) / pl.col("Cumulative_Range"))) - 1.0)
          ).alias("Volume_Force")
    ]).with_columns([
        # Klinger Line = EMA(Volume Force, 34) - EMA(Volume Force, 55)
        pl.col("Volume_Force").ewm_mean(span=34, adjust=False).over("Ticker").alias("VF_EMA_34"),
        pl.col("Volume_Force").ewm_mean(span=55, adjust=False).over("Ticker").alias("VF_EMA_55")
    ]).with_columns([
        (pl.col("VF_EMA_34") - pl.col("VF_EMA_55")).alias("Klinger_Line")
    ]).with_columns([
        # Klinger Signal Line = EMA(Klinger Line, 13)
        pl.col("Klinger_Line").ewm_mean(span=13, adjust=False).over("Ticker").alias("Klinger_Signal")
    ])

    # -------------------------------------------------------------
    # LAYER 3: CONSTRUCT THE BOOLEAN FEATURE
    # -------------------------------------------------------------
    # True if BOTH Signal Lines >= 0, else False
    df_final = df_kvo.with_columns([
        (pl.col("MACD_Signal") >= 0.0).alias("Signal_MACD_Bullish"),
        (pl.col("Klinger_Signal") >= 0.0).alias("Signal_KVO_Bullish")
    ])

    # Drop intermediary columns to keep your machine learning data clear and compact
    columns_to_drop = [
        "EMA_12", "EMA_26", "MACD_Line", "Typical_Price",
        "Daily_Range", "Cumulative_Range", "Trend_Direction", "Volume_Force",
        "VF_EMA_34", "VF_EMA_55", "Klinger_Line"
    ]

    return df_final.drop(columns_to_drop)

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