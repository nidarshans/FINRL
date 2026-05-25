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