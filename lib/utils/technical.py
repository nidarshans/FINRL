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
        # Compute Volume Force (VF)
        pl.when(pl.col("Daily_Range") == 0)
          .then(0.0)
          .otherwise(
              pl.col("Volume") * pl.col("Trend_Direction") * 100.0 *
              ((2.0 * ((pl.col("High") - pl.col("Low")) / pl.col("Daily_Range"))) - 1.0)
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
        "Daily_Range", "Trend_Direction", "Volume_Force",
        "VF_EMA_34", "VF_EMA_55", "Klinger_Line"
    ]

    return df_final.drop(columns_to_drop)