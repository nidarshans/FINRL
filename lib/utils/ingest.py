from typing import List
import pandas as pd
import polars as pl
import yfinance as yf


def pipeline_ingest_data(
    tickers: List[str], start_date: str, end_date: str
) -> pl.DataFrame:
    """Stage 1 & 2 of the Quant Pipeline: Correctly flattens multi-ticker data

    by stacking on the index, then cleans using Polars.
    """
    # 1. Fetch data
    df_pandas = yf.download(
        tickers, start=start_date, end=end_date, auto_adjust=True
    )

    # CRITICAL FIX: Ensure index is named Date, and stack BEFORE resetting index.
    # This prevents the dates from turning into nulls.
    df_pandas.index.name = "Date"

    if isinstance(df_pandas.columns, pd.MultiIndex):
        # Stack 'Ticker' from columns to rows while keeping Date as the index
        df_pandas = df_pandas.stack(level="Ticker", future_stack=True)

    # Now reset the index to turn Date and Ticker into regular columns
    df_pandas = df_pandas.reset_index()

    # Convert to Polars DataFrame
    df = pl.from_pandas(df_pandas)

    # Clean up names (yfinance sometimes leaves column names capitalized or lowercase)
    # This maps whatever yfinance returns to standard OHLCV
    df = df.rename(
        {col: col.title() for col in df.columns if col.title() in ["Volume"]}
    )

    # Cast Date to proper Polars Date type
    df = df.with_columns(pl.col("Date").cast(pl.Date))

    # Ensure clean column structure
    df = df.select(
        ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    )

    # 2. Clean (Forward-fill then Backward-fill per Ticker)
    cleaned_df = (
        df.sort("Date")
        .group_by("Ticker", maintain_order=True)
        .agg(
            [
                pl.col("Date"),
                pl.col("Open").forward_fill().backward_fill(),
                pl.col("High").forward_fill().backward_fill(),
                pl.col("Low").forward_fill().backward_fill(),
                pl.col("Close").forward_fill().backward_fill(),
                pl.col("Volume").forward_fill().backward_fill(),
            ]
        )
        .explode(pl.all().exclude("Ticker"))
    )

    # Return the clean dataframe sorted chronologically
    return cleaned_df.sort("Ticker", "Date")