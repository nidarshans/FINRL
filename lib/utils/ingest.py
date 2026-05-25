from typing import List
import yfinance as yf
import pandas as pd
import polars as pl
from lib.utils.technical import *


def pipeline_ingest_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """
    Stage 1 & 2 of quant pipeline:
    - Fetch OHLCV
    - Flatten multi-index
    - Clean missing values
    - Append VWAP
    """

    # -----------------------------
    # Fetch Data
    # -----------------------------

    df_pandas = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
    )

    df_pandas.index.name = "Date"

    # -----------------------------
    # Flatten MultiIndex
    # -----------------------------

    if isinstance(df_pandas.columns, pd.MultiIndex):

        df_pandas = df_pandas.stack(
            level="Ticker",
            future_stack=True,
        )

    df_pandas = df_pandas.reset_index()

    # -----------------------------
    # Convert to Polars
    # -----------------------------

    df = pl.from_pandas(df_pandas)

    # -----------------------------
    # Normalize column names
    # -----------------------------

    df = df.rename(
        {
            col: col.title()
            for col in df.columns
        }
    )

    # -----------------------------
    # Cast Date
    # -----------------------------

    df = df.with_columns(
        pl.col("Date").cast(pl.Date)
    )

    # -----------------------------
    # Select canonical columns
    # -----------------------------

    df = df.select(
        [
            "Date",
            "Ticker",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )

    # -----------------------------
    # Clean missing values
    # -----------------------------

    cols_to_fill = ["Open", "High", "Low", "Close", "Volume"]
    cleaned_df = (
        df.sort(["Ticker", "Date"])
        .with_columns(
            [
                pl.col(c)
                    .forward_fill()
                    .backward_fill()
                    .over("Ticker")
                for c in cols_to_fill
            ]
        )
    )

    # -----------------------------
    # Append VWAP
    # -----------------------------

    cleaned_df = append_monthly_vwap(cleaned_df)

    return cleaned_df

# ============================================================
# 1. BUILD HIERARCHY MAPS
# ============================================================

def build_hierarchy_maps(
    gics_dict: Dict
):
    """
    Creates:

        ticker -> sector
        ticker -> industry

    and:

        sector -> tickers
        industry -> tickers
    """

    ticker_to_sector = {}
    ticker_to_industry = {}

    sector_to_tickers = {}
    industry_to_tickers = {}

    for sector, industries in gics_dict.items():

        sector_tickers = []

        for industry, tickers in industries.items():

            industry_unique = sorted(list(set(tickers)))

            industry_to_tickers[industry] = industry_unique

            for ticker in industry_unique:

                ticker_to_sector[ticker] = sector
                ticker_to_industry[ticker] = industry

                sector_tickers.append(ticker)

        sector_to_tickers[sector] = sorted(
            list(set(sector_tickers))
        )

    return (
        ticker_to_sector,
        ticker_to_industry,
        sector_to_tickers,
        industry_to_tickers,
    )


# ============================================================
# 2. CREATE FEATURE MATRIX
# ============================================================

def create_feature_matrix(
    df: pl.DataFrame,
    feature_col: str,
):
    """
    Converts:

        long-format dataframe

    into:

        X[T, N]
    """

    wide_df = (
        df.pivot(
            index="Date",
            on="Ticker",
            values=feature_col,
            aggregate_function="first",
        )
        .sort("Date")
    )

    dates = wide_df["Date"].to_numpy()

    tickers = [
        c for c in wide_df.columns
        if c != "Date"
    ]

    X = (
        wide_df
        .select(tickers)
        .fill_null(strategy="forward")
        .fill_null(0.0)
        .to_numpy()
    )

    return X, tickers, dates