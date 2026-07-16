"""Causal market-relative features aligned to a daily benchmark series."""

from __future__ import annotations

import polars as pl

from finrl.data.schema import enforce_ohlcv_schema


def compute_market_relative_features(
    assets: pl.DataFrame,
    benchmark: pl.DataFrame,
) -> pl.DataFrame:
    """Return trailing stock-versus-benchmark features for each asset/date.

    Benchmark observations are joined by their actual dates.  Missing dates
    produce null feature values; they are never substituted with zero returns.
    """

    if benchmark.is_empty():
        raise ValueError("Market-relative features require non-empty benchmark OHLCV.")
    benchmark_returns = enforce_ohlcv_schema(benchmark).sort("date").select(
        "date", pl.col("close").pct_change().alias("_market_return"),
        pl.col("close").alias("_market_close"),
    )
    output = enforce_ohlcv_schema(assets).sort(["ticker", "date"]).join(
        benchmark_returns, on="date", how="left"
    ).with_columns(
        pl.col("close").pct_change().over("ticker").alias("_stock_return"),
    ).with_columns(
        (pl.col("_stock_return") * pl.col("_market_return")).alias("_return_product"),
        (pl.col("_market_return") ** 2).alias("_market_return_sq"),
    ).with_columns(
        pl.col("_stock_return").rolling_mean(252, min_samples=2).over("ticker").alias("_stock_mean"),
        pl.col("_market_return").rolling_mean(252, min_samples=2).over("ticker").alias("_market_mean"),
        pl.col("_return_product").rolling_mean(252, min_samples=2).over("ticker").alias("_cross_mean"),
        pl.col("_market_return_sq").rolling_mean(252, min_samples=2).over("ticker").alias("_market_sq_mean"),
    ).with_columns(
        ((pl.col("_cross_mean") - pl.col("_stock_mean") * pl.col("_market_mean")) /
         (pl.col("_market_sq_mean") - pl.col("_market_mean") ** 2 + 1e-9)).alias("beta_252"),
    ).with_columns(
        (pl.col("_stock_return") - pl.col("beta_252") * pl.col("_market_return")).alias("_residual_return"),
        (
            pl.col("close") / pl.col("close").shift(63).over("ticker")
            - pl.col("_market_close") / pl.col("_market_close").shift(63)
        ).alias("relative_strength_63"),
    ).with_columns(
        pl.col("_residual_return").rolling_std(60, min_samples=2).over("ticker").alias("idio_vol_60"),
        (
            pl.col("close").shift(21).over("ticker")
            / pl.col("close").shift(126).over("ticker")
            - pl.col("beta_252")
            * (
                pl.col("_market_close").shift(21) / pl.col("_market_close").shift(126)
                - 1.0
            )
        ).alias("residual_mom_126_21"),
    )
    return output.select(
        "date", "ticker", "relative_strength_63", "beta_252",
        "residual_mom_126_21", "idio_vol_60",
    )
