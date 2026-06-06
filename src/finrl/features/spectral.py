"""Trailing spectral market features."""

from __future__ import annotations

import numpy as np
import polars as pl


def _trailing_eigenspectrum(
    data: pl.DataFrame,
    value_col: str,
    prefix: str,
    lookback: int,
    n_components: int,
) -> pl.DataFrame:
    required = {"date", "ticker", value_col}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing spectral input columns: {sorted(missing)}")

    pivot = (
        data.select(["date", "ticker", value_col])
        .pivot(index="date", on="ticker", values=value_col, aggregate_function="first")
        .sort("date")
    )
    dates = pivot.get_column("date").to_list()
    tickers = [column for column in pivot.columns if column != "date"]
    matrix = pivot.select(tickers).fill_null(0.0).to_numpy()
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        start = max(0, index - lookback + 1)
        window = matrix[start : index + 1]
        if window.shape[0] < 2:
            values = np.zeros(n_components, dtype=float)
        else:
            covariance = np.cov(window, rowvar=False)
            covariance = np.atleast_2d(covariance)
            eigvals = np.linalg.eigvalsh(covariance)
            values = np.sort(eigvals)[::-1]
            values = np.pad(values, (0, max(0, n_components - values.shape[0])))
            values = values[:n_components]
        row = {"date": date}
        row.update({f"{prefix}_{i}": float(values[i]) for i in range(n_components)})
        rows.append(row)
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def compute_volume_eigenspectrum(
    data: pl.DataFrame,
    lookback: int = 20,
    n_components: int = 8,
) -> pl.DataFrame:
    """Compute trailing covariance eigenspectrum of dollar volume."""

    return _trailing_eigenspectrum(data, "dollar_volume", "volume_eigen", lookback, n_components)


def compute_liquidity_eigenspectrum(
    data: pl.DataFrame,
    lookback: int = 20,
    n_components: int = 8,
) -> pl.DataFrame:
    """Compute trailing covariance eigenspectrum of Amihud illiquidity."""

    return _trailing_eigenspectrum(
        data, "amihud_illiquidity", "liquidity_eigen", lookback, n_components
    )


def compute_sector_flow_indicators(data: pl.DataFrame) -> pl.DataFrame:
    """Compute market-wide flow proxies without sector metadata.

    Sector classifications are not available in Phase 5 ingestion, so this
    returns cross-sectional market flow aggregates and names them as proxies.
    """

    required = {"date", "return", "dollar_volume", "amihud_illiquidity"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing flow input columns: {sorted(missing)}")
    return (
        data.group_by("date")
        .agg(
            pl.mean("return").alias("sector_flow_return_mean"),
            pl.mean("dollar_volume").alias("sector_flow_dollar_volume_mean"),
            pl.mean("amihud_illiquidity").alias("sector_flow_illiquidity_mean"),
            pl.std("return").fill_null(0.0).alias("sector_flow_return_dispersion"),
        )
        .sort("date")
    )


def compute_spectral_features(
    asset_features: pl.DataFrame,
    lookback: int = 20,
    spectral_dim: int = 20,
) -> pl.DataFrame:
    """Build a fixed-width trailing spectral feature table."""

    volume_components = min(8, spectral_dim)
    liquidity_components = min(8, max(0, spectral_dim - volume_components))
    volume = compute_volume_eigenspectrum(asset_features, lookback, volume_components)
    liquidity = compute_liquidity_eigenspectrum(asset_features, lookback, liquidity_components)
    flows = compute_sector_flow_indicators(asset_features)
    spectral = volume.join(liquidity, on="date", how="inner").join(flows, on="date", how="inner")

    feature_columns = [column for column in spectral.columns if column != "date"]
    for index in range(len(feature_columns), spectral_dim):
        spectral = spectral.with_columns(pl.lit(0.0).alias(f"spectral_pad_{index}"))
    selected = ["date", *[column for column in spectral.columns if column != "date"][:spectral_dim]]
    return spectral.select(selected).sort("date")
