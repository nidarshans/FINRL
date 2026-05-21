import numpy as np
import polars as pl
from typing import Dict, List


def utils_calculate_covariance_matrix(
    df: pl.DataFrame, column: str = None
) -> pl.DataFrame:
    """Pivots the DataFrame into wide format and returns its cross-ticker covariance matrix."""
    # Use override if provided, otherwise default to the GARCH string
    target_col = column

    wide_df = df.pivot(
        index="Date",
        on="Ticker",
        values=target_col,
        aggregate_function="first",
    )

    ticker_names = [col for col in wide_df.columns if col != "Date"]
    cleaned_wide = wide_df.select(ticker_names).drop_nulls()
    matrix_data = cleaned_wide.to_numpy()

    if matrix_data.shape[0] <= 1:
        raise ValueError("No overlapping data points left after dropping nulls.")

    cov_array = np.cov(matrix_data.T)
    cov_array = np.atleast_2d(cov_array)

    cov_df = pl.DataFrame(cov_array, schema=ticker_names)
    return cov_df.insert_column(0, pl.Series("Ticker", ticker_names))


def utils_calculate_correlation_matrix(
    df: pl.DataFrame, column: str = None
) -> pl.DataFrame:
    """Pivots the DataFrame into wide format and returns its cross-ticker correlation matrix."""
    target_col = column

    wide_df = df.pivot(
        index="Date",
        on="Ticker",
        values=target_col,
        aggregate_function="first",
    )

    ticker_names = [col for col in wide_df.columns if col != "Date"]
    cleaned_wide = wide_df.select(ticker_names).drop_nulls()
    matrix_data = cleaned_wide.to_numpy()

    if matrix_data.shape[0] <= 1:
        raise ValueError("No overlapping data points left after dropping nulls.")

    corr_array = np.corrcoef(matrix_data.T)
    corr_array = np.atleast_2d(corr_array)

    corr_df = pl.DataFrame(corr_array, schema=ticker_names)
    return corr_df.insert_column(0, pl.Series("Ticker", ticker_names))

def utils_calculate_eigenvalues(cov_df: pl.DataFrame) -> pl.DataFrame:
    """Takes a Polars covariance matrix DataFrame, calculates its eigenvalues,

    and returns them sorted in descending order.
    """
    # 1. Drop the 'Ticker' label column to get a pure numeric matrix
    ticker_names = [col for col in cov_df.columns if col != "Ticker"]
    matrix_data = cov_df.select(ticker_names).to_numpy()

    # 2. Compute eigenvalues
    # Since covariance matrices are symmetric, eigvalsh is faster and more stable than eigvals
    eigenvalues = np.linalg.eigvalsh(matrix_data)

    # 3. Sort eigenvalues descending (highest variance/principal components first)
    eigenvalues = eigenvalues[::-1]

    # 4. Return as a clean Polars DataFrame
    return pl.DataFrame(
        {
            "Component": [f"PC_{i+1}" for i in range(len(eigenvalues))],
            "Eigenvalue": eigenvalues,
        }
    )

def pipeline_rolling_subsector_eigenvalues(
    df_features: pl.DataFrame,
    gics_dict: Dict[str, Dict[str, List[str]]],
    column: str,
    lookback_window: int,
    method: str = "correlation",
) -> pl.DataFrame:
    """Computes a historical time-series of PC1 and PC2 eigenvalues

    individually for EVERY GICS subsector in the dataset using an optimized
    matrix execution loop.
    """
    # 1. Flatten the GICS dictionary structure into a high-performance metadata table
    flat_gics = []
    for sector, subsectors in gics_dict.items():
        for subsector, tickers in subsectors.items():
            for t in tickers:
                flat_gics.append((t, sector, subsector))

    df_meta = pl.DataFrame(
        flat_gics, schema=["Ticker", "Sector", "Sub_Industry"]
    )

    # 2. Join asset metadata onto the core features dataframe
    # This labels every historical data point with its specific subsector
    df_labeled = df_features.join(df_meta, on="Ticker", how="inner")

    # Get unique, chronologically sorted dates and identify active subsectors
    unique_dates = df_labeled["Date"].unique().sort()
    total_days = len(unique_dates)
    subsectors_list = df_meta["Sub_Industry"].unique().to_list()

    if total_days < lookback_window:
        raise ValueError(
            f"Dataset length ({total_days} days) is less than the lookback window ({lookback_window})."
        )

    # Accumulator for our final long-form panel output dataset
    all_subsector_results = []

    # 3. Time-Timeline Vectorized Windowing
    # We step through time chronologically, capturing the required lookback chunk
    for i in range(lookback_window, total_days + 1):
        window_dates = unique_dates[i - lookback_window : i]
        current_date = window_dates[-1]

        # Isolate the cross-sectional rows belonging strictly to this time window
        window_df = df_labeled.filter(pl.col("Date").is_in(window_dates))

        # 4. Intra-Window Subsector Slicing
        for subsector in subsectors_list:
            subsector_df = window_df.filter(
                pl.col("Sub_Industry") == subsector
            )

            # Skip subsectors that don't have enough active tickers in this timeframe
            unique_tickers = subsector_df["Ticker"].n_unique()
            if unique_tickers < 2:
                continue

            try:
                # Reuse your mathematical matrix functions
                if method.lower() == "covariance":
                    matrix_df = utils_calculate_covariance_matrix(
                        subsector_df, column=column
                    )
                else:
                    matrix_df = utils_calculate_correlation_matrix(
                        subsector_df, column=column
                    )

                # Extract eigenvalues
                eigen_df = utils_calculate_eigenvalues(matrix_df)

                pc1 = (
                    eigen_df["Eigenvalue"][0] if len(eigen_df) >= 1 else np.nan
                )
                pc2 = (
                    eigen_df["Eigenvalue"][1] if len(eigen_df) >= 2 else np.nan
                )

            except (ValueError, LinAlgError):
                # Catches matrices collapsed by missing data or singular transformations
                pc1, pc2 = np.nan, np.nan

            # Append the data point tied to its time and subsector coordinates
            all_subsector_results.append(
                {
                    "Date": current_date,
                    "Sub_Industry": subsector,
                    f"{column}_PC1": pc1,
                    f"{column}_PC2": pc2,
                }
            )

    # 5. Compile into a master long-form panel DataFrame
    if not all_subsector_results:
        return pl.DataFrame()

    return pl.DataFrame(all_subsector_results).sort(["Sub_Industry", "Date"])