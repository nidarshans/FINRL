"""
Correlation matrix construction, PCA analysis, and feature injection pipeline.

This module orchestrates the cross-sector regime detection workflow:
  1. compute_metric()       — dispatch to garch_returns / kvo_pct / raw_returns
  2. build_corr_matrix()    — rolling correlation matrix from metric DataFrame
  3. run_pca_on_corr()      — eigendecomposition per date
  4. inject_pca_features()  — merge eigenvalue columns into per-sector DataFrames
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from lib.regime_detection.src.constants import (
    CORR_METRIC, CORR_WINDOW, GARCH_P, GARCH_Q, PCA_N_COMPONENTS,
)


# ==============================================================================
# 1. Metric Computation
# ==============================================================================

def compute_metric(data_dict: dict, metric: str = CORR_METRIC) -> pd.DataFrame:
    """
    Compute the cross-sector metric panel used to build the correlation matrix.

    Parameters
    ----------
    data_dict : dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame.
    metric : str
        One of "garch_returns", "kvo_pct", "raw_returns".

    Returns
    -------
    pd.DataFrame
        Index = dates, columns = sector tickers, values = metric values.
    """
    if metric == "garch_returns":
        return _metric_garch(data_dict)
    elif metric == "kvo_pct":
        return _metric_kvo_pct(data_dict)
    elif metric == "raw_returns":
        return _metric_raw_returns(data_dict)
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Choose from: 'garch_returns', 'kvo_pct', 'raw_returns'"
        )


def _metric_garch(data_dict: dict) -> pd.DataFrame:
    """GARCH-adjusted standardized residuals across all sectors."""
    from lib.regime_detection.src.features.garch import fit_garch_all
    return fit_garch_all(data_dict, p=GARCH_P, q=GARCH_Q)


def _metric_kvo_pct(data_dict: dict) -> pd.DataFrame:
    """KVO oscillator percent change across all sectors."""
    from lib.regime_detection.src.features.signals import build_signals

    kvo_series = {}
    for ticker, df in data_dict.items():
        df_sig, _ = build_signals(df)
        if "KVO" in df_sig.columns:
            kvo_pct = df_sig["KVO"].pct_change().replace([np.inf, -np.inf], np.nan)
            kvo_series[ticker] = kvo_pct

    metric_df = pd.DataFrame(kvo_series)
    return metric_df.dropna(how="all")


def _metric_raw_returns(data_dict: dict) -> pd.DataFrame:
    """Simple close-to-close percentage returns."""
    returns = {}
    for ticker, df in data_dict.items():
        returns[ticker] = df["Close"].pct_change()

    metric_df = pd.DataFrame(returns)
    return metric_df.dropna(how="all")


# ==============================================================================
# 2. Rolling Correlation Matrix
# ==============================================================================

def build_corr_matrix(
    metric_df: pd.DataFrame, window: int = CORR_WINDOW
) -> dict:
    """
    Construct a rolling correlation matrix from the metric DataFrame.

    Parameters
    ----------
    metric_df : pd.DataFrame
        Index = dates, columns = sector tickers.
    window : int
        Rolling window size in trading days.

    Returns
    -------
    dict[pd.Timestamp, np.ndarray]
        Mapping of date → (n_sectors × n_sectors) correlation matrix.
        Only dates with a full window of data are included.
    """
    corr_dict = {}
    dates = metric_df.index
    n_cols = metric_df.shape[1]

    for i in range(window - 1, len(dates)):
        date = dates[i]
        window_data = metric_df.iloc[i - window + 1 : i + 1]

        # Need at least window/2 non-NaN observations per column
        min_obs = max(window // 2, 5)
        valid_cols = window_data.columns[window_data.notna().sum() >= min_obs]

        if len(valid_cols) < 2:
            continue

        corr = window_data[valid_cols].corr().values

        # Handle NaN in correlation matrix (can occur with constant columns)
        if np.isnan(corr).any():
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)

        # If not all columns were valid, embed into full-size matrix
        if len(valid_cols) < n_cols:
            full_corr = np.eye(n_cols)
            valid_idx = [metric_df.columns.get_loc(c) for c in valid_cols]
            for ii, vi in enumerate(valid_idx):
                for jj, vj in enumerate(valid_idx):
                    full_corr[vi, vj] = corr[ii, jj]
            corr_dict[date] = full_corr
        else:
            corr_dict[date] = corr

    print(f"  [CORR] Built {len(corr_dict)} rolling correlation matrices "
          f"(window={window}, sectors={n_cols})")
    return corr_dict


# ==============================================================================
# 3. PCA on Correlation Matrices
# ==============================================================================

def run_pca_on_corr(
    corr_dict: dict,
    sectors: list = None,
    n_components: int = PCA_N_COMPONENTS,
) -> tuple:
    """
    Run eigendecomposition on each rolling correlation matrix.

    Parameters
    ----------
    corr_dict : dict[pd.Timestamp, np.ndarray]
        Output of build_corr_matrix().
    sectors : list[str], optional
        Sector names for labeling PC loadings.
    n_components : int or None
        Number of principal components to retain. None = all.

    Returns
    -------
    eigenvalues_df : pd.DataFrame
        Index = dates, columns = ['Eigenvalue_1', ..., 'Eigenvalue_K'].
    pc1_loadings_df : pd.DataFrame
        Index = dates, columns = sector names. PC1 loadings over time.
    explained_variance_df : pd.DataFrame
        Index = dates, columns = ['PC_1_var', ..., 'PC_K_var'].
        Fraction of variance explained by each component.
    """
    if not corr_dict:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    dates = sorted(corr_dict.keys())
    sample_matrix = corr_dict[dates[0]]
    n_sectors = sample_matrix.shape[0]
    k = n_components if n_components is not None else n_sectors

    eigenvalue_rows = []
    pc1_loading_rows = []
    explained_rows = []

    for date in dates:
        corr = corr_dict[date]

        # Eigendecomposition (correlation matrix is symmetric → use eigh)
        eigenvalues, eigenvectors = np.linalg.eigh(corr)

        # eigh returns in ascending order; reverse to descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clamp negative eigenvalues (numerical noise) to zero
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Top-k eigenvalues
        top_eigenvalues = eigenvalues[:k]
        eigenvalue_rows.append(top_eigenvalues)

        # PC1 loadings (first eigenvector)
        pc1_loading_rows.append(eigenvectors[:, 0])

        # Explained variance ratio
        total = eigenvalues.sum()
        if total > 0:
            explained = eigenvalues[:k] / total
        else:
            explained = np.zeros(k)
        explained_rows.append(explained)

    # Build DataFrames
    eig_cols = [f"Eigenvalue_{i+1}" for i in range(k)]
    eigenvalues_df = pd.DataFrame(eigenvalue_rows, index=dates, columns=eig_cols)

    sector_labels = sectors if sectors is not None else [f"Sector_{i}" for i in range(n_sectors)]
    # Trim labels if there are more sectors than matrix dimensions
    sector_labels = sector_labels[:n_sectors]
    pc1_loadings_df = pd.DataFrame(pc1_loading_rows, index=dates, columns=sector_labels)

    var_cols = [f"PC_{i+1}_var" for i in range(k)]
    explained_variance_df = pd.DataFrame(explained_rows, index=dates, columns=var_cols)

    print(f"  [PCA] Computed eigenvalues for {len(dates)} dates, "
          f"retaining {k} components")
    return eigenvalues_df, pc1_loadings_df, explained_variance_df


# ==============================================================================
# 4. Feature Injection
# ==============================================================================

def inject_pca_features(
    data_dict: dict, eigenvalues_df: pd.DataFrame
) -> dict:
    """
    Merge eigenvalue columns (Eigenvalue_1, Eigenvalue_2) into each sector's
    DataFrame by date-index alignment.

    Parameters
    ----------
    data_dict : dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV+features DataFrame.
    eigenvalues_df : pd.DataFrame
        Output of run_pca_on_corr(). Must contain 'Eigenvalue_1', 'Eigenvalue_2'.

    Returns
    -------
    dict[str, pd.DataFrame]
        Same data_dict with Eigenvalue_1 and Eigenvalue_2 columns added.
    """
    eig_cols = [c for c in eigenvalues_df.columns if c.startswith("Eigenvalue_")][:2]
    if len(eig_cols) < 2:
        eig_cols = [c for c in eigenvalues_df.columns if c.startswith("Eigenvalue_")]

    eig_subset = eigenvalues_df[eig_cols]

    result = {}
    for ticker, df in data_dict.items():
        df = df.copy()
        # Align eigenvalue features to sector's date index
        aligned = eig_subset.reindex(df.index)
        for col in eig_cols:
            df[col] = aligned[col]
        result[ticker] = df

    injected_count = sum(1 for df in result.values() if "Eigenvalue_1" in df.columns)
    print(f"  [INJECT] Added {eig_cols} to {injected_count}/{len(data_dict)} sectors")
    return result
