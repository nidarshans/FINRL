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

from lib.regime_detection.src.filters.garch import fit_garch_all, fit_garch


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
        One of "garch_returns", "volume" (GARCH-filtered percent change), "raw_returns".

    Returns
    -------
    pd.DataFrame
        Index = dates, columns = sector tickers, values = metric values.
    """
    if metric == "garch_returns":
        return _metric_garch(data_dict)
    elif metric == "volume":
        return _metric_volume(data_dict)
    elif metric == "raw_returns":
        return _metric_raw_returns(data_dict)
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Choose from: 'garch_returns', 'volume', 'raw_returns'"
        )


def _metric_garch(data_dict: dict) -> pd.DataFrame:
    """GARCH-adjusted standardized residuals across all sectors."""
    return fit_garch_all(data_dict, p=GARCH_P, q=GARCH_Q)


def _metric_volume(data_dict: dict) -> pd.DataFrame:
    """GARCH-filtered volume percent change across all sectors."""
    volume_series = {}
    for ticker, df in data_dict.items():
        if "Volume" in df.columns:
            # 1. Compute volume percent change
            # Replace inf/-inf with nan to avoid numerical issues
            vol_pct = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
            
            # 2. Apply GARCH filter to the volume series
            vol_garch = fit_garch(vol_pct, p=GARCH_P, q=GARCH_Q)
            volume_series[ticker] = vol_garch

    metric_df = pd.DataFrame(volume_series)
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
    dict[pd.Timestamp, tuple(np.ndarray, list)]
        Mapping of date → (corr_matrix, valid_columns).
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

        # Store the minimal correlation matrix and the list of valid columns
        corr_dict[date] = (corr, valid_cols.tolist())

    from lib.regime_detection.src.constants import VERBOSE
    if VERBOSE:
        print(f"  [CORR] Built {len(corr_dict)} rolling correlation matrices "
              f"(window={window})")
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
        Fraction of variance explained by each component relative to available sectors.
    """
    if not corr_dict:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    dates = sorted(corr_dict.keys())
    # Use provided sectors or infer from metadata in corr_dict
    if sectors is None:
        # Fallback: union of all valid_cols seen across all dates
        all_seen = set()
        for _, cols in corr_dict.values():
            all_seen.update(cols)
        all_sectors = sorted(list(all_seen))
    else:
        all_sectors = sectors

    n_sectors = len(all_sectors)
    k = n_components if n_components is not None else n_sectors

    eigenvalue_rows = []
    pc1_loading_rows = []
    explained_rows = []

    for date in dates:
        corr, valid_cols = corr_dict[date]
        n_valid = len(valid_cols)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(corr)

        # Reverse to descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clamp negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Top-k eigenvalues for this date
        current_k = min(k, n_valid)
        top_eigenvalues = np.zeros(k)
        top_eigenvalues[:current_k] = eigenvalues[:current_k]
        eigenvalue_rows.append(top_eigenvalues)

        # PC1 loadings (padded to full n_sectors size)
        full_pc1 = np.zeros(n_sectors)
        if n_valid > 0:
            # Map valid loadings back to their original sector indices
            valid_idx = [all_sectors.index(c) for c in valid_cols]
            for i, vi in enumerate(valid_idx):
                full_pc1[vi] = eigenvectors[i, 0]
        pc1_loading_rows.append(full_pc1)

        # Explained variance ratio (relative to n_valid)
        total = eigenvalues.sum()
        explained = np.zeros(k)
        if total > 0:
            explained[:current_k] = eigenvalues[:current_k] / total
        explained_rows.append(explained)

    # Build DataFrames
    eig_cols = [f"Eigenvalue_{i+1}" for i in range(k)]
    eigenvalues_df = pd.DataFrame(eigenvalue_rows, index=dates, columns=eig_cols)

    sector_labels = all_sectors
    pc1_loadings_df = pd.DataFrame(pc1_loading_rows, index=dates, columns=sector_labels)

    var_cols = [f"PC_{i+1}_var" for i in range(k)]
    explained_variance_df = pd.DataFrame(explained_rows, index=dates, columns=var_cols)

    from lib.regime_detection.src.constants import VERBOSE
    if VERBOSE:
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

    from lib.regime_detection.src.constants import VERBOSE
    if VERBOSE:
        print(f"  [INJECT] Added {eig_cols} to {len(result)}/{len(data_dict)} sectors")
    return result


def compute_corr_features(
    corr_dict: dict,
    eigenvalues_df: pd.DataFrame,
    explained_variance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distil rolling correlation matrices + PCA results into a per-date
    feature DataFrame for HMM consumption.
    """
    if not corr_dict or eigenvalues_df.empty or explained_variance_df.empty:
        return pd.DataFrame()

    from lib.regime_detection.src.constants import CORR_DELTA_WINDOW

    dates = sorted(corr_dict.keys())
    
    corr_means = []
    corr_disps = []
    
    for date in dates:
        corr, valid_cols = corr_dict[date]
        if corr.shape[0] > 1:
            mask = ~np.eye(corr.shape[0], dtype=bool)
            off_diag = corr[mask]
            corr_means.append(off_diag.mean())
            corr_disps.append(off_diag.std())
        else:
            corr_means.append(0.0)
            corr_disps.append(0.0)
            
    out_df = pd.DataFrame(index=dates)
    out_df['Eigenvalue_1'] = eigenvalues_df['Eigenvalue_1']
    if 'Eigenvalue_2' in eigenvalues_df.columns:
        out_df['Eigenvalue_2'] = eigenvalues_df['Eigenvalue_2']
    else:
        out_df['Eigenvalue_2'] = 0.0
    out_df['Absorption_Ratio'] = explained_variance_df['PC_1_var']
    
    from lib.regime_detection.src.filters.garch import fit_garch
    ar_returns = out_df['Absorption_Ratio'].pct_change()
    out_df['Absorption_Ratio_Garch'] = fit_garch(ar_returns)
    
    out_df['Corr_Mean'] = corr_means
    out_df['Corr_Dispersion'] = corr_disps
    out_df['Eigenvalue_1_Delta'] = eigenvalues_df['Eigenvalue_1'].diff(periods=CORR_DELTA_WINDOW)
    
    return out_df
