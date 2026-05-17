"""
Correlation matrix construction, PCA analysis, and feature injection pipeline.

Workflow:
  1. compute_metric()       — dispatch to garch_returns / volume / raw_returns
  2. build_corr_matrix()    — rolling correlation matrix from metric DataFrame
  3. run_pca_on_corr()      — eigendecomposition per date
  4. inject_pca_features()  — merge eigenvalue columns into per-sector DataFrames
  5. compute_corr_features()— distil PCA + correlation stats into HMM feature table
"""

import numpy as np
import pandas as pd

from lib.regime_detection.src.constants import (
    CORR_DELTA_WINDOW, CORR_METRIC, CORR_WINDOW,
    GARCH_P, GARCH_Q, PCA_N_COMPONENTS, VERBOSE,
)
from lib.regime_detection.src.filters.garch import fit_garch, fit_garch_all


# ==============================================================================
# Helpers
# ==============================================================================


def _log(msg: str) -> None:
    if VERBOSE:
        print(msg)


# ==============================================================================
# 1. Metric Computation
# ==============================================================================

def compute_metric(data_dict: dict, metric: str = CORR_METRIC) -> pd.DataFrame:
    """
    Build the cross-sector panel used as input to the rolling correlation matrix.

    Parameters
    ----------
    data_dict : dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame.
    metric : {"garch_returns", "volume", "raw_returns"}

    Returns
    -------
    pd.DataFrame  —  index=dates, columns=tickers, values=metric values.
    """
    dispatch = {
        "garch_returns": _metric_garch,
        "volume":        _metric_volume,
        "raw_returns":   _metric_raw_returns,
    }
    if metric not in dispatch:
        raise ValueError(f"Unknown metric {metric!r}. Choose from: {list(dispatch)}")
    return dispatch[metric](data_dict)


def _metric_garch(data_dict: dict) -> pd.DataFrame:
    """GARCH-standardised residuals for each sector."""
    return fit_garch_all(data_dict, p=GARCH_P, q=GARCH_Q)


def _metric_volume(data_dict: dict) -> pd.DataFrame:
    """GARCH-filtered volume percent-change for each sector."""
    series = {}
    for ticker, df in data_dict.items():
        if "Volume" not in df.columns:
            continue
        vol_pct = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)

        # Guard: skip constant or near-constant series — these produce
        # degenerate covariance matrices inside GARCH.
        if vol_pct.dropna().std() < 1e-8:
            continue

        series[ticker] = fit_garch(vol_pct, p=GARCH_P, q=GARCH_Q)

    return pd.DataFrame(series).dropna(how="all")


def _metric_raw_returns(data_dict: dict) -> pd.DataFrame:
    """Simple close-to-close returns for each sector."""
    series = {ticker: df["Close"].pct_change() for ticker, df in data_dict.items()}
    return pd.DataFrame(series).dropna(how="all")


# ==============================================================================
# 2. Rolling Correlation Matrix
# ==============================================================================

def build_corr_matrix(metric_df: pd.DataFrame, window: int = CORR_WINDOW) -> dict:
    """
    Compute a rolling Pearson correlation matrix across sectors.

    Parameters
    ----------
    metric_df : pd.DataFrame  —  index=dates, columns=tickers.
    window : int              —  rolling window in trading days.

    Returns
    -------
    dict[pd.Timestamp, tuple[np.ndarray, list[str]]]
        date → (corr_matrix, valid_column_names)
        Only dates with enough data are included.
    """
    min_obs = max(window // 2, 5)
    corr_dict = {}

    for i in range(window - 1, len(metric_df)):
        date = metric_df.index[i]
        window_slice = metric_df.iloc[i - window + 1 : i + 1]

        # Drop columns with too few observations
        valid_cols = window_slice.columns[window_slice.notna().sum() >= min_obs].tolist()
        if len(valid_cols) < 2:
            continue
        corr_dict[date] = (window_slice[valid_cols].corr().values, valid_cols)

    _log(f"  [CORR] Built {len(corr_dict)} rolling correlation matrices (window={window})")
    return corr_dict


# ==============================================================================
# 3. PCA on Correlation Matrices
# ==============================================================================

def run_pca_on_corr(
    corr_dict: dict,
    sectors: list = None,
    n_components: int = PCA_N_COMPONENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Eigendecompose each rolling correlation matrix and collect results.

    Parameters
    ----------
    corr_dict : dict[pd.Timestamp, tuple[np.ndarray, list[str]]]
        Output of build_corr_matrix().
    sectors : list[str], optional
        Canonical sector order for PC1 loading alignment.
        Inferred from corr_dict if not provided.
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    eigenvalues_df        : pd.DataFrame  —  columns = ['Eigenvalue_1', ...]
    pc1_loadings_df       : pd.DataFrame  —  columns = sector names
    explained_variance_df : pd.DataFrame  —  columns = ['PC_1_var', ...]
    """
    if not corr_dict:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    dates = sorted(corr_dict.keys())

    if sectors is None:
        all_cols = (col for _, cols in corr_dict.values() for col in cols)
        sectors = sorted(set(all_cols))

    k = n_components if n_components is not None else len(sectors)

    eigenvalue_rows, pc1_loading_rows, explained_rows = [], [], []

    for date in dates:
        corr, valid_cols = corr_dict[date]
        n_valid = len(valid_cols)
        current_k = min(k, n_valid)

        # eigh returns ascending eigenvalues; reverse both together to get descending
        raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(corr)
        eigenvalues = np.maximum(raw_eigenvalues[::-1], 0.0)
        eigenvectors = raw_eigenvectors[:, ::-1]

        # --- Eigenvalues (NaN-padded to length k) ---
        top_k = np.full(k, np.nan)
        top_k[:current_k] = eigenvalues[:current_k]
        eigenvalue_rows.append(top_k)

        # --- PC1 loadings with stable sign convention ---
        pc1 = eigenvectors[:, 0].copy()
        if pc1[np.argmax(np.abs(pc1))] < 0:
            pc1 = -pc1

        full_pc1 = pd.Series(pc1, index=valid_cols).reindex(sectors, fill_value=0.0)
        pc1_loading_rows.append(full_pc1.values)

        # --- Explained variance (NaN-padded to length k) ---
        total_variance = eigenvalues.sum()  # sum of clamped eigenvalues
        explained = np.full(k, np.nan)
        if total_variance > 0:
            explained[:current_k] = eigenvalues[:current_k] / total_variance
        explained_rows.append(explained)

    eigenvalues_df = pd.DataFrame(
        eigenvalue_rows, index=dates,
        columns=[f"Eigenvalue_{i+1}" for i in range(k)],
    )
    pc1_loadings_df = pd.DataFrame(
        pc1_loading_rows, index=dates, columns=sectors,
    )
    explained_variance_df = pd.DataFrame(
        explained_rows, index=dates,
        columns=[f"PC_{i+1}_var" for i in range(k)],
    )

    _log(f"  [PCA] Computed eigenvalues for {len(dates)} dates, retaining {k} components")
    return eigenvalues_df, pc1_loadings_df, explained_variance_df


# ==============================================================================
# 4. Feature Injection
# ==============================================================================

def inject_pca_features(data_dict: dict, eigenvalues_df: pd.DataFrame) -> dict:
    """
    Attach the top-2 eigenvalue columns to each sector's DataFrame.

    Parameters
    ----------
    data_dict      : dict[str, pd.DataFrame]  —  ticker → OHLCV+features.
    eigenvalues_df : pd.DataFrame             —  output of run_pca_on_corr().

    Returns
    -------
    dict[str, pd.DataFrame]  —  same structure, with Eigenvalue_1/2 added.
    """
    eig_cols = [c for c in eigenvalues_df.columns if c.startswith("Eigenvalue_")][:2]
    eig_subset = eigenvalues_df[eig_cols]

    result = {
        ticker: df.assign(**eig_subset.reindex(df.index))
        for ticker, df in data_dict.items()
    }

    _log(f"  [INJECT] Added {eig_cols} to {len(result)}/{len(data_dict)} sectors")
    return result


# ==============================================================================
# 5. Correlation Feature Table (for HMM)
# ==============================================================================

def compute_corr_features(
    corr_dict: dict,
    eigenvalues_df: pd.DataFrame,
    explained_variance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distil rolling correlation matrices + PCA results into a per-date
    feature DataFrame for HMM consumption.

    Columns produced
    ----------------
    Eigenvalue_1/2        : top PCA eigenvalues
    Absorption_Ratio      : fraction of variance explained by PC1
    Absorption_Ratio_Garch: GARCH-filtered changes in Absorption_Ratio
    Corr_Mean             : mean off-diagonal correlation
    Corr_Dispersion       : std  off-diagonal correlation
    Eigenvalue_1_Delta    : change in Eigenvalue_1 over CORR_DELTA_WINDOW days
    """
    if not corr_dict or eigenvalues_df.empty or explained_variance_df.empty:
        return pd.DataFrame()

    dates = sorted(corr_dict.keys())

    # Off-diagonal correlation stats per date
    corr_means, corr_disps = [], []
    for date in dates:
        corr, _ = corr_dict[date]
        if corr.shape[0] > 1:
            off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
            corr_means.append(off_diag.mean())
            corr_disps.append(off_diag.std())
        else:
            corr_means.append(0.0)
            corr_disps.append(0.0)

    out = pd.DataFrame(index=dates)
    out["Eigenvalue_1"]   = eigenvalues_df["Eigenvalue_1"]
    out["Eigenvalue_2"]   = eigenvalues_df.get("Eigenvalue_2", 0.0)
    out["Absorption_Ratio"] = explained_variance_df["PC_1_var"]
    out["Corr_Mean"]        = corr_means
    out["Corr_Dispersion"]  = corr_disps
    out["Eigenvalue_1_Delta"] = eigenvalues_df["Eigenvalue_1"].diff(periods=CORR_DELTA_WINDOW)

    # GARCH-filter the Absorption Ratio changes.
    # Guard against constant/degenerate series (all-zero AR → zero variance → degenerate GARCH).
    ar_returns = out["Absorption_Ratio"].pct_change().replace([np.inf, -np.inf], np.nan)
    if ar_returns.dropna().std() > 1e-8:
        out["Absorption_Ratio_Garch"] = fit_garch(ar_returns)
    else:
        out["Absorption_Ratio_Garch"] = np.nan

    return out