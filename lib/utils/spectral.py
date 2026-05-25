import numpy as np
import polars as pl
from typing import Dict, List
from sklearn.covariance import LedoitWolf
from numpy.linalg import eigvalsh

from lib.utils.ingest import build_hierarchy_maps, create_feature_matrix
from lib.utils.utils import ewma_zscore

def rolling_correlation_matrices(
    X: np.ndarray,
    window: int = 60,
):
    """
    Returns:

        corr_matrices[T, N, N]
    """

    T, N = X.shape

    matrices = []

    for t in range(window, T + 1):

        W = X[t - window:t]

        C = np.corrcoef(W.T)

        C = np.nan_to_num(C)

        matrices.append(C)

    return np.stack(matrices)

def spectral_features_from_matrix(
    C: np.ndarray
):
    """
    Computes spectral descriptors.
    """

    eigenvalues = eigvalsh(C)

    eigenvalues = np.clip(
        eigenvalues,
        1e-12,
        None
    )

    eigenvalues = np.sort(eigenvalues)[::-1]

    # --------------------------------------------------------
    # Largest Eigenvalue
    # --------------------------------------------------------

    lambda1 = eigenvalues[0]

    # --------------------------------------------------------
    # Spectral Gap
    # --------------------------------------------------------

    if len(eigenvalues) > 1:
        spectral_gap = (
            eigenvalues[0]
            - eigenvalues[1]
        )
    else:
        spectral_gap = 0.0

    # --------------------------------------------------------
    # Spectral Entropy
    # --------------------------------------------------------

    p = eigenvalues / eigenvalues.sum()

    spectral_entropy = -np.sum(
        p * np.log(p)
    )

    # --------------------------------------------------------
    # Participation Ratio
    # --------------------------------------------------------

    participation_ratio = (
        np.sum(eigenvalues)**2
        /
        np.sum(eigenvalues**2)
    )

    return {
        "largest_eigenvalue": lambda1,
        "spectral_gap": spectral_gap,
        "spectral_entropy": spectral_entropy,
        "participation_ratio": participation_ratio,
    }

def compute_hierarchical_spectral_features(
    corr_matrices: np.ndarray,
    dates,
    tickers: List[str],
    gics_dict: Dict,
):
    """
    Computes:

        - global spectral features
        - sector spectral features
        - industry spectral features
    """

    (
        ticker_to_sector,
        ticker_to_industry,
        sector_to_tickers,
        industry_to_tickers,
    ) = build_hierarchy_maps(gics_dict)

    ticker_to_idx = {
        t: i
        for i, t in enumerate(tickers)
    }

    # Precompute indices outside the time loop to save redundant computations
    valid_sectors = {}
    for sector, sector_tickers in sector_to_tickers.items():
        idx = [ticker_to_idx[t] for t in sector_tickers if t in ticker_to_idx]
        if len(idx) >= 2:
            valid_sectors[sector] = idx

    valid_industries = {}
    for industry, industry_tickers in industry_to_tickers.items():
        idx = [ticker_to_idx[t] for t in industry_tickers if t in ticker_to_idx]
        if len(idx) >= 2:
            valid_industries[industry] = idx

    rows = []

    # ========================================================
    # LOOP OVER TIME
    # ========================================================

    for t_idx, C in enumerate(corr_matrices):

        date = dates[t_idx]

        # ====================================================
        # GLOBAL FEATURES
        # ====================================================

        global_features = spectral_features_from_matrix(C)

        rows.append({
            "Date": date,
            "Level": "global",
            "Group": "ALL",
            **global_features
        })

        # ====================================================
        # SECTOR FEATURES
        # ====================================================

        for sector, idx in valid_sectors.items():

            C_sector = C[np.ix_(idx, idx)]

            features = spectral_features_from_matrix(
                C_sector
            )

            rows.append({
                "Date": date,
                "Level": "sector",
                "Group": sector,
                **features
            })

        # ====================================================
        # INDUSTRY FEATURES
        # ====================================================

        for industry, idx in valid_industries.items():

            C_industry = C[np.ix_(idx, idx)]

            features = spectral_features_from_matrix(
                C_industry
            )

            rows.append({
                "Date": date,
                "Level": "industry",
                "Group": industry,
                **features
            })

    return pl.DataFrame(rows)

def pipeline_hierarchical_spectral_features(
    df: pl.DataFrame,
    gics_dict: Dict,
    feature_col: str,
    ewma_span: int = 20,
    corr_window: int = 60,
):
    """
    Full hierarchical spectral pipeline.
    """

    # --------------------------------------------------------
    # FEATURE MATRIX
    # --------------------------------------------------------

    X, tickers, dates = create_feature_matrix(
        df=df,
        feature_col=feature_col,
    )

    # --------------------------------------------------------
    # EWMA NORMALIZATION
    # --------------------------------------------------------

    X_norm = ewma_zscore(
        X,
        span=ewma_span,
    )

    # --------------------------------------------------------
    # ROLLING CORRELATION MATRICES
    # --------------------------------------------------------

    corr_matrices = rolling_correlation_matrices(
        X_norm,
        window=corr_window,
    )

    # --------------------------------------------------------
    # ALIGN DATES
    # --------------------------------------------------------

    aligned_dates = dates[corr_window - 1:]

    # --------------------------------------------------------
    # HIERARCHICAL SPECTRAL FEATURES
    # --------------------------------------------------------

    hierarchical_df = (
        compute_hierarchical_spectral_features(
            corr_matrices=corr_matrices,
            dates=aligned_dates,
            tickers=tickers,
            gics_dict=gics_dict,
        )
    )

    return hierarchical_df