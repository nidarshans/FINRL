"""
GARCH(p,q) wrapper for computing volatility-adjusted (standardized) residuals.

Uses the `arch` library (Kevin Sheppard) as a hard dependency.
Each sector's return series is fitted with a GARCH model; the standardized
residuals (return / conditional_volatility) are returned as the metric.
"""

import warnings
import numpy as np
import pandas as pd
from arch import arch_model

from lib.regime_detection.src.constants import GARCH_P, GARCH_Q


def fit_garch(returns_series: pd.Series, p: int = GARCH_P, q: int = GARCH_Q) -> pd.Series:
    """
    Fit a GARCH(p,q) model to a single return series.

    Parameters
    ----------
    returns_series : pd.Series
        Daily arithmetic returns (e.g. Close.pct_change()).
    p : int
        ARCH lag order for the variance equation.
    q : int
        GARCH lag order for the variance equation.

    Returns
    -------
    pd.Series
        Standardized residuals (return / conditional_vol), same index as input
        (NaN where the model cannot estimate).
    """
    clean = returns_series.dropna()
    if len(clean) < 30:
        return pd.Series(np.nan, index=returns_series.index, name=returns_series.name)
    
    scale_factor = 100
    scaled = clean * scale_factor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(scaled, vol="Garch", p=p, q=q, mean="Zero", dist="normal")
        try:
            result = model.fit(disp="off", show_warning=False)
        except Exception:
            # If GARCH fitting fails, return NaN
            return pd.Series(np.nan, index=returns_series.index, name=returns_series.name)

    std_resid = result.std_resid
    # Reindex to original index (keeps NaN for leading values)
    return std_resid.reindex(returns_series.index)


def fit_garch_all(data_dict: dict, p: int = GARCH_P, q: int = GARCH_Q) -> pd.DataFrame:
    """
    Fit GARCH(p,q) across all sectors and return a DataFrame of standardized
    residuals aligned on a common date index.

    Parameters
    ----------
    data_dict : dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame (must have 'Close' column).
    p, q : int
        GARCH order parameters.

    Returns
    -------
    pd.DataFrame
        Index = dates, columns = sector tickers, values = standardized residuals.
    """
    residuals = {}
    for ticker, df in data_dict.items():
        returns = df["Close"].pct_change()
        residuals[ticker] = fit_garch(returns, p=p, q=q)
        print(f"  [GARCH] {ticker}: fitted GARCH({p},{q})")

    metric_df = pd.DataFrame(residuals)
    # Drop rows that are all-NaN (warmup period)
    metric_df = metric_df.dropna(how="all")
    return metric_df
