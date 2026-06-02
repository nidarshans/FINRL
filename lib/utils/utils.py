import polars as pl
import numpy as np

def utils_standardize_rolling(
    df: pl.DataFrame,
    ticker_columns: dict[str, int],   # {col: window}
    min_periods: int | None = None,
) -> pl.DataFrame:
    """
    Roll-wise standardization per ticker + market-wide normalization.

    - For ticker_columns: rolling z-score per ticker.
    - For market_columns: rolling z-score on the whole column (no over("Ticker")).
    """

    ticker_exprs = [
        ((
            pl.col(c) - pl.col(c).rolling_mean(w, min_periods=min_periods or w).over("Ticker")
        ) / pl.col(c).rolling_std(w, min_periods=min_periods or w).over("Ticker")
        ).alias(c)
        for c, w in ticker_columns.items()
    ]
    '''
    market_exprs = [
        ((
            pl.col(c) - pl.col(c).rolling_mean(w, min_periods=min_periods or w)
        ) / pl.col(c).rolling_std(w, min_periods=min_periods or w)
        ).alias(c)
        for c, w in market_columns.items()
    ]
    '''
    return df.with_columns(ticker_exprs)

def utils_rolling_pct_change(col_name: str, n: int = 1) -> pl.Expr:
    """Returns a Polars expression calculating percentage change over 'n' periods.

    Formula: (x_t / x_{t-n}) - 1
    """
    return (
        ((pl.col(col_name) / pl.col(col_name).shift(n)) - 1).alias(
            f"{col_name}_pct_chg_{n}d"
        )
    )

def ewma_zscore(
    X: np.ndarray,
    span: int = 20,
    eps: float = 1e-8,
):
    """
    EWMA normalization.
    """

    alpha = 2 / (span + 1)

    T, N = X.shape

    mean = np.zeros((T, N))
    var = np.zeros((T, N))

    mean[0] = X[0]

    for t in range(1, T):

        mean[t] = (
            alpha * X[t]
            + (1 - alpha) * mean[t - 1]
        )

        var[t] = (
            alpha * (X[t] - mean[t])**2
            + (1 - alpha) * var[t - 1]
        )

    std = np.sqrt(var + eps)

    return (X - mean) / std