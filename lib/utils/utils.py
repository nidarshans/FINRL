import polars as pl
import numpy as np

def utils_standardize(
    col_name: str, rolling_window: int = None
) -> pl.Expr:
    """Returns a Polars expression to Z-score standardize a column.

    If rolling_window is provided, computes a rolling Z-score.
    """
    col = pl.col(col_name)

    if rolling_window:
        # Prevent look-ahead bias with a moving window
        mean = col.rolling_mean(window_size=rolling_window)
        std = col.rolling_std(window_size=rolling_window)
    else:
        # Global calculation
        mean = col.mean()
        std = col.std()

    return ((col - mean) / std).alias(f"{col_name}_zscore")

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