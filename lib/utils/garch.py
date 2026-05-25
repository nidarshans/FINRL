import numpy as np
from arch import arch_model
import polars as pl
from lib.utils.utils import utils_rolling_pct_change

def utils_garch_filter_series(series: pl.Series, p: int = 1, q: int = 1) -> pl.Series:
    """Fits a GARCH model to a specific Polars Series and returns conditional volatility."""
    # Drop leading nulls (caused by prior shifting/pct change calculations)
    clean_series = series.drop_nulls()

    if len(clean_series) < 10:  # Safety check for sufficient data points
        return pl.Series(
            f"{series.name}_garch_vol", [None] * len(series), dtype=pl.Float64
        )

    returns = clean_series.to_numpy()

    # Fit GARCH(p,q)
    model = arch_model(
        returns, vol="Garch", p=p, q=q, dist="normal", rescale=True
    )
    res = model.fit(disp="off")

    # If the model rescaled the data for convergence, scale it back down
    scale_factor = res.scale if hasattr(res, "scale") else 1.0
    cond_vol = res.conditional_volatility / scale_factor

    # Re-align with original series shape by padding back the dropped leading nulls
    null_padding = len(series) - len(clean_series)
    full_cond_vol = np.concatenate(([np.nan] * null_padding, cond_vol))

    return pl.Series(f"{series.name}_garch_vol", full_cond_vol)

def pipeline_add_garch(df: pl.DataFrame, metric) -> pl.DataFrame:
    """Stage 3: Calculates Pct Change and its GARCH Conditional Volatility."""

    # 1. First pass: Calculate the rolling percentage change of volume
    #    Partition by Ticker to prevent cross-ticker data leakage
    df_with_pct = df.sort(["Ticker", "Date"]).with_columns(
        utils_rolling_pct_change(metric, n=1).over("Ticker")
    )

    # 2. Second pass: Apply GARCH filter group-by-group to protect ticker isolation
    def apply_garch_per_group(group_df: pl.DataFrame) -> pl.DataFrame:
        garch = utils_garch_filter_series(group_df[f"{metric}_pct_chg_1d"])
        return group_df.with_columns(garch)

    # Use map_groups to run the python function safely over each ticker slice
    final_df = df_with_pct.group_by("Ticker", maintain_order=True).map_groups(
        apply_garch_per_group
    )

    return final_df.sort("Date", "Ticker")