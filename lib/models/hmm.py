import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM


def train_sector_regime_hmm(
    df_features_panel: pl.DataFrame,
    df_subsector_risk: pl.DataFrame,
    subsector_name: str,
    tickers: list[str],
    n_regimes: int = 3,
    metric: str = ""
) -> pl.DataFrame:
    """Combines individual stock features with structural spectral features to train

    a Gaussian Hidden Markov Model for market regime detection.
    """
    # -----------------------------------------------------------------
    # STEP 1: Process and Align Asset Features
    # -----------------------------------------------------------------
    # Calculate log returns and volume changes for individual stocks
    df_assets_processed = (
        df_features_panel.filter(pl.col("Ticker").is_in(tickers))
        .sort(["Ticker", "Date"])
        .with_columns([
            (pl.col("Close").log() - pl.col("Close").log().shift(1)).over("Ticker").alias("Log_Return"),
            (pl.col("Volume").log() - pl.col("Volume").log().shift(1)).over("Ticker").alias("Volume_Delta"),

            # Cast both new split boolean features to floats for the matrix
            pl.col("Signal_MACD_Bullish").cast(pl.Float64).alias("MACD_Flag"),
            pl.col("Signal_KVO_Bullish").cast(pl.Float64).alias("KVO_Flag")
        ])
        .drop_nulls()
    )

    # -----------------------------------------------------------------
    # STEP 2: Pivot Asset Columns to Align with Spectral Risk
    # -----------------------------------------------------------------
    # Pivot individual asset metrics into wide format to join with sector data chronologically
    df_wide_assets = df_assets_processed.pivot(
        index="Date",
        on="Ticker",
        values=["Log_Return", "Volume_Delta", "KVO_Flag", "MACD_Flag"],
        aggregate_function="first"
    )

    # Filter down to our targeted subsector's PC1 Eigenvalue
    df_spectral = df_subsector_risk.filter(
        pl.col("Sub_Industry") == subsector_name
    ).select(["Date", pl.col(f"{metric}_PC1").alias("PC1_Eigenvalue")])

    # Construct the complete synchronized master training matrix
    df_training_master = df_wide_assets.join(
        df_spectral, on="Date", how="inner"
    ).sort("Date")

    # -----------------------------------------------------------------
    # STEP 3: Extract & Normalize the Feature Matrix
    # -----------------------------------------------------------------
    # Drop non-feature identification columns
    feature_cols = [
        c for c in df_training_master.columns if c not in ["Date", "Sub_Industry"]
    ]
    X = df_training_master.select(feature_cols).to_numpy()

    # Apply z-score normalization to handle variations in asset pricing vs. eigenvalues
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # -----------------------------------------------------------------
    # STEP 4: Train the Unsupervised Gaussian HMM
    # -----------------------------------------------------------------
    # covariance_type="diag" prevents matrix singularity and weights collapse
    hmm_model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="diag",
        n_iter=100,
        random_state=42,
    )
    hmm_model.fit(X_scaled)

    # Predict hidden states (0, 1, or 2)
    hidden_regimes = hmm_model.predict(X_scaled)

    # Append our discovered states back into our main frame
    df_regimes_output = df_training_master.with_columns(
        pl.Series("Hidden_Regime", hidden_regimes)
    )

    return df_regimes_output, hmm_model