from lib.utils.spectral import utils_calculate_covariance_matrix, utils_calculate_correlation_matrix, utils_calculate_eigenvalues
import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import yfinance as yf

def utils_calculate_rolling_eigenvalues(
    df: pl.DataFrame,
    column: str,
    lookback_window: int,
    method: str = "covariance",
) -> pl.DataFrame:
    """Calculates a chronological series of the 1st and 2nd eigenvalues over a rolling window

    by directly leveraging existing matrix and eigenvalue utility functions.
    """
    method = method.lower()
    if method not in ["covariance", "correlation"]:
        raise ValueError("Method must be either 'covariance' or 'correlation'")

    # 1. Isolate the chronological sequence of unique dates in the data
    unique_dates = df["Date"].unique().sort()
    total_days = len(unique_dates)

    if total_days < lookback_window:
        raise ValueError(
            f"Available historical dates ({total_days}) is less than the lookback window ({lookback_window})."
        )

    rolling_dates = []
    pc1_series = []
    pc2_series = []

    # 2. Map across the historical timeline using a windowed filter slice
    for i in range(lookback_window, total_days + 1):
        window_dates = unique_dates[i - lookback_window : i]
        current_date = window_dates[-1]

        # Isolate rows belonging exclusively to this specific historical time window
        window_df = df.filter(pl.col("Date").is_in(window_dates))

        try:
            # REUSE UTILITIES: Calculate the requested matrix type using your clean signatures
            if method == "covariance":
                matrix_df = utils_calculate_covariance_matrix(
                    window_df, column=column
                )
            else:
                matrix_df = utils_calculate_correlation_matrix(
                    window_df, column=column
                )

            # REUSE UTILITY: Extract eigenvalues directly from the matrix dataframe
            eigen_df = utils_calculate_eigenvalues(matrix_df)

            # Safely grab the top 2 values out of the resulting Polars Series
            pc1 = eigen_df["Eigenvalue"][0] if len(eigen_df) >= 1 else np.nan
            pc2 = eigen_df["Eigenvalue"][1] if len(eigen_df) >= 2 else np.nan

        except ValueError:
            # Catches windows wiped out by non-overlapping nulls
            pc1, pc2 = np.nan, np.nan

        # Append data points chronologically
        rolling_dates.append(current_date)
        pc1_series.append(pc1)
        pc2_series.append(pc2)

    # 3. Deliver a clean, structured chronological timeseries dataframe
    return pl.DataFrame(
        {
            "Date": rolling_dates,
            f"{column}_eigen_PC1": pc1_series,
            f"{column}_eigen_PC2": pc2_series,
        }
    ).sort("Date")


def utils_plot_rolling_eigenvalues(
    rolling_eigen_df: pl.DataFrame, column: str
) -> None:
    """Plots the historical series of the 1st and 2nd eigenvalues using Plotly."""
    pc1_col = f"{column}_eigen_PC1"
    pc2_col = f"{column}_eigen_PC2"

    # Create the interactive chart figure
    fig = go.Figure()

    # Add Principal Component 1 Trace
    fig.add_trace(
        go.Scatter(
            x=rolling_eigen_df["Date"].to_list(),
            y=rolling_eigen_df[pc1_col].to_list(),
            mode="lines",
            name="Principal Component 1 (PC1)",
            line=dict(color="#1f77b4", width=2.5),
        )
    )

    # Add Principal Component 2 Trace
    fig.add_trace(
        go.Scatter(
            x=rolling_eigen_df["Date"].to_list(),
            y=rolling_eigen_df[pc2_col].to_list(),
            mode="lines",
            name="Principal Component 2 (PC2)",
            line=dict(color="#ff7f0e", width=2.5, dash="dash"),
        )
    )

    # Update clean and professional chart layout
    fig.update_layout(
        title=dict(
            text=f"Rolling Eigenvalues Over Time<br><sup>Analyzed Column: {column}</sup>",
            font=dict(size=16),
        ),
        xaxis_title="Date",
        yaxis_title="Eigenvalue Value",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.show()

def utils_plot_subsector_comparison(
    df_risk_panel: pl.DataFrame,
    target_subsectors: list,
    metric_column: str,
    pc_type: str = "PC1",
) -> None:
    """Generates an interactive Plotly multi-line chart comparing the eigenvalue

    trajectories of specific GICS subsectors over time.
    """
    # 1. Target column identification based on your metric name structure
    target_value_col = f"{metric_column}_{pc_type}"

    # Verify columns exist in the incoming dataframe
    if target_value_col not in df_risk_panel.columns:
        raise KeyError(
            f"Expected metric column '{target_value_col}' not found in DataFrame panel."
        )

    # 2. Initialize the chart
    fig = go.Figure()

    # 3. Filter and plot each subsector as an independent line trace
    for subsector in target_subsectors:
        # Isolate rows for this specific subsector slice
        sub_df = df_risk_panel.filter(pl.col("Sub_Industry") == subsector).sort(
            "Date"
        )

        if sub_df.is_empty():
            print(
                f"Warning: No historical entries found for subsector '{subsector}'."
            )
            continue

        # Add line trace to the canvas
        fig.add_trace(
            go.Scatter(
                x=sub_df["Date"].to_list(),
                y=sub_df[target_value_col].to_list(),
                mode="lines",
                name=subsector,
                line=dict(width=2),
                hoverlabel=dict(namelength=-1),  # Prevents long subsector names from clipping
            )
        )

    # 4. Refine layout configuration for institutional presentation
    fig.update_layout(
        title=dict(
            text=f"Subsector Structural Risk Comparison ({pc_type})<br><sup>Underlying Feature Universe: {metric_column}</sup>",
            font=dict(size=16),
        ),
        xaxis_title="Timeline Date",
        yaxis_title=f"{pc_type} Eigenvalue Magnitude",
        hovermode="x unified",  # Groups all lines inside a single tooltip box on hover
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,  # Places legend safely on the right side of the canvas grid
        ),
        margin=dict(l=60, r=160, t=80, b=60),
    )

    # Adjust vertical formatting behavior based on the scaling method applied
    fig.update_yaxes(rangemode="tozero")

    fig.show()

def utils_plot_risk_vs_price(
    df: pl.DataFrame, tickers: list[str], title_suffix: str = ""
) -> None:
    """Plots the Custom Structural Risk Measure (PC1 Eigenvalue) against a

    Market-Cap-Weighted Price Index using an interactive Plotly dual-axis chart.

    Parameters:
    -----------
    df : pl.DataFrame
        Must contain 'Date' and your risk metric column 'PC1_Eigenvalue'.
        Must also contain individual price columns for each ticker in `tickers`.
    tickers : list[str]
        List of tickers matching the price columns present in df.
    title_suffix : str
        Optional string to append to the chart title.
    """
    # 1. Fetch live market caps to compute outstanding shares
    shares_outstanding = {}

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            mcap = info.get("marketCap")
            price = info.get("currentPrice") or info.get("regularMarketPrice")

            if mcap and price:
                shares_outstanding[ticker] = mcap / price
            else:
                shares_outstanding[ticker] = 1.0
        except Exception:
            shares_outstanding[ticker] = 1.0  # Fallback safety allocation

    # 2. Build the market-cap-weighted price expression using Polars horizontal summing
    weighted_numerator = pl.sum_horizontal(
        [pl.col(ticker) * shares_outstanding[ticker] for ticker in tickers]
    )
    total_shares = sum(shares_outstanding[ticker] for ticker in tickers)

    # Inject the calculated weighted price into our plotting view dataframe
    plot_df = df.with_columns(
        (weighted_numerator / total_shares).alias("Weighted_Sector_Price")
    ).sort("Date")

    # 3. Create the dual-axis canvas
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 4. Add Trace 1: The Market-Cap-Weighted Price Index (Left Y-Axis)
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"].to_list(),
            y=plot_df["Weighted_Sector_Price"].to_list(),
            mode="lines",
            name="Cap-Weighted Price ($)",
            line=dict(color="#1f77b4", width=2.5),  # Ocean Blue
        ),
        secondary_y=False,
    )

    # 5. Add Trace 2: The Structural Risk Measure (Right Y-Axis)
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"].to_list(),
            y=plot_df["PC1_Eigenvalue"].to_list(),
            mode="lines",
            name="Risk Measure (PC1)",
            line=dict(
                color="#d62728", width=2, dash="dashdot"
            ),  # Crimson Red Dash
        ),
        secondary_y=True,
    )

    # 6. Final Layout & Polish
    fig.update_layout(
        title=dict(
            text=f"Structural Risk (PC1) vs. Market-Cap-Weighted Price {title_suffix}",
            font=dict(size=16),
        ),
        xaxis_title="Date",
        hovermode="x unified",  # Locks tooltips together for comparison on hover
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        margin=dict(l=60, r=60, t=90, b=60),
    )

    # Set individual axis labels
    fig.update_yaxes(
        title_text="Market-Cap-Weighted Price ($)", secondary_y=False
    )
    fig.update_yaxes(
        title_text="Risk Measure (PC1 Eigenvalue)", secondary_y=True
    )

    fig.show()

def utils_plot_subsector_comparison(
    df_risk_panel: pl.DataFrame,
    target_subsectors: list,
    metric_column: str,
    pc_type: str = "PC1",
) -> None:
    """Generates an interactive Plotly multi-line chart comparing the eigenvalue

    trajectories of specific GICS subsectors over time.
    """
    # 1. Target column identification based on your metric name structure
    target_value_col = f"{metric_column}_{pc_type}"

    # Verify columns exist in the incoming dataframe
    if target_value_col not in df_risk_panel.columns:
        raise KeyError(
            f"Expected metric column '{target_value_col}' not found in DataFrame panel."
        )

    # 2. Initialize the chart
    fig = go.Figure()

    # 3. Filter and plot each subsector as an independent line trace
    for subsector in target_subsectors:
        # Isolate rows for this specific subsector slice
        sub_df = df_risk_panel.filter(pl.col("Sub_Industry") == subsector).sort(
            "Date"
        )

        if sub_df.is_empty():
            print(
                f"Warning: No historical entries found for subsector '{subsector}'."
            )
            continue

        # Add line trace to the canvas
        fig.add_trace(
            go.Scatter(
                x=sub_df["Date"].to_list(),
                y=sub_df[target_value_col].to_list(),
                mode="lines",
                name=subsector,
                line=dict(width=2),
                hoverlabel=dict(namelength=-1),  # Prevents long subsector names from clipping
            )
        )

    # 4. Refine layout configuration for institutional presentation
    fig.update_layout(
        title=dict(
            text=f"Subsector Structural Risk Comparison ({pc_type})<br><sup>Underlying Feature Universe: {metric_column}</sup>",
            font=dict(size=16),
        ),
        xaxis_title="Timeline Date",
        yaxis_title=f"{pc_type} Eigenvalue Magnitude",
        hovermode="x unified",  # Groups all lines inside a single tooltip box on hover
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,  # Places legend safely on the right side of the canvas grid
        ),
        margin=dict(l=60, r=160, t=80, b=60),
    )

    # Adjust vertical formatting behavior based on the scaling method applied
    fig.update_yaxes(rangemode="tozero")

    fig.show()

def utils_plot_risk_vs_price(
    df: pl.DataFrame, tickers: list[str], title_suffix: str = ""
) -> None:
    """Plots the Custom Structural Risk Measure (PC1 Eigenvalue) against a

    Market-Cap-Weighted Price Index using an interactive Plotly dual-axis chart.

    Parameters:
    -----------
    df : pl.DataFrame
        Must contain 'Date' and your risk metric column 'PC1_Eigenvalue'.
        Must also contain individual price columns for each ticker in `tickers`.
    tickers : list[str]
        List of tickers matching the price columns present in df.
    title_suffix : str
        Optional string to append to the chart title.
    """
    # 1. Fetch live market caps to compute outstanding shares
    shares_outstanding = {}

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            mcap = info.get("marketCap")
            price = info.get("currentPrice") or info.get("regularMarketPrice")

            if mcap and price:
                shares_outstanding[ticker] = mcap / price
            else:
                shares_outstanding[ticker] = 1.0
        except Exception:
            shares_outstanding[ticker] = 1.0  # Fallback safety allocation

    # 2. Build the market-cap-weighted price expression using Polars horizontal summing
    weighted_numerator = pl.sum_horizontal(
        [pl.col(ticker) * shares_outstanding[ticker] for ticker in tickers]
    )
    total_shares = sum(shares_outstanding[ticker] for ticker in tickers)

    # Inject the calculated weighted price into our plotting view dataframe
    plot_df = df.with_columns(
        (weighted_numerator / total_shares).alias("Weighted_Sector_Price")
    ).sort("Date")

    # 3. Create the dual-axis canvas
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 4. Add Trace 1: The Market-Cap-Weighted Price Index (Left Y-Axis)
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"].to_list(),
            y=plot_df["Weighted_Sector_Price"].to_list(),
            mode="lines",
            name="Cap-Weighted Price ($)",
            line=dict(color="#1f77b4", width=2.5),  # Ocean Blue
        ),
        secondary_y=False,
    )

    # 5. Add Trace 2: The Structural Risk Measure (Right Y-Axis)
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"].to_list(),
            y=plot_df["PC1_Eigenvalue"].to_list(),
            mode="lines",
            name="Risk Measure (PC1)",
            line=dict(
                color="#d62728", width=2, dash="dashdot"
            ),  # Crimson Red Dash
        ),
        secondary_y=True,
    )

    # 6. Final Layout & Polish
    fig.update_layout(
        title=dict(
            text=f"Structural Risk (PC1) vs. Market-Cap-Weighted Price {title_suffix}",
            font=dict(size=16),
        ),
        xaxis_title="Date",
        hovermode="x unified",  # Locks tooltips together for comparison on hover
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        margin=dict(l=60, r=60, t=90, b=60),
    )

    # Set individual axis labels
    fig.update_yaxes(
        title_text="Market-Cap-Weighted Price ($)", secondary_y=False
    )
    fig.update_yaxes(
        title_text="Risk Measure (PC1 Eigenvalue)", secondary_y=True
    )

    fig.show()

def utils_plot_indicators(df_panel: pl.DataFrame, ticker: str) -> None:
    """
    Generates an interactive 3-pane stacked Plotly layout displaying the
    asset Price, MACD (with Signal), and Klinger Volume Oscillator (with Signal).
    """
    # 1. Filter out data strictly for our target ticker
    df_ticker = df_panel.filter(pl.col("Ticker") == ticker).sort("Date")

    # Recalculate raw line metrics explicitly for the visual canvas
    df_plot = df_ticker.with_columns([
        pl.col("Close").ewm_mean(span=12, adjust=False).alias("EMA_12"),
        pl.col("Close").ewm_mean(span=26, adjust=False).alias("EMA_26"),
        ((pl.col("High") + pl.col("Low") + pl.col("Close")) / 3.0).alias("Typical_Price"),
        (pl.col("High") - pl.col("Low")).alias("Daily_Range")
    ]).with_columns([
        (pl.col("EMA_12") - pl.col("EMA_26")).alias("MACD_Line"),
        pl.when(pl.col("Typical_Price") >= pl.col("Typical_Price").shift(1)).then(1).otherwise(-1).alias("Trend_Dir")
    ]).with_columns([
        pl.when(pl.col("Daily_Range") == 0).then(0.0).otherwise(
            pl.col("Volume") * pl.col("Trend_Dir") * 100.0 * ((2.0 * ((pl.col("High") - pl.col("Low")) / pl.col("Daily_Range"))) - 1.0)
        ).alias("VF")
    ]).with_columns([
        pl.col("MACD_Line").ewm_mean(span=9, adjust=False).alias("MACD_Signal"),
        pl.col("VF").ewm_mean(span=34, adjust=False).alias("Klinger_Line")
    ]).with_columns([
        pl.col("Klinger_Line").ewm_mean(span=13, adjust=False).alias("Klinger_Signal")
    ])

    dates = df_plot["Date"].to_list()

    # 2. Initialize the 3-Row Canvas
    # Row 1: Price | Row 2: MACD | Row 3: KVO
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25]
    )

    # -------------------------------------------------------------
    # PANE 1: Close Price & Structural Signals
    # -------------------------------------------------------------
    fig.add_trace(
        go.Scatter(x=dates, y=df_plot["Close"].to_list(), name=f"{ticker} Close", line=dict(color="#2c3e50", width=2)),
        row=1, col=1
    )

    # Highlight background zones green when your boolean structural criteria is met (Both Signals >= 0)
    # This turns your target math flag into clear visual contextual zones
    boolean_signal = df_plot["Signal_KVO_MACD_Bullish"].to_list() if "Signal_KVO_MACD_Bullish" in df_plot.columns else [False]*len(dates)

    # -------------------------------------------------------------
    # PANE 2: MACD Analysis
    # -------------------------------------------------------------
    fig.add_trace(
        go.Scatter(x=dates, y=df_plot["MACD_Line"].to_list(), name="MACD Line", line=dict(color="#1f77b4", width=1.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=df_plot["MACD_Signal"].to_list(), name="MACD Signal", line=dict(color="#ff7f0e", width=1.5, dash="dot")),
        row=2, col=1
    )
    # Zero Reference Line
    fig.add_shape(type="line", x0=dates[0], y0=0, x1=dates[-1], y1=0, line=dict(color="gray", width=1, dash="dash"), row=2, col=1)

    # -------------------------------------------------------------
    # PANE 3: Klinger Volume Oscillator (KVO)
    # -------------------------------------------------------------
    fig.add_trace(
        go.Scatter(x=dates, y=df_plot["Klinger_Line"].to_list(), name="Klinger Line", line=dict(color="#2ca02c", width=1.5)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=df_plot["Klinger_Signal"].to_list(), name="Klinger Signal", line=dict(color="#d62728", width=1.5, dash="dot")),
        row=3, col=1
    )
    # Zero Reference Line
    fig.add_shape(type="line", x0=dates[0], y0=0, x1=dates[-1], y1=0, line=dict(color="gray", width=1, dash="dash"), row=3, col=1)

    # -------------------------------------------------------------
    # Layout Configurations
    # -------------------------------------------------------------
    fig.update_layout(
        title=dict(text=f"Technical Indicators Pipeline Structure: {ticker}", font=dict(size=16)),
        hovermode="x unified",
        template="plotly_white",
        height=750,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="Klinger KVO", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    fig.show()

def utils_plot_hmm_regimes(
    df_regimes: pl.DataFrame, tickers: list[str]
) -> None:
    """Plots the market-cap weighted price index calculated from log returns,

    colored dynamically by the active HMM hidden regime.
    """
    # 1. Fetch live market caps to establish dynamic weights
    shares_outstanding = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            mcap = info.get("marketCap")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            shares_outstanding[ticker] = mcap / price
        except Exception:
            shares_outstanding[ticker] = 1.0

    total_shares = sum(shares_outstanding[ticker] for ticker in tickers)

    # 2. Reconstruct prices from Log_Return columns using cumulative sums
    # Price_t = Price_0 * exp(cumsum(log_returns))
    # We will initialize the baseline index at $100 for clean scaling
    reconstructed_exprs = []
    for ticker in tickers:
        log_ret_col = f"Log_Return_{ticker}"
        if log_ret_col in df_regimes.columns:
            # Reconstruct the asset price series starting at 100
            expr = (
                (pl.col(log_ret_col).cum_sum().exp() * 100.0)
                * shares_outstanding[ticker]
            )
            reconstructed_exprs.append(expr)

    # 3. Sum horizontally across the reconstructed share pools
    weighted_num = pl.sum_horizontal(reconstructed_exprs)

    # Create the clean plotting view with the unified weighted index price
    df_plot = df_regimes.with_columns(
        (weighted_num / total_shares).alias("Weighted_Price_Index")
    ).sort("Date")

    # 4. Generate the interactive chart colored by Hidden Regime
    fig = px.scatter(
        df_plot.to_pandas(),
        x="Date",
        y="Weighted_Price_Index",
        color="Hidden_Regime",
        title="Gaussian HMM Sector Regime Tracking (Pivoted Alignment)",
        labels={
            "Weighted_Price_Index": "Cap-Weighted Index Baseline ($100)",
            "Hidden_Regime": "Regime ID",
        },
        color_continuous_scale=px.colors.qualitative.Set1,
        template="plotly_white",
    )

    # Connect the timeline with a subtle underlying gray line
    fig.add_trace(
        go.Scatter(
            x=df_plot["Date"].to_list(),
            y=df_plot["Weighted_Price_Index"].to_list(),
            mode="lines",
            line=dict(color="rgba(150, 150, 150, 0.3)", width=1),
            showlegend=False,
        )
    )

    # Ensure the Regime ID colorbar displays discrete integer blocks (0, 1, 2)
    fig.update_layout(
        height=600,
        hovermode="x unified",
        coloraxis_colorbar=dict(
            title="Regime ID",
            tickvals=df_plot["Hidden_Regime"].unique().to_list(),
            dtick=1,
        ),
    )

    fig.show()

def profile_hmm_regimes(df_regimes: pl.DataFrame, tickers: list[str]):
    """Calculates empirical metrics for each HMM hidden regime state using your

    new split features (MACD vs. KVO) to identify structural profiles.
    """
    # 1. Isolate the return columns
    log_return_cols = [
        f"Log_Return_{t}"
        for t in tickers
        if f"Log_Return_{t}" in df_regimes.columns
    ]

    # Calculate average return across all tickers for a baseline
    mean_return_expr = pl.mean_horizontal(log_return_cols).alias(
        "Avg_Asset_Return"
    )

    # 2. Build explicit expressions for the new split features
    macd_cols = [
        f"MACD_Flag_{t}" for t in tickers if f"MACD_Flag_{t}" in df_regimes.columns
    ]
    kvo_cols = [
        f"KVO_Flag_{t}" for t in tickers if f"KVO_Flag_{t}" in df_regimes.columns
    ]

    # 3. Group and aggregate by the hidden regime
    profile_df = (
        df_regimes.with_columns(mean_return_expr)
        .group_by("Hidden_Regime")
        .agg(
            [
                # Performance profile
                pl.col("Avg_Asset_Return").mean().alias("Mean_Daily_Return"),
                pl.col("Avg_Asset_Return").std().alias("Return_Volatility"),
                # Sector Risk profile
                pl.col("PC1_Eigenvalue").mean().alias("Mean_PC1_Risk"),
                # Momentum vs. Volume behavior
                pl.mean_horizontal(macd_cols).mean().alias("MACD_Bullish_Ratio"),
                pl.mean_horizontal(kvo_cols).mean().alias("KVO_Bullish_Ratio"),
                # Sample count
                pl.len().alias("Days_In_Regime"),
            ]
        )
        .sort("Hidden_Regime")
    )

    print("=== UPDATED HMM REGIME STRUCTURAL PROFILES ===")
    print(profile_df)
    return profile_df
