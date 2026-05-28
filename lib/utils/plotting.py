import matplotlib.pyplot as plt
import polars as pl
import plotly.graph_objects as go
import pandas as pd

def plot_price_vs_vwap(
    df: pl.DataFrame,
    ticker: str,
    date_col: str = "Date",
    close_col: str = "Close",
    vwap_col: str = "VWAP",
    ticker_col: str = "Ticker",
):
    """
    Plot stock Close price vs VWAP for a single ticker.
    """

    # Filter ticker
    stock_df = (
        df.filter(pl.col(ticker_col) == ticker)
        .sort(date_col)
    )

    # Convert to pandas for matplotlib
    pdf = stock_df.select(
        [date_col, close_col, vwap_col]
    ).to_pandas()

    # Plot
    plt.figure(figsize=(12, 6))

    plt.plot(
        pdf[date_col],
        pdf[close_col],
        label="Close Price",
    )

    plt.plot(
        pdf[date_col],
        pdf[vwap_col],
        label="VWAP",
    )

    plt.title(f"{ticker} Close Price vs VWAP")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_ticker_amihud(df: pl.DataFrame, ticker: str, window_size: int = 5):
    """
    Filters a multi-ticker panel DataFrame for a specific asset and plots
    its Close price alongside its rolling Amihud Illiquidity Ratio.
    """
    rolling_col = f"Amihud_{window_size}d"

    # 1. Filter for the target ticker and drop initial execution nulls
    ticker_df = (
        df.filter(pl.col("Ticker") == ticker)
        .filter(pl.col(rolling_col).is_not_null())
    )

    if ticker_df.is_empty():
        print(f"No data or Amihud metrics found for ticker: {ticker}")
        return

    # 2. Extract Polars columns into NumPy arrays for Matplotlib
    dates = ticker_df.get_column("Date").to_numpy()
    close_prices = ticker_df.get_column("Close").to_numpy()
    amihud_values = ticker_df.get_column(rolling_col).to_numpy()

    # 3. Create the dual-axis chart
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    # --- Primary Axis: Stock Price ---
    color_price = '#1f77b4' # Line blue
    ax1.set_xlabel('Date', fontweight='bold', labelpad=10)
    ax1.set_ylabel(f'{ticker} Price ($)', color=color_price, fontweight='bold')
    ax1.plot(dates, close_prices, color=color_price, linewidth=2.5, label='Close Price')
    ax1.tick_params(axis='y', labelcolor=color_price)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # --- Secondary Axis: Amihud Ratio ---
    ax2 = ax1.twinx()
    color_amihud = '#e377c2' # Contrasting pink/magenta
    ax2.set_ylabel(f'Amihud Ratio ({window_size}d MA)', color=color_amihud, fontweight='bold')
    ax2.plot(dates, amihud_values, color=color_amihud, linewidth=1.5, linestyle='--', label='Amihud Ratio')
    ax2.tick_params(axis='y', labelcolor=color_amihud)

    # --- Title & Layout Adjustment ---
    plt.title(f'{ticker} Structure: Price vs. Amihud Illiquidity Compression',
              fontsize=13, fontweight='bold', pad=15)
    fig.tight_layout()

    plt.show() # Uncomment if working in an interactive environment

def plot_price_vs_amihud(
    df: pl.DataFrame,
    ticker: str,
    short_window: int = 5,
    long_window: int = 20,
) -> go.Figure:
    ratio_col = f"Amihud_{short_window}d_{long_window}d_Ratio"

    ticker_df = (
        df.filter(pl.col("Ticker") == ticker)
        .sort("Date")
        .with_columns([
            ((pl.col("Close") - pl.col("Close").mean()) / pl.col("Close").std()).alias("Close_Z"),
            ((pl.col(ratio_col) - pl.col(ratio_col).mean()) / pl.col(ratio_col).std()).alias("Ratio_Z"),
        ])
        .to_pandas()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ticker_df["Date"],
            y=ticker_df["Close_Z"],
            name="Close (z-score)",
            line=dict(color="#378ADD", width=1.8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ticker_df["Date"],
            y=ticker_df["Ratio_Z"],
            name=f"Amihud {short_window}d/{long_window}d (z-score)",
            line=dict(color="#D85A30", width=1.5, dash="dot"),
        )
    )

    fig.update_layout(
        title=dict(text=f"{ticker} — Close price vs Amihud ratio (z-scored)", font_size=15),
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0),
        margin=dict(l=60, r=40, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(
            title="z-score",
            showgrid=True,
            gridcolor="#EBEBEB",
            zeroline=True,
            zerolinecolor="#CCCCCC",
            zerolinewidth=1,
        ),
        xaxis=dict(showgrid=True, gridcolor="#EBEBEB"),
    )

    return fig

def plot_hmm_regimes(ticker_df, ticker_name):
    fig, ax = plt.subplots(figsize=(14, 7))

    # Convert to pandas for easier plotting with colored segments
    pdf = ticker_df.to_pandas()
    pdf['Date'] = pd.to_datetime(pdf['Date'])

    colors = ['blue', 'red', 'green']
    labels = ['Regime 0', 'Regime 1', 'Regime 2']

    for i in range(3):
        mask = pdf['Regime'] == i
        ax.scatter(pdf.loc[mask, 'Date'], pdf.loc[mask, 'Close'],
                   c=colors[i], label=labels[i], s=10, alpha=0.8)

    ax.plot(pdf['Date'], pdf['Close'], color='black', alpha=0.3, linewidth=1)
    ax.set_title(f'Market Regime Classification for {ticker_name}', fontsize=16)
    ax.set_ylabel('Price ($)')
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.show()