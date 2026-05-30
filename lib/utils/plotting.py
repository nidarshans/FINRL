import matplotlib.pyplot as plt
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def plot_kvo_macd_weekly_hist(df: pl.DataFrame, ticker: str):
    
    ticker_df = (
        df.filter(pl.col("Ticker") == ticker)
        .sort("Date")
    )

    #Plot in plotly. With z-scored close price
    pdf = ticker_df.with_columns([
        ((pl.col("Close") - pl.col("Close").mean()) / pl.col("Close").std()).alias("Close_Z")
    ]).to_pandas()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pdf["Date"],
            y=pdf["Close_Z"],
            name="Close (z-score)",
            line=dict(color="#378ADD", width=1.8),
        )
    )
    #Plot hist for kvo and macd on separate subplots, with secondary y-axis for hist
    fig.add_trace(
        go.Bar(
            x=pdf["Date"],
            y=pdf["W_KLINGER_HIST"],
            name="W_KLINGER_HIST",
        )
    )
    fig.add_trace(
        go.Bar(
            x=pdf["Date"],
            y=pdf["W_MACD_HIST"],
            name="W_MACD_HIST",
        )
    )

    fig.update_layout(
        title=dict(text=f"{ticker} — Close price vs Klinger and MACD Signals (z-scored)", font_size=15),
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
            ((pl.col("Close") - pl.col("Close").mean()) / pl.col("Close").std()).alias("Close_Z")
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
            y=ticker_df[ratio_col],
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

def plot_hmm_regimes(ticker_df, ticker_name) -> go.Figure:
    # Convert to pandas for easier plotting with datetime handling
    pdf = ticker_df.to_pandas()
    pdf['Date'] = pd.to_datetime(pdf['Date'])

    # Dynamically find regime probability columns
    prob_cols = sorted([col for col in pdf.columns if col.startswith('Regime_Prob_')])
    num_regimes = len(prob_cols)
    
    if num_regimes > 0:
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08,
            subplot_titles=(f"{ticker_name} Close Price with Overlaid Regimes", "Regime Probabilities Over Time")
        )
        
        # Plot Close Price base line on Row 1 (in a clean slate gray)
        fig.add_trace(
            go.Scatter(
                x=pdf['Date'],
                y=pdf['Close'],
                name="Close Price",
                line=dict(color="#7F8C8D", width=1.5),
            ),
            row=1, col=1
        )
        
        # Plot Regime Probabilities and Overlay Markers
        colors = ["#2ECC71", "#E74C3C", "#9B59B6", "#F1C40F", "#1ABC9C", "#E67E22"]
        for idx, col in enumerate(prob_cols):
            color = colors[idx % len(colors)]
            
            # Identify periods where this regime is the predicted argmax active state
            regime_mask = pdf['Regime'] == idx
            regime_df = pdf[regime_mask]
            
            # Overlay Regime i markers on Close Price plot (Row 1)
            fig.add_trace(
                go.Scatter(
                    x=regime_df['Date'],
                    y=regime_df['Close'],
                    mode='markers',
                    name=f"Regime {idx} Price Markers",
                    marker=dict(color=color, size=5.5, opacity=0.85),
                    legendgroup=f"Regime {idx}",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Plot Regime i Probability line on Row 2
            fig.add_trace(
                go.Scatter(
                    x=pdf['Date'],
                    y=pdf[col],
                    name=f"Regime {idx} Prob",
                    line=dict(color=color, width=1.8),
                    legendgroup=f"Regime {idx}",
                    showlegend=True
                ),
                row=2, col=1
            )
            
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Probability", range=[-0.05, 1.05], row=2, col=1)
    else:
        # Fallback to standard 2-subplot layout if no probability columns exist
        unique_regimes = sorted(pdf['Regime'].unique())
        num_unique = len(unique_regimes)
        
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08,
            subplot_titles=(f"{ticker_name} Close Price with Overlaid Regimes", f"Market Regime Classification")
        )
        
        # Plot Close Price on Row 1
        fig.add_trace(
            go.Scatter(
                x=pdf['Date'],
                y=pdf['Close'],
                name="Close Price",
                line=dict(color="#7F8C8D", width=1.5),
            ),
            row=1, col=1
        )
        
        # Plot step Regime on Row 2
        fig.add_trace(
            go.Scatter(
                x=pdf['Date'],
                y=pdf['Regime'],
                name="Regime Class",
                line=dict(color="#34495E", width=2, shape="hv"),
            ),
            row=2, col=1
        )
        
        # Plot markers on Row 1 for each unique regime using a matplotlib colormap converted to hex
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        cmap = plt.colormaps.get_cmap('tab10' if num_unique <= 10 else 'tab20')
        colors = [cmap(i) for i in np.linspace(0, 1, num_unique)]
        
        for idx, r in enumerate(unique_regimes):
            hex_color = mcolors.to_hex(colors[idx])
            regime_df = pdf[pdf['Regime'] == r]
            fig.add_trace(
                go.Scatter(
                    x=regime_df['Date'],
                    y=regime_df['Close'],
                    mode='markers',
                    name=f"Regime {r} Price Markers",
                    marker=dict(color=hex_color, size=5.5, opacity=0.85),
                    legendgroup=f"Regime {r}",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Regime Class", row=2, col=1)

    # Style Layout
    fig.update_layout(
        title=dict(text=f"HMM Market Regime Analysis — {ticker_name}", font_size=16, x=0.5, font=dict(weight="bold")),
        height=650,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.03, x=0, yanchor="bottom"),
        margin=dict(l=60, r=40, t=80, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    # Configure grids and axes
    fig.update_xaxes(showgrid=True, gridcolor="#EBEBEB", row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#EBEBEB", title_text="Date", row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#EBEBEB", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#EBEBEB", row=2, col=1)

    return fig