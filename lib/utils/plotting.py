import matplotlib.pyplot as plt
import polars as pl


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