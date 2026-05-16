"""
Visualization utilities for the correlation / PCA regime detection pipeline.

Provides heatmaps, eigenvalue evolution plots, PC1 loading charts, and
overlay plots to diagnose regime change signals.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_corr_heatmap(corr_matrix: np.ndarray, date, sectors: list) -> go.Figure:
    """
    Heatmap of the correlation matrix at a specific date.

    Parameters
    ----------
    corr_matrix : np.ndarray
        (n_sectors × n_sectors) correlation matrix.
    date : str or pd.Timestamp
        Date label for the title.
    sectors : list[str]
        Sector ticker labels.

    Returns
    -------
    go.Figure
    """
    labels = sectors[: corr_matrix.shape[0]]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    fig.update_layout(
        title=f"Cross-Sector Correlation Matrix — {date}",
        xaxis_title="Sector",
        yaxis_title="Sector",
        width=700,
        height=600,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_eigenvalue_evolution(eigenvalues_df: pd.DataFrame, top_n: int = 3) -> go.Figure:
    """
    Time series of the top-N eigenvalues from the rolling PCA.

    Spikes in λ₁ indicate increasing cross-sector correlation (regime change).

    Parameters
    ----------
    eigenvalues_df : pd.DataFrame
        Output of run_pca_on_corr().
    top_n : int
        Number of top eigenvalues to plot.

    Returns
    -------
    go.Figure
    """
    cols = [c for c in eigenvalues_df.columns if c.startswith("Eigenvalue_")][:top_n]

    fig = go.Figure()
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

    for i, col in enumerate(cols):
        fig.add_trace(
            go.Scatter(
                x=eigenvalues_df.index,
                y=eigenvalues_df[col],
                mode="lines",
                name=f"λ{i+1}",
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

    fig.update_layout(
        title="Eigenvalue Evolution — Rolling Correlation PCA",
        xaxis_title="Date",
        yaxis_title="Eigenvalue",
        template="plotly_dark",
        hovermode="x unified",
        width=1000,
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_pc1_loadings(
    pc1_loadings_df: pd.DataFrame, sectors: list, n_dates: int = 4
) -> go.Figure:
    """
    Bar charts of PC1 sector loadings at evenly-spaced snapshot dates.

    Parameters
    ----------
    pc1_loadings_df : pd.DataFrame
        Output of run_pca_on_corr().
    sectors : list[str]
        Sector ticker labels.
    n_dates : int
        Number of snapshot dates to display.

    Returns
    -------
    go.Figure
    """
    dates = pc1_loadings_df.index
    step = max(len(dates) // n_dates, 1)
    snapshot_dates = [dates[i] for i in range(0, len(dates), step)][:n_dates]

    fig = make_subplots(
        rows=1,
        cols=len(snapshot_dates),
        subplot_titles=[str(d.date()) if hasattr(d, "date") else str(d) for d in snapshot_dates],
        shared_yaxes=True,
    )

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    labels = sectors[: pc1_loadings_df.shape[1]]

    for i, date in enumerate(snapshot_dates):
        loadings = pc1_loadings_df.loc[date].values
        fig.add_trace(
            go.Bar(
                x=labels,
                y=loadings,
                marker_color=colors[i % len(colors)],
                name=str(date.date()) if hasattr(date, "date") else str(date),
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        title="PC1 Sector Loadings — Snapshots",
        template="plotly_dark",
        height=400,
        width=300 * len(snapshot_dates),
    )
    return fig


def plot_corr_regime_overlay(
    eigenvalues_df: pd.DataFrame,
    price_df: pd.DataFrame,
    sector: str,
) -> go.Figure:
    """
    Overlay the first eigenvalue evolution with a sector's price to visualise
    regime timing.

    Parameters
    ----------
    eigenvalues_df : pd.DataFrame
        Must contain 'Eigenvalue_1' column.
    price_df : pd.DataFrame
        Must contain the specified sector column.
    sector : str
        Sector ticker to overlay.

    Returns
    -------
    go.Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Price
    if sector in price_df.columns:
        fig.add_trace(
            go.Scatter(
                x=price_df.index,
                y=price_df[sector],
                mode="lines",
                name=f"{sector} Price",
                line=dict(color="#4ECDC4", width=1.5),
            ),
            secondary_y=False,
        )

    # Eigenvalue
    if "Eigenvalue_1" in eigenvalues_df.columns:
        fig.add_trace(
            go.Scatter(
                x=eigenvalues_df.index,
                y=eigenvalues_df["Eigenvalue_1"],
                mode="lines",
                name="λ₁ (Correlation)",
                line=dict(color="#FF6B6B", width=2),
                fill="tozeroy",
                fillcolor="rgba(255,107,107,0.15)",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=f"Regime Overlay — {sector} Price vs λ₁",
        template="plotly_dark",
        hovermode="x unified",
        width=1000,
        height=450,
    )
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Eigenvalue λ₁", secondary_y=True)
    return fig
