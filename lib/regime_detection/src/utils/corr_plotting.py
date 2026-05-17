"""
Visualization utilities for the correlation / PCA regime detection pipeline.

Provides heatmaps, eigenvalue evolution plots, PC1 loading charts, and
overlay plots to diagnose regime change signals.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


# Set Matplotlib style for consistency
plt.style.use('dark_background')


def plot_corr_heatmap(corr_data, date, sectors: list) -> go.Figure:
    """
    Heatmap of the correlation matrix at a specific date.
    """
    if isinstance(corr_data, tuple):
        corr_matrix, labels = corr_data
    else:
        corr_matrix = corr_data
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


# ==============================================================================
# Matplotlib Fallbacks (More reliable in Colab)
# ==============================================================================

def plot_eigenvalue_evolution_plt(eigenvalues_df: pd.DataFrame, top_n: int = 3):
    """Matplotlib version of eigenvalue evolution."""
    plt.figure(figsize=(12, 5))
    cols = [c for c in eigenvalues_df.columns if c.startswith("Eigenvalue_")][:top_n]
    for col in cols:
        plt.plot(eigenvalues_df.index, eigenvalues_df[col], label=col)
    plt.title("Eigenvalue Evolution (Matplotlib)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

def plot_pc1_loadings_plt(pc1_loadings_df: pd.DataFrame, sectors: list, n_dates: int = 4):
    """Matplotlib version of PC1 loadings snapshots."""
    dates = pc1_loadings_df.index
    step = max(len(dates) // n_dates, 1)
    snapshot_dates = [dates[i] for i in range(0, len(dates), step)][:n_dates]
    
    fig, axes = plt.subplots(1, len(snapshot_dates), figsize=(4*len(snapshot_dates), 4), sharey=True)
    if len(snapshot_dates) == 1: axes = [axes]
    
    labels = sectors[: pc1_loadings_df.shape[1]]
    for i, date in enumerate(snapshot_dates):
        axes[i].bar(labels, pc1_loadings_df.loc[date])
        axes[i].set_title(str(date.date()) if hasattr(date, "date") else str(date))
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.suptitle("PC1 Sector Loadings snapshots")
    plt.tight_layout()
    plt.show()

def plot_corr_heatmap_plt(corr_data, date, sectors: list):
    """Matplotlib version of correlation heatmap."""
    if isinstance(corr_data, tuple):
        corr_matrix, labels = corr_data
    else:
        corr_matrix = corr_data
        labels = sectors[: corr_matrix.shape[0]]

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title(f"Correlation Matrix — {date}")
    plt.show()

def plot_corr_regime_overlay_plt(eigenvalues_df: pd.DataFrame, price_df: pd.DataFrame, sector: str):
    """Matplotlib version of regime overlay."""
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    if sector in price_df.columns:
        ax1.plot(price_df.index, price_df[sector], color='#4ECDC4', label=f"{sector} Price")
        ax1.set_ylabel("Price", color='#4ECDC4')
        ax1.tick_params(axis='y', labelcolor='#4ECDC4')
    
    if "Eigenvalue_1" in eigenvalues_df.columns:
        ax2 = ax1.twinx()
        ax2.fill_between(eigenvalues_df.index, 0, eigenvalues_df["Eigenvalue_1"], color='#FF6B6B', alpha=0.2, label="λ₁")
        ax2.plot(eigenvalues_df.index, eigenvalues_df["Eigenvalue_1"], color='#FF6B6B', label="λ₁")
        ax2.set_ylabel("Eigenvalue λ₁", color='#FF6B6B')
        ax2.tick_params(axis='y', labelcolor='#FF6B6B')
    
    plt.title(f"Regime Overlay — {sector} vs λ₁")
    plt.show()

def plot_absorption_ratio_plt(explained_variance_df: pd.DataFrame, k: int = 1):
    """Matplotlib version of Absorption Ratio plot (Cumulative)."""
    if explained_variance_df.empty:
        print("Explained variance DataFrame is empty.")
        return
        
    plt.figure(figsize=(12, 5))
    colors = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#45B7D1", "#96CEB4"]
    
    for i in range(1, k + 1):
        ar_series = explained_variance_df.iloc[:, :i].sum(axis=1)
        plt.plot(ar_series.index, ar_series, 
                 label=f"AR (k={i})", 
                 color=colors[(i-1) % len(colors)],
                 linewidth=2 if i == k else 1.5,
                 alpha=1.0 if i == k else 0.6)
                 
    plt.fill_between(ar_series.index, 0, ar_series, color=colors[(k-1) % len(colors)], alpha=0.05)
    plt.title(f"Cumulative Absorption Ratio Evolution (Top {k} Components)")
    plt.xlabel("Date")
    plt.ylabel("Fraction of Total Variance")
    plt.ylim(0, 1.1)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()
