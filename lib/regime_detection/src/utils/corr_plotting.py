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
    with plt.style.context('dark_background'):
        plt.figure(figsize=(12, 5))
        cols = [c for c in eigenvalues_df.columns if c.startswith("Eigenvalue_")][:top_n]
        for col in cols:
            plt.plot(eigenvalues_df.index, eigenvalues_df[col], label=col)
        plt.title("Eigenvalue Evolution (Matplotlib)", color='#e0e0e0')
        plt.xlabel("Date", color='#e0e0e0')
        plt.ylabel("Value", color='#e0e0e0')
        plt.legend(facecolor='#1e1e2e', edgecolor='#2a2a3a')
        plt.grid(alpha=0.2)
        plt.show()

def plot_pc1_loadings_plt(pc1_loadings_df: pd.DataFrame, sectors: list, n_dates: int = 4):
    """Matplotlib version of PC1 loadings snapshots."""
    with plt.style.context('dark_background'):
        dates = pc1_loadings_df.index
        step = max(len(dates) // n_dates, 1)
        snapshot_dates = [dates[i] for i in range(0, len(dates), step)][:n_dates]
        
        fig, axes = plt.subplots(1, len(snapshot_dates), figsize=(4*len(snapshot_dates), 4), sharey=True)
        if len(snapshot_dates) == 1: axes = [axes]
        
        labels = sectors[: pc1_loadings_df.shape[1]]
        for i, date in enumerate(snapshot_dates):
            axes[i].bar(labels, pc1_loadings_df.loc[date])
            axes[i].set_title(str(date.date()) if hasattr(date, "date") else str(date), color='#e0e0e0')
            axes[i].tick_params(axis='x', rotation=45, colors='#e0e0e0')
            axes[i].tick_params(axis='y', colors='#e0e0e0')
            axes[i].grid(alpha=0.1)
        
        plt.suptitle("PC1 Sector Loadings snapshots", color='#e0e0e0')
        plt.tight_layout()
        plt.show()

def plot_corr_heatmap_plt(corr_data, date, sectors: list):
    """Matplotlib version of correlation heatmap."""
    with plt.style.context('dark_background'):
        if isinstance(corr_data, tuple):
            corr_matrix, labels = corr_data
        else:
            corr_matrix = corr_data
            labels = sectors[: corr_matrix.shape[0]]

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
        plt.title(f"Correlation Matrix — {date}", color='#e0e0e0')
        plt.show()

def plot_corr_regime_overlay_plt(eigenvalues_df: pd.DataFrame, price_df: pd.DataFrame, sector: str):
    """Matplotlib version of regime overlay."""
    with plt.style.context('dark_background'):
        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        if sector in price_df.columns:
            ax1.plot(price_df.index, price_df[sector], color='#4ECDC4', label=f"{sector} Price")
            ax1.set_ylabel("Price", color='#4ECDC4')
            ax1.tick_params(axis='y', labelcolor='#4ECDC4')
            ax1.tick_params(axis='x', colors='#e0e0e0')
        
        if "Eigenvalue_1" in eigenvalues_df.columns:
            ax2 = ax1.twinx()
            ax2.fill_between(eigenvalues_df.index, 0, eigenvalues_df["Eigenvalue_1"], color='#FF6B6B', alpha=0.2, label="λ₁")
            ax2.plot(eigenvalues_df.index, eigenvalues_df["Eigenvalue_1"], color='#FF6B6B', label="λ₁")
            ax2.set_ylabel("Eigenvalue λ₁", color='#FF6B6B')
            ax2.tick_params(axis='y', labelcolor='#FF6B6B')
        
        plt.title(f"Regime Overlay — {sector} vs λ₁", color='#e0e0e0')
        plt.show()

def plot_absorption_ratio_plt(explained_variance_df: pd.DataFrame, k: int = 1):
    """Matplotlib version of Absorption Ratio plot (Cumulative)."""
    if explained_variance_df.empty:
        print("Explained variance DataFrame is empty.")
        return
        
    with plt.style.context('dark_background'):
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
        plt.title(f"Cumulative Absorption Ratio Evolution (Top {k} Components)", color='#e0e0e0')
        plt.xlabel("Date", color='#e0e0e0')
        plt.ylabel("Fraction of Total Variance", color='#e0e0e0')
        plt.ylim(0, 1.1)
        plt.grid(alpha=0.2)
        plt.legend(facecolor='#1e1e2e', edgecolor='#2a2a3a')
        plt.show()


def plot_corr_hmm_overlay(
    corr_features_df: pd.DataFrame,
    hmm_results_df: pd.DataFrame,
    price_df: pd.DataFrame,
    sector: str,
) -> go.Figure:
    """
    Stunning three-panel interactive dashboard aligning HMM regimes with correlation dynamics.
    
    Parameters
    ----------
    corr_features_df : pd.DataFrame
        Distilled correlation features including 'Absorption_Ratio', 'Corr_Mean', 'Eigenvalue_1_Delta'.
    hmm_results_df : pd.DataFrame
        HMM output including 'Regime', 'P_Bull', 'P_Bear', 'Rank_Score'.
    price_df : pd.DataFrame
        Asset prices DataFrame containing sector.
    sector : str
        Ticker symbol.
    """
    from lib.regime_detection.src.constants import REGIME_COLORS
    
    common_idx = corr_features_df.index.intersection(hmm_results_df.index).intersection(price_df.index)
    if len(common_idx) == 0:
        print("  [PLOT] No overlapping dates between dataframes.")
        return go.Figure()
        
    corr_sub = corr_features_df.loc[common_idx]
    hmm_sub = hmm_results_df.loc[common_idx]
    price_sub = price_df.loc[common_idx]
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            f"{sector} Price & HMM Predicted Regimes",
            "Systemic Risk: Absorption Ratio & Average Sector Correlation",
            "GARCH-Adjusted Absorption Ratio Shocks",
            "Correlation Acceleration (λ₁ Delta)"
        ),
        row_heights=[0.35, 0.2, 0.2, 0.25]
    )
    
    # Extract price series robustly (handles both panel format and single-sector format dataframes)
    if sector in price_sub.columns:
        price_series = price_sub[sector]
    elif 'Close' in price_sub.columns:
        price_series = price_sub['Close']
    else:
        price_series = price_sub.iloc[:, 0]

    # 1. Top Panel: Price + Regime Bands
    fig.add_trace(
        go.Scatter(
            x=common_idx,
            y=price_series,
            mode="lines",
            name=f"{sector} Price",
            line=dict(color="#FFFFFF", width=2),
        ),
        row=1, col=1
    )
    
    # Contiguous regime blocks
    regimes = hmm_sub['Regime'].values
    dates = common_idx.tolist()
    
    i = 0
    shapes = []
    while i < len(regimes):
        r = regimes[i]
        start_date = dates[i]
        while i < len(regimes) and (regimes[i] == r or (pd.isnull(regimes[i]) and pd.isnull(r))):
            i += 1
        end_date = dates[min(i, len(regimes) - 1)]
        
        if pd.notnull(r):
            color = REGIME_COLORS.get(r, '#7f7f7f')
            shapes.append(dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start_date,
                y0=0.0,
                x1=end_date,
                y1=1.0,
                fillcolor=color,
                opacity=0.15,
                layer="below",
                line_width=0,
            ))
        
    for r, col in REGIME_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=col, symbol="square"),
                name=f"{r} Regime Band",
                showlegend=True
            ),
            row=1, col=1
        )
        
    # 2. Middle Panel: Absorption Ratio & Corr Mean
    fig.add_trace(
        go.Scatter(
            x=common_idx,
            y=corr_sub['Absorption_Ratio'],
            mode="lines",
            name="Absorption Ratio",
            line=dict(color="#FF6B6B", width=2),
            fill="tozeroy",
            fillcolor="rgba(255,107,107,0.05)"
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=common_idx,
            y=corr_sub['Corr_Mean'],
            mode="lines",
            name="Avg Correlation",
            line=dict(color="#4ECDC4", width=1.5, dash="dash"),
        ),
        row=2, col=1
    )
    
    # 3. Third Panel: Absorption Ratio GARCH Residuals
    if 'Absorption_Ratio_Garch' in corr_sub.columns:
        fig.add_trace(
            go.Scatter(
                x=common_idx,
                y=corr_sub['Absorption_Ratio_Garch'],
                mode="lines",
                name="AR GARCH Residuals",
                line=dict(color="#FFD93D", width=2),
                fill="tozeroy",
                fillcolor="rgba(255,217,61,0.08)"
            ),
            row=3, col=1
        )
        
    # 4. Bottom Panel: Eigenvalue 1 Delta
    delta_vals = corr_sub['Eigenvalue_1_Delta'].fillna(0.0).values
    bar_colors = ["#F44336" if d >= 0 else "#4CAF50" for d in delta_vals]
    
    fig.add_trace(
        go.Bar(
            x=common_idx,
            y=delta_vals,
            name="λ₁ Acceleration",
            marker_color=bar_colors,
            opacity=0.8
        ),
        row=4, col=1
    )
    
    fig.update_layout(
        title=dict(
            text=f"Correlation & HMM Regime Dashboard — {sector}",
            x=0.5,
            xanchor="center",
            font=dict(size=20)
        ),
        template="plotly_dark",
        hovermode="x unified",
        height=850,
        width=1100,
        shapes=shapes,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Ratio / Corr", row=2, col=1)
    fig.update_yaxes(title_text="GARCH Resid", row=3, col=1)
    fig.update_yaxes(title_text="Change", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    
    return fig


def plot_corr_hmm_overlay_plt(
    corr_features_df: pd.DataFrame,
    hmm_results_df: pd.DataFrame,
    price_df: pd.DataFrame,
    sector: str,
):
    """Matplotlib version of the dashboard."""
    from lib.regime_detection.src.constants import REGIME_COLORS
    
    common_idx = corr_features_df.index.intersection(hmm_results_df.index).intersection(price_df.index)
    if len(common_idx) == 0:
        print("No overlapping dates.")
        return
        
    corr_sub = corr_features_df.loc[common_idx]
    hmm_sub = hmm_results_df.loc[common_idx]
    price_sub = price_df.loc[common_idx]
    
    # Extract price series robustly (handles both panel format and single-sector format dataframes)
    if sector in price_sub.columns:
        price_series = price_sub[sector]
    elif 'Close' in price_sub.columns:
        price_series = price_sub['Close']
    else:
        price_series = price_sub.iloc[:, 0]

    with plt.style.context('dark_background'):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        # Style axes
        for ax in (ax1, ax2, ax3, ax4):
            ax.tick_params(colors='#e0e0e0', which='both', labelsize=10)
            ax.grid(True, color='#2a2a3a', alpha=0.3, linestyle=':')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color('#2a2a3a')

        # 1. Top Panel
        ax1.plot(common_idx, price_series, color='white', linewidth=2, label=f"{sector} Price")
        
        # Background bands
        regimes = hmm_sub['Regime'].values
        dates = common_idx.tolist()
        i = 0
        while i < len(regimes):
            r = regimes[i]
            start_date = dates[i]
            while i < len(regimes) and (regimes[i] == r or (pd.isnull(regimes[i]) and pd.isnull(r))):
                i += 1
            end_date = dates[min(i, len(regimes) - 1)]
            if pd.notnull(r):
                color = REGIME_COLORS.get(r, '#7f7f7f')
                ax1.axvspan(start_date, end_date, color=color, alpha=0.15)
            
        ax1.set_ylabel("Price", color='#e0e0e0')
        ax1.set_title(f"{sector} Price & HMM Regimes", color='#e0e0e0', fontsize=12, fontweight='bold')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=col, alpha=0.3, label=f"{r} Regime") for r, col in REGIME_COLORS.items()]
        ax1.legend(handles=legend_elements, loc='upper left', facecolor='#1e1e2e', edgecolor='#2a2a3a', labelcolor='#e0e0e0')
        
        # 2. Second Panel
        ax2.plot(common_idx, corr_sub['Absorption_Ratio'], color='#FF6B6B', linewidth=2, label="Absorption Ratio")
        ax2.fill_between(common_idx, 0, corr_sub['Absorption_Ratio'], color='#FF6B6B', alpha=0.05)
        ax2.plot(common_idx, corr_sub['Corr_Mean'], color='#4ECDC4', linewidth=1.5, linestyle='--', label="Avg Correlation")
        ax2.set_ylabel("Ratio / Correlation", color='#e0e0e0')
        ax2.legend(loc='upper left', facecolor='#1e1e2e', edgecolor='#2a2a3a', labelcolor='#e0e0e0')
        ax2.set_title("Systemic Risk Metrics", color='#e0e0e0', fontsize=12, fontweight='bold')
        
        # 3. Third Panel
        if 'Absorption_Ratio_Garch' in corr_sub.columns:
            ax3.plot(common_idx, corr_sub['Absorption_Ratio_Garch'], color='#FFD93D', linewidth=2, label="AR GARCH Residuals")
            ax3.fill_between(common_idx, 0, corr_sub['Absorption_Ratio_Garch'], color='#FFD93D', alpha=0.1)
        ax3.set_ylabel("GARCH Residual", color='#e0e0e0')
        ax3.legend(loc='upper left', facecolor='#1e1e2e', edgecolor='#2a2a3a', labelcolor='#e0e0e0')
        ax3.set_title("GARCH-Adjusted Absorption Ratio Shocks", color='#e0e0e0', fontsize=12, fontweight='bold')
        
        # 4. Bottom Panel
        delta_vals = corr_sub['Eigenvalue_1_Delta'].fillna(0.0).values
        bar_colors = ["#F44336" if d >= 0 else "#4CAF50" for d in delta_vals]
        ax4.bar(common_idx, delta_vals, color=bar_colors, width=1.0)
        ax4.set_ylabel("λ₁ Delta", color='#e0e0e0')
        ax4.set_xlabel("Date", color='#e0e0e0')
        ax4.set_title("Correlation Acceleration (λ₁ Delta)", color='#e0e0e0', fontsize=12, fontweight='bold')
        
        plt.suptitle(f"HMM Regime & Correlation Analysis — {sector}", color='#e0e0e0', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
