import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from lib.regime_detection.src.constants import SECTOR_COLORS

def plot_results(result, weights_aligned, decoded_all, close_prices, output_path,
                 wf_windows=None, mode_label="Train/Test"):
    print("\n=== PLOTTING ===")
    tickers = list(decoded_all.keys())
    colors  = {t: SECTOR_COLORS[i % len(SECTOR_COLORS)] for i, t in enumerate(tickers)}
    n_panels = 5 if wf_windows else 4
    fig_height = 22 if not wf_windows else 27
    fig = plt.figure(figsize=(20, fig_height))
    fig.patch.set_facecolor('#0f0f1a')
    gs = gridspec.GridSpec(n_panels, 1, figure=fig, hspace=0.45,
                           height_ratios=[2.5, 1.2, 2.5, 2.5] if not wf_windows else [2.5, 1.2, 2.5, 2.5, 1.2])

    text_color, grid_color, bg_color = '#e0e0e0', '#2a2a3a', '#16162a'
    ax_style = dict(facecolor=bg_color)

    # 1. Equity Curves
    ax1 = fig.add_subplot(gs[0], **ax_style)
    prices = result.prices
    for col in prices.columns:
        norm = prices[col] / prices[col].iloc[0] * 100
        ax1.plot(prices.index, norm, label=col, lw=2.5 if "HMM" in col else 1.5,
                 ls='-' if "HMM" in col else '--', color='#00e5ff' if "HMM" in col else '#ff9800')
    ax1.legend(facecolor='#1e1e2e', labelcolor=text_color)
    ax1.set_title("Equity Curves", color=text_color)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1], **ax_style)
    for col in prices.columns:
        dd = (prices[col] / prices[col].cummax() - 1) * 100
        ax2.plot(dd.index, dd, color='#00e5ff' if "HMM" in col else '#ff9800', label=col)
    ax2.set_title("Drawdown (%)", color=text_color)

    # 3. Allocation
    ax3 = fig.add_subplot(gs[2], **ax_style)
    wa_w = weights_aligned.resample('W').last().ffill().fillna(0)
    bottom = np.zeros(len(wa_w))
    for ticker in wa_w.columns:
        if wa_w[ticker].sum() > 0:
            ax3.fill_between(wa_w.index, bottom, bottom + wa_w[ticker], color=colors.get(ticker, '#ccc'), label=ticker)
            bottom += wa_w[ticker]
    ax3.set_title("Sector Allocation", color=text_color)

    # 4. Regimes
    ax4 = fig.add_subplot(gs[3], **ax_style)
    regime_map = {'Bull': 1, 'Stagnant': 0, 'Bear': -1}
    regime_frames = {t: df['Regime'].map(regime_map) for t, df in decoded_all.items() if 'Regime' in df.columns}
    if regime_frames:
        rdf = pd.DataFrame(regime_frames).resample('W').last().ffill().reindex(columns=wa_w.columns).dropna(how='all')
        ax4.imshow(rdf.T.values, aspect='auto', cmap=ListedColormap(['#F44336', '#FFC107', '#4CAF50']), vmin=-1, vmax=1,
                   extent=[matplotlib.dates.date2num(rdf.index[0]), matplotlib.dates.date2num(rdf.index[-1]), -0.5, len(rdf.columns)-0.5])
        ax4.set_yticks(range(len(rdf.columns)))
        ax4.set_yticklabels(rdf.columns, color=text_color)
        ax4.xaxis_date()

    # 5. WF Map
    if wf_windows:
        ax5 = fig.add_subplot(gs[4], **ax_style)
        for idx, (tr_s, tr_e, oos_s, oos_e) in enumerate(wf_windows):
            y = idx % 3
            ax5.barh(y, (pd.Timestamp(tr_e) - pd.Timestamp(tr_s)).days, left=matplotlib.dates.date2num(pd.Timestamp(tr_s)), color='#1565C0', height=0.35)
            ax5.barh(y, (pd.Timestamp(oos_e) - pd.Timestamp(oos_s)).days, left=matplotlib.dates.date2num(pd.Timestamp(oos_s)), color='#00e5ff', height=0.35)
        ax5.xaxis_date()

    plt.show()


def print_stats(result):
    print("\n" + "=" * 60 + "\nPERFORMANCE SUMMARY\n" + "=" * 60)
    stats = result.stats.T
    for col in ['total_return', 'cagr', 'max_drawdown', 'sharpe', 'daily_vol']:
        if col in stats.columns:
            vals = " | ".join(f"{n}: {v:.4f}" for n, v in stats[col].items())
            print(f"  {col:<22} {vals}")
    print("=" * 60)
