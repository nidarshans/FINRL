"""Reporting and Plotly visualization helpers for experiments."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from finrl.backtest.results import WalkForwardResult


def build_performance_figure(result: WalkForwardResult):
    """Return a Plotly figure comparing portfolio and SPY equity curves."""

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.portfolio_curve["decision_date"],
            y=result.portfolio_curve["equity"],
            mode="lines",
            name="Portfolio",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.spy_curve["decision_date"],
            y=result.spy_curve["equity"],
            mode="lines",
            name="S&P 500 / SPY",
        )
    )
    fig.update_layout(
        title="Walk-Forward Performance vs S&P 500",
        xaxis_title="Decision Date",
        yaxis_title="Growth of $1",
        template="plotly_white",
    )
    return fig


def build_spectral_figure(
    result: WalkForwardResult,
    value_columns: tuple[str, ...] | None = None,
):
    """Return a Plotly line figure for spectral feature evolution."""

    import plotly.express as px

    id_columns = {"decision_date", "split_index"}
    if value_columns is None:
        selected_columns = [
            column for column in result.spectral_features.columns if column not in id_columns
        ]
    else:
        missing = set(value_columns).difference(result.spectral_features.columns)
        if missing:
            raise ValueError(f"Missing spectral columns: {sorted(missing)}")
        selected_columns = list(value_columns)
    if not selected_columns:
        return px.line(
            pl.DataFrame({"decision_date": [], "value": [], "spectral_feature": []}),
            x="decision_date",
            y="value",
            color="spectral_feature",
            title="Spectral Feature Evolution",
            template="plotly_white",
        )
    long = result.spectral_features.unpivot(
        index=["decision_date", "split_index"],
        on=selected_columns,
        variable_name="spectral_feature",
        value_name="value",
    )
    return px.line(
        long,
        x="decision_date",
        y="value",
        color="spectral_feature",
        title="Spectral Feature Evolution",
        template="plotly_white",
    )


def build_allocation_figure(
    result: WalkForwardResult,
    include_cash: bool = True,
    top_n: int | None = None,
):
    """Return a stacked area figure for portfolio allocation over time."""

    import plotly.express as px

    id_columns = {"decision_date", "split_index"}
    asset_columns = [column for column in result.allocations.columns if column not in id_columns]
    if not include_cash:
        asset_columns = [column for column in asset_columns if column != "CASH"]
    if top_n is not None:
        if top_n <= 0:
            raise ValueError("top_n must be positive.")
        means = result.allocations.select(asset_columns).mean().row(0, named=True)
        asset_columns = sorted(asset_columns, key=lambda column: means[column], reverse=True)[:top_n]
    long = result.allocations.unpivot(
        index=["decision_date", "split_index"],
        on=asset_columns,
        variable_name="asset",
        value_name="weight",
    )
    return px.area(
        long,
        x="decision_date",
        y="weight",
        color="asset",
        title="Portfolio Allocation Over Time",
        template="plotly_white",
    )


def build_holdings_heatmap_granular(
    result: WalkForwardResult,
    min_weight: float = 0.001,
    top_n: int | None = None,
    freq: str | None = None,
    height_per_ticker: int = 35,
    include_cash: bool = False,
):
    """Return a granular heatmap of portfolio weights over time."""

    import plotly.graph_objects as go

    if min_weight < 0.0:
        raise ValueError("min_weight must be non-negative.")
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be positive.")
    if height_per_ticker <= 0:
        raise ValueError("height_per_ticker must be positive.")

    id_columns = {"decision_date", "split_index"}
    asset_columns = [column for column in result.allocations.columns if column not in id_columns]
    if not include_cash:
        asset_columns = [column for column in asset_columns if column != "CASH"]
    if not asset_columns:
        return _empty_holdings_heatmap(height=600)

    allocations = result.allocations.select(["decision_date", *asset_columns]).sort("decision_date")
    if freq is not None:
        allocations = (
            allocations.group_by_dynamic(
                "decision_date",
                every=_plotly_resample_frequency(freq),
            )
            .agg(pl.all().exclude("decision_date").last())
            .sort("decision_date")
        )

    active_columns = []
    for column in asset_columns:
        if bool(allocations.select((pl.col(column).abs() > min_weight).any()).item()):
            active_columns.append(column)
    if top_n is not None:
        usage = allocations.select(
            [pl.col(column).abs().sum().alias(column) for column in active_columns]
        ).row(0, named=True)
        active_columns = sorted(active_columns, key=lambda column: usage[column], reverse=True)[:top_n]
    if not active_columns:
        return _empty_holdings_heatmap(height=600)

    weights = allocations.select(active_columns).to_numpy().T
    dates = allocations.get_column("decision_date").to_list()
    zmax = max(float(allocations.select(active_columns).max().max_horizontal().max()), 1e-9)
    fig = go.Figure(
        data=go.Heatmap(
            z=weights,
            x=dates,
            y=active_columns,
            zmin=0,
            zmax=zmax,
            colorbar={"title": "Weight"},
            hovertemplate=(
                "Date: %{x}<br>"
                "Ticker: %{y}<br>"
                "Weight: %{z:.2%}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Portfolio Holdings Over Time",
        xaxis_title="Date",
        yaxis_title="Ticker",
        height=max(600, height_per_ticker * len(active_columns)),
        hovermode="closest",
        template="plotly_white",
        xaxis={
            "rangeslider": {"visible": True},
            "rangeselector": {
                "buttons": [
                    {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                    {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                    {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ]
            },
        },
    )
    return fig


def _plotly_resample_frequency(freq: str) -> str:
    """Map common pandas-style plot frequencies to Polars dynamic windows."""

    mapping = {
        "D": "1d",
        "W": "1w",
        "M": "1mo",
        "ME": "1mo",
        "Q": "3mo",
        "QE": "3mo",
        "Y": "1y",
        "YE": "1y",
    }
    return mapping.get(freq.upper(), freq)


def _empty_holdings_heatmap(height: int):
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Heatmap(z=[], x=[], y=[]))
    fig.update_layout(
        title="Portfolio Holdings Over Time",
        xaxis_title="Date",
        yaxis_title="Ticker",
        height=height,
        template="plotly_white",
    )
    return fig


def build_regime_portfolio_figure(result: WalkForwardResult):
    """Return a two-panel Plotly figure for portfolio equity and HMM regimes."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    id_columns = {"decision_date", "split_index"}
    regime_columns = [
        column
        for column in result.regime_probabilities.columns
        if column not in id_columns
    ]
    if not regime_columns:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(
                x=result.portfolio_curve["decision_date"],
                y=result.portfolio_curve["equity"],
                mode="lines",
                name="Portfolio",
            )
        )
        fig.update_layout(template="plotly_white", title="Portfolio Equity")
        return fig

    probabilities = result.regime_probabilities.select(regime_columns)
    dates = result.regime_probabilities["decision_date"]
    palette = (
        "#2ecc71",
        "#e74c3c",
        "#3498db",
        "#f39c12",
        "#9b59b6",
        "#16a085",
    )
    portfolio = result.portfolio_curve.join(
        result.regime_probabilities.select(["decision_date", *regime_columns]),
        on="decision_date",
        how="inner",
    )
    portfolio_dates = portfolio["decision_date"]
    portfolio_equity = portfolio["equity"]
    portfolio_dominant = portfolio.select(regime_columns).to_numpy().argmax(axis=1)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Portfolio Equity with Overlaid Regimes",
            "Regime Probabilities Over Time",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=portfolio_dates,
            y=portfolio_equity,
            mode="lines",
            name="Portfolio",
            line={"color": "#7f8c8d", "width": 1.5},
        ),
        row=1,
        col=1,
    )
    for regime_index, column in enumerate(regime_columns):
        mask = portfolio_dominant == regime_index
        fig.add_trace(
            go.Scatter(
                x=portfolio_dates.filter(mask),
                y=portfolio_equity.filter(mask),
                mode="markers",
                name=f"Regime {regime_index}",
                marker={
                    "color": palette[regime_index % len(palette)],
                    "size": 6,
                },
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    for regime_index, column in enumerate(regime_columns):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=result.regime_probabilities[column],
                mode="lines",
                name=f"Regime {regime_index} Prob",
                line={
                    "color": palette[regime_index % len(palette)],
                    "width": 2,
                },
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        title="HMM Market Regime Analysis — Portfolio",
        template="plotly_white",
        legend_title_text="Series",
        hovermode="x unified",
        height=720,
    )
    fig.update_xaxes(title_text="Decision Date", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Equity", row=1, col=1)
    fig.update_yaxes(title_text="Probability", range=[0.0, 1.05], row=2, col=1)
    return fig


def metrics_to_frame(result: WalkForwardResult) -> pl.DataFrame:
    """Convert split metrics to a Polars table."""

    rows = []
    for split_result in result.split_results:
        metrics = split_result.metrics
        benchmark = split_result.benchmark_metrics
        rows.append(
            {
                "split_index": split_result.split_index,
                "test_start": split_result.test_start,
                "test_end": split_result.test_end,
                "portfolio_cumulative_return": metrics.cumulative_return,
                "spy_cumulative_return": benchmark.cumulative_return,
                "spy_relative_alpha": metrics.spy_relative_alpha,
                "portfolio_max_drawdown": metrics.max_drawdown,
                "portfolio_mean_turnover": metrics.mean_turnover,
                "portfolio_total_transaction_cost": metrics.total_transaction_cost,
            }
        )
    return pl.DataFrame(rows)


def write_report(result: WalkForwardResult, output_dir: str | Path) -> None:
    """Write metrics, curves, and Plotly HTML reports."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    metrics_to_frame(result).write_csv(path / "split_metrics.csv")
    result.portfolio_curve.write_csv(path / "portfolio_curve.csv")
    result.spy_curve.write_csv(path / "spy_curve.csv")
    result.allocations.write_csv(path / "allocations.csv")
    result.regime_probabilities.write_csv(path / "regime_probabilities.csv")
    result.spectral_features.write_csv(path / "spectral_features.csv")
    build_performance_figure(result).write_html(path / "performance_vs_spy.html")
    build_allocation_figure(result).write_html(path / "allocations.html")
    build_holdings_heatmap_granular(result).write_html(path / "holdings_heatmap.html")
    build_regime_portfolio_figure(result).write_html(path / "regime_portfolio.html")
    build_spectral_figure(result).write_html(path / "spectral_features.html")
