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


def build_spectral_figure(result: WalkForwardResult):
    """Return a Plotly line figure for spectral feature evolution."""

    import plotly.express as px

    id_columns = {"decision_date", "split_index"}
    value_columns = [column for column in result.spectral_features.columns if column not in id_columns]
    long = result.spectral_features.unpivot(
        index=["decision_date", "split_index"],
        on=value_columns,
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
    result.spectral_features.write_csv(path / "spectral_features.csv")
    build_performance_figure(result).write_html(path / "performance_vs_spy.html")
    build_spectral_figure(result).write_html(path / "spectral_features.html")

