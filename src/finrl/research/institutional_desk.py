"""Causal, OHLCV-based analytics for an interactive institutional research desk.

The module deliberately separates observable market state from discretionary
trade scenarios. Scenario status is not a probability or a confidence estimate.
All historical structure fields become available on their confirmation date.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True, slots=True)
class DeskConfig:
    """Configuration for causal daily desk analytics."""

    period: str = "5y"
    chart_bars: int = 220
    ema_fast_span: int = 20
    ema_mid_span: int = 50
    ema_slow_span: int = 200
    atr_window: int = 20
    realized_vol_window: int = 20
    downside_vol_window: int = 60
    drawdown_window: int = 126
    momentum_short_window: int = 21
    momentum_medium_window: int = 126
    momentum_skip_window: int = 21
    high_lookback: int = 252
    volume_window: int = 20
    swing_left: int = 3
    swing_right: int = 3
    liquidity_lookback: int = 60
    zone_lookback: int = 120
    stress_lookback: int = 90
    zone_cluster_atr: float = 0.75
    zone_half_width_atr: float = 0.35
    max_trigger_distance_atr: float = 2.5
    max_stop_distance_atr: float = 2.0
    max_risk_fraction: float = 0.08
    target_1_r_multiple: float = 2.0
    target_2_r_multiple: float = 3.0

    def __post_init__(self) -> None:
        integer_fields = (
            "chart_bars",
            "ema_fast_span",
            "ema_mid_span",
            "ema_slow_span",
            "atr_window",
            "realized_vol_window",
            "downside_vol_window",
            "drawdown_window",
            "momentum_short_window",
            "momentum_medium_window",
            "high_lookback",
            "volume_window",
            "swing_left",
            "swing_right",
            "liquidity_lookback",
            "zone_lookback",
            "stress_lookback",
        )
        if any(int(getattr(self, field)) <= 0 for field in integer_fields):
            raise ValueError("Desk lookbacks and spans must be positive.")
        if self.momentum_skip_window < 0:
            raise ValueError("momentum_skip_window must be non-negative.")
        if not (
            self.ema_fast_span < self.ema_mid_span < self.ema_slow_span
        ):
            raise ValueError("EMA spans must be strictly increasing.")
        if self.momentum_skip_window >= self.momentum_medium_window:
            raise ValueError("Momentum skip window must be shorter than its horizon.")
        positive_floats = (
            "zone_cluster_atr",
            "zone_half_width_atr",
            "max_trigger_distance_atr",
            "max_stop_distance_atr",
            "max_risk_fraction",
            "target_1_r_multiple",
            "target_2_r_multiple",
        )
        if any(
            not math.isfinite(float(getattr(self, field)))
            or float(getattr(self, field)) <= 0.0
            for field in positive_floats
        ):
            raise ValueError("Desk risk and zone parameters must be finite and positive.")
        if self.target_2_r_multiple <= self.target_1_r_multiple:
            raise ValueError("Target 2 must use a larger R multiple than target 1.")

    @property
    def minimum_bars(self) -> int:
        """Minimum history required for a fully populated desk snapshot."""

        return max(
            self.ema_slow_span,
            self.high_lookback,
            self.momentum_medium_window,
            self.drawdown_window,
            self.zone_lookback,
        ) + self.swing_right + 1


@dataclass(frozen=True, slots=True)
class TradeScenario:
    """A rule-based scenario with explicit trigger and invalidation levels."""

    direction: str
    status: str
    eligible: bool
    trigger: float
    stop: float
    target_1: float
    target_2: float
    risk_per_share: float
    risk_fraction: float
    reward_risk_1: float
    reward_risk_2: float
    trigger_distance_atr: float
    thesis: str
    invalidation: str


@dataclass(frozen=True, slots=True)
class DeskSnapshot:
    """Latest observable state for one ticker."""

    ticker: str
    as_of: pd.Timestamp
    last_close: float
    primary_trend: str
    trend_score: float
    market_structure: str
    structure_score: float
    momentum_21d: float
    momentum_126_21d: float
    near_52w_high: float
    realized_vol_20: float
    downside_vol_60: float
    max_drawdown_126: float
    atr: float
    volume_z_20: float
    price_volume_state: str
    log_adv_20: float
    amihud_20: float
    support: float
    resistance: float
    demand_zone_low: float
    demand_zone_high: float
    demand_zone_touches: int
    supply_zone_low: float
    supply_zone_high: float
    supply_zone_touches: int
    swing_avwap: float
    stress_avwap: float
    avwap_state: str
    bars_since_swing_high: float
    bars_since_swing_low: float
    long_scenario: TradeScenario
    bearish_scenario: TradeScenario


@dataclass(frozen=True, slots=True)
class DeskAnalysis:
    """Desk snapshot plus the single immutable downloaded history used for it."""

    snapshot: DeskSnapshot
    history: pd.DataFrame


DataDownloader = Callable[[str, DeskConfig], pd.DataFrame]


def download_ohlcv(ticker: str, config: DeskConfig) -> pd.DataFrame:
    """Download one adjusted daily OHLCV history from Yahoo Finance."""

    import yfinance as yf

    data = yf.download(
        ticker,
        period=config.period,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return prepare_ohlcv(data, config)


def prepare_ohlcv(data: pd.DataFrame, config: DeskConfig) -> pd.DataFrame:
    """Validate and normalize an adjusted daily OHLCV table."""

    if data.empty:
        raise ValueError("OHLCV history is empty.")
    if isinstance(data.columns, pd.MultiIndex):
        raise ValueError("prepare_ohlcv expects one ticker, not MultiIndex columns.")
    normalized = data.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    required = ("open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    normalized = normalized.loc[:, list(required)].copy()
    normalized.index = pd.DatetimeIndex(normalized.index).tz_localize(None)
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    for column in required:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.replace([np.inf, -np.inf], np.nan).dropna()
    if len(normalized) < config.minimum_bars:
        raise ValueError(
            f"At least {config.minimum_bars} completed bars are required; "
            f"received {len(normalized)}."
        )
    if (normalized[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise ValueError("OHLC prices must be positive.")
    if (normalized["volume"] < 0.0).any():
        raise ValueError("Volume cannot be negative.")
    if (
        (normalized["high"] < normalized[["open", "close"]].max(axis=1)).any()
        or (normalized["low"] > normalized[["open", "close"]].min(axis=1)).any()
        or (normalized["high"] < normalized["low"]).any()
    ):
        raise ValueError("OHLC bar geometry is invalid.")
    return normalized


def _rolling_lagged_zscore(series: pd.Series, window: int) -> pd.Series:
    lagged = series.shift(1)
    mean = lagged.rolling(window, min_periods=window).mean()
    std = lagged.rolling(window, min_periods=window).std(ddof=1)
    return (series - mean) / std.replace(0.0, np.nan)


def _candidate_swings(
    high: pd.Series,
    low: pd.Series,
    left: int,
    right: int,
) -> tuple[pd.Series, pd.Series]:
    swing_high = pd.Series(True, index=high.index, dtype=bool)
    swing_low = pd.Series(True, index=low.index, dtype=bool)
    for offset in range(1, left + 1):
        swing_high &= high > high.shift(offset)
        swing_low &= low < low.shift(offset)
    for offset in range(1, right + 1):
        swing_high &= high >= high.shift(-offset)
        swing_low &= low <= low.shift(-offset)
    return swing_high.fillna(False), swing_low.fillna(False)


def _add_confirmed_structure(features: pd.DataFrame, config: DeskConfig) -> pd.DataFrame:
    output = features.copy()
    swing_high, swing_low = _candidate_swings(
        output["high"],
        output["low"],
        config.swing_left,
        config.swing_right,
    )
    positions = pd.Series(np.arange(len(output), dtype=float), index=output.index)
    output["swing_high_pivot_visual"] = swing_high
    output["swing_low_pivot_visual"] = swing_low
    output["confirmed_swing_high_event"] = output["high"].where(swing_high).shift(
        config.swing_right
    )
    output["confirmed_swing_low_event"] = output["low"].where(swing_low).shift(
        config.swing_right
    )
    output["confirmed_swing_high_pivot_pos"] = positions.where(swing_high).shift(
        config.swing_right
    )
    output["confirmed_swing_low_pivot_pos"] = positions.where(swing_low).shift(
        config.swing_right
    )

    last_highs: list[float] = []
    last_lows: list[float] = []
    structure_labels: list[str] = []
    structure_scores: list[float] = []
    high_ages: list[float] = []
    low_ages: list[float] = []
    last_high_confirmation: int | None = None
    last_low_confirmation: int | None = None
    for position, (_, row) in enumerate(output.iterrows()):
        if pd.notna(row["confirmed_swing_high_event"]):
            last_highs.append(float(row["confirmed_swing_high_event"]))
            last_highs = last_highs[-2:]
            last_high_confirmation = position
        if pd.notna(row["confirmed_swing_low_event"]):
            last_lows.append(float(row["confirmed_swing_low_event"]))
            last_lows = last_lows[-2:]
            last_low_confirmation = position

        if len(last_highs) < 2 or len(last_lows) < 2:
            label = "insufficient confirmed structure"
            score = 0.0
        else:
            high_up = last_highs[-1] > last_highs[-2]
            low_up = last_lows[-1] > last_lows[-2]
            high_down = last_highs[-1] < last_highs[-2]
            low_down = last_lows[-1] < last_lows[-2]
            if high_up and low_up:
                label, score = "higher highs and higher lows", 1.0
            elif high_down and low_down:
                label, score = "lower highs and lower lows", -1.0
            elif high_up and low_down:
                label, score = "expanding range", 0.0
            else:
                label, score = "compression / balance", 0.0
        structure_labels.append(label)
        structure_scores.append(score)
        high_ages.append(
            np.nan
            if last_high_confirmation is None
            else float(position - last_high_confirmation)
        )
        low_ages.append(
            np.nan
            if last_low_confirmation is None
            else float(position - last_low_confirmation)
        )
    output["market_structure"] = structure_labels
    output["structure_score"] = structure_scores
    output["bars_since_swing_high"] = high_ages
    output["bars_since_swing_low"] = low_ages
    return output


def _add_liquidity_levels(features: pd.DataFrame, config: DeskConfig) -> pd.DataFrame:
    output = features.copy()
    prior_high = output["high"].shift(1).rolling(
        config.liquidity_lookback,
        min_periods=1,
    ).max()
    prior_low = output["low"].shift(1).rolling(
        config.liquidity_lookback,
        min_periods=1,
    ).min()
    confirmed_highs: list[tuple[int, float]] = []
    confirmed_lows: list[tuple[int, float]] = []
    supports: list[float] = []
    resistances: list[float] = []
    for position, (_, row) in enumerate(output.iterrows()):
        if pd.notna(row["confirmed_swing_high_event"]):
            confirmed_highs.append(
                (position, float(row["confirmed_swing_high_event"]))
            )
        if pd.notna(row["confirmed_swing_low_event"]):
            confirmed_lows.append((position, float(row["confirmed_swing_low_event"])))
        cutoff = position - config.liquidity_lookback
        confirmed_highs = [event for event in confirmed_highs if event[0] >= cutoff]
        confirmed_lows = [event for event in confirmed_lows if event[0] >= cutoff]
        close = float(row["close"])
        above = [level for _, level in confirmed_highs if level > close]
        below = [level for _, level in confirmed_lows if level < close]
        resistance = min(above) if above else float(prior_high.iloc[position])
        support = max(below) if below else float(prior_low.iloc[position])
        resistances.append(resistance)
        supports.append(support)
    output["support"] = supports
    output["resistance"] = resistances
    return output


def _prefix_vwap(
    cumulative_pv: np.ndarray,
    cumulative_volume: np.ndarray,
    anchor: int,
    end: int,
) -> float:
    pv = cumulative_pv[end + 1] - cumulative_pv[anchor]
    volume = cumulative_volume[end + 1] - cumulative_volume[anchor]
    return float(pv / volume) if volume > 0.0 else float("nan")


def _add_causal_avwap(features: pd.DataFrame, config: DeskConfig) -> pd.DataFrame:
    output = features.copy()
    typical = (output["high"] + output["low"] + output["close"]) / 3.0
    pv = (typical * output["volume"]).to_numpy(dtype=float)
    volume = output["volume"].to_numpy(dtype=float)
    cumulative_pv = np.concatenate([[0.0], np.cumsum(pv)])
    cumulative_volume = np.concatenate([[0.0], np.cumsum(volume)])

    swing_values: list[float] = []
    swing_anchor: int | None = None
    for position, pivot_position in enumerate(
        output["confirmed_swing_low_pivot_pos"].to_numpy()
    ):
        if np.isfinite(pivot_position):
            swing_anchor = int(pivot_position)
        swing_values.append(
            float("nan")
            if swing_anchor is None
            else _prefix_vwap(
                cumulative_pv,
                cumulative_volume,
                swing_anchor,
                position,
            )
        )
    output["swing_avwap"] = swing_values

    returns = output["return"]
    return_z = _rolling_lagged_zscore(returns, max(config.volume_window, 10))
    range_ratio = output["true_range"] / output["close"].shift(1)
    range_z = _rolling_lagged_zscore(range_ratio, max(config.volume_window, 10))
    stress_score = (
        (-return_z.clip(upper=0.0)).fillna(0.0)
        + output["volume_z_20"].clip(lower=0.0).fillna(0.0)
        + range_z.clip(lower=0.0).fillna(0.0)
    )
    eligible = (returns < 0.0) & (stress_score >= 2.0)
    stress_values: list[float] = []
    stress_anchor_positions: list[float] = []
    for position in range(len(output)):
        start = max(0, position - config.stress_lookback + 1)
        window_scores = stress_score.iloc[start : position + 1].where(
            eligible.iloc[start : position + 1]
        )
        if window_scores.notna().any():
            anchor_label = window_scores.idxmax()
            anchor = int(output.index.get_loc(anchor_label))
            stress_values.append(
                _prefix_vwap(cumulative_pv, cumulative_volume, anchor, position)
            )
            stress_anchor_positions.append(float(anchor))
        else:
            stress_values.append(float("nan"))
            stress_anchor_positions.append(float("nan"))
    output["stress_score"] = stress_score
    output["stress_avwap"] = stress_values
    output["stress_anchor_pos"] = stress_anchor_positions
    output["swing_avwap_distance_atr"] = (
        output["close"] - output["swing_avwap"]
    ) / output["atr"]
    return output


def compute_desk_features(data: pd.DataFrame, config: DeskConfig) -> pd.DataFrame:
    """Return causal daily desk features for one normalized ticker history."""

    output = prepare_ohlcv(data, config)
    output["return"] = output["close"].pct_change()
    output["log_return"] = np.log(output["close"] / output["close"].shift(1))
    previous_close = output["close"].shift(1)
    output["true_range"] = pd.concat(
        [
            output["high"] - output["low"],
            (output["high"] - previous_close).abs(),
            (output["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    output["atr"] = output["true_range"].ewm(
        alpha=1.0 / config.atr_window,
        adjust=False,
        min_periods=config.atr_window,
    ).mean()
    output["ema_fast"] = output["close"].ewm(
        span=config.ema_fast_span,
        adjust=False,
        min_periods=config.ema_fast_span,
    ).mean()
    output["ema_mid"] = output["close"].ewm(
        span=config.ema_mid_span,
        adjust=False,
        min_periods=config.ema_mid_span,
    ).mean()
    output["ema_slow"] = output["close"].ewm(
        span=config.ema_slow_span,
        adjust=False,
        min_periods=config.ema_slow_span,
    ).mean()
    output["realized_vol_20"] = output["return"].rolling(
        config.realized_vol_window,
        min_periods=config.realized_vol_window,
    ).std(ddof=1) * math.sqrt(252.0)
    downside_squared = output["return"].clip(upper=0.0).pow(2)
    output["downside_vol_60"] = np.sqrt(
        downside_squared.rolling(
            config.downside_vol_window,
            min_periods=config.downside_vol_window,
        ).mean()
        * 252.0
    )
    output["max_drawdown_126"] = (
        output["close"]
        / output["close"].rolling(
            config.drawdown_window,
            min_periods=config.drawdown_window,
        ).max()
        - 1.0
    )
    output["momentum_21d"] = (
        output["close"] / output["close"].shift(config.momentum_short_window) - 1.0
    )
    output["momentum_126_21d"] = (
        output["close"].shift(config.momentum_skip_window)
        / output["close"].shift(config.momentum_medium_window)
        - 1.0
    )
    output["near_52w_high"] = (
        output["close"]
        / output["close"].rolling(
            config.high_lookback,
            min_periods=config.high_lookback,
        ).max()
        - 1.0
    )
    output["dollar_volume"] = output["close"] * output["volume"]
    output["log_adv_20"] = np.log(
        output["dollar_volume"].rolling(
            config.volume_window,
            min_periods=config.volume_window,
        ).mean()
    )
    output["volume_z_20"] = _rolling_lagged_zscore(
        output["volume"],
        config.volume_window,
    )
    output["amihud_20"] = (
        output["return"].abs() / output["dollar_volume"].replace(0.0, np.nan)
    ).rolling(config.volume_window, min_periods=config.volume_window).mean()
    intraday_range = (output["high"] - output["low"]).replace(0.0, np.nan)
    output["close_location"] = (
        (output["close"] - output["low"])
        - (output["high"] - output["close"])
    ) / intraday_range
    output = _add_confirmed_structure(output, config)
    output = _add_liquidity_levels(output, config)
    output = _add_causal_avwap(output, config)
    return output


def _cluster_zone(
    levels: Sequence[float],
    close: float,
    atr: float,
    side: str,
    config: DeskConfig,
    fallback: float,
) -> tuple[float, float, int]:
    if side == "demand":
        selected = sorted(level for level in levels if level < close)
    elif side == "supply":
        selected = sorted(level for level in levels if level > close)
    else:
        raise ValueError("Zone side must be 'demand' or 'supply'.")
    if not selected:
        selected = [fallback]
    clusters: list[list[float]] = []
    tolerance = config.zone_cluster_atr * atr
    for level in selected:
        if not clusters or abs(level - float(np.mean(clusters[-1]))) > tolerance:
            clusters.append([level])
        else:
            clusters[-1].append(level)
    cluster = (
        max(clusters, key=lambda values: float(np.mean(values)))
        if side == "demand"
        else min(clusters, key=lambda values: float(np.mean(values)))
    )
    center = float(np.median(cluster))
    half_width = max(config.zone_half_width_atr * atr, center * 0.0025)
    return center - half_width, center + half_width, len(cluster)


def _scenario_status(
    close: float,
    trigger: float,
    atr: float,
    direction: str,
    config: DeskConfig,
) -> tuple[str, float]:
    distance = (trigger - close) / atr if direction == "long" else (close - trigger) / atr
    if distance <= 0.0:
        return "triggered", distance
    if distance <= config.max_trigger_distance_atr:
        return "watch", distance
    return "distant", distance


def _build_scenarios(
    close: float,
    atr: float,
    support: float,
    resistance: float,
    demand_zone: tuple[float, float],
    supply_zone: tuple[float, float],
    swing_avwap: float,
    stress_avwap: float,
    config: DeskConfig,
) -> tuple[TradeScenario, TradeScenario]:
    finite_avwaps = [value for value in (swing_avwap, stress_avwap) if np.isfinite(value)]
    bullish_reference = max(finite_avwaps) if finite_avwaps else close
    bearish_reference = min(finite_avwaps) if finite_avwaps else close

    long_trigger = max(resistance, supply_zone[1], bullish_reference) + 0.10 * atr
    raw_long_stop = min(resistance, supply_zone[0]) - 0.25 * atr
    long_stop = max(raw_long_stop, long_trigger - config.max_stop_distance_atr * atr)
    long_risk = long_trigger - long_stop
    long_status, long_distance = _scenario_status(
        close, long_trigger, atr, "long", config
    )
    long_risk_fraction = long_risk / long_trigger
    long_eligible = (
        long_risk > 0.0
        and long_risk_fraction <= config.max_risk_fraction
        and long_distance <= config.max_trigger_distance_atr
    )
    long_scenario = TradeScenario(
        direction="long",
        status=long_status,
        eligible=long_eligible,
        trigger=round(long_trigger, 2),
        stop=round(long_stop, 2),
        target_1=round(long_trigger + config.target_1_r_multiple * long_risk, 2),
        target_2=round(long_trigger + config.target_2_r_multiple * long_risk, 2),
        risk_per_share=round(long_risk, 2),
        risk_fraction=round(long_risk_fraction, 4),
        reward_risk_1=config.target_1_r_multiple,
        reward_risk_2=config.target_2_r_multiple,
        trigger_distance_atr=round(long_distance, 2),
        thesis="Acceptance above confirmed resistance, supply references, and AVWAP.",
        invalidation="Close or stop execution back through the breakout reference.",
    )

    short_trigger = min(support, demand_zone[0], bearish_reference) - 0.10 * atr
    raw_short_stop = max(support, demand_zone[1]) + 0.25 * atr
    short_stop = min(raw_short_stop, short_trigger + config.max_stop_distance_atr * atr)
    short_risk = short_stop - short_trigger
    short_status, short_distance = _scenario_status(
        close, short_trigger, atr, "bearish", config
    )
    short_risk_fraction = short_risk / short_trigger if short_trigger > 0.0 else math.inf
    short_eligible = (
        short_trigger > 0.0
        and short_risk > 0.0
        and short_risk_fraction <= config.max_risk_fraction
        and short_distance <= config.max_trigger_distance_atr
    )
    bearish_scenario = TradeScenario(
        direction="bearish",
        status=short_status,
        eligible=short_eligible,
        trigger=round(short_trigger, 2),
        stop=round(short_stop, 2),
        target_1=round(short_trigger - config.target_1_r_multiple * short_risk, 2),
        target_2=round(short_trigger - config.target_2_r_multiple * short_risk, 2),
        risk_per_share=round(short_risk, 2),
        risk_fraction=round(short_risk_fraction, 4),
        reward_risk_1=config.target_1_r_multiple,
        reward_risk_2=config.target_2_r_multiple,
        trigger_distance_atr=round(short_distance, 2),
        thesis="Acceptance below confirmed support, demand references, and AVWAP.",
        invalidation="Close or stop execution back through the breakdown reference.",
    )
    return long_scenario, bearish_scenario


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Latest {name} is not finite; increase the history period.")
    return result


def _build_snapshot(
    ticker: str,
    features: pd.DataFrame,
    config: DeskConfig,
) -> DeskSnapshot:
    last = features.iloc[-1]
    close = _finite(last["close"], "close")
    atr = _finite(last["atr"], "ATR")
    ema_fast = _finite(last["ema_fast"], "fast EMA")
    ema_mid = _finite(last["ema_mid"], "mid EMA")
    ema_slow = _finite(last["ema_slow"], "slow EMA")
    if close > ema_fast > ema_mid > ema_slow:
        trend, trend_score = "bullish", 1.0
    elif close < ema_fast < ema_mid < ema_slow:
        trend, trend_score = "bearish", -1.0
    else:
        trend, trend_score = "transitional", 0.0

    recent = features.tail(config.zone_lookback)
    demand_levels = recent["confirmed_swing_low_event"].dropna().astype(float).tolist()
    supply_levels = recent["confirmed_swing_high_event"].dropna().astype(float).tolist()
    support = _finite(last["support"], "support")
    resistance = _finite(last["resistance"], "resistance")
    demand_low, demand_high, demand_touches = _cluster_zone(
        demand_levels,
        close,
        atr,
        "demand",
        config,
        support,
    )
    supply_low, supply_high, supply_touches = _cluster_zone(
        supply_levels,
        close,
        atr,
        "supply",
        config,
        resistance,
    )
    swing_avwap = float(last["swing_avwap"])
    stress_avwap = float(last["stress_avwap"])
    available_avwaps = [
        value for value in (swing_avwap, stress_avwap) if math.isfinite(value)
    ]
    if available_avwaps and close > max(available_avwaps):
        avwap_state = "above all available AVWAP references"
    elif available_avwaps and close < min(available_avwaps):
        avwap_state = "below all available AVWAP references"
    elif available_avwaps:
        avwap_state = "between AVWAP references"
    else:
        avwap_state = "AVWAP reference unavailable"

    recent_volume = features.tail(config.volume_window)
    up_volume = recent_volume.loc[recent_volume["return"] > 0.0, "volume"].mean()
    down_volume = recent_volume.loc[recent_volume["return"] < 0.0, "volume"].mean()
    if pd.isna(up_volume) or pd.isna(down_volume):
        price_volume_state = "mixed"
    elif up_volume > down_volume * 1.10:
        price_volume_state = "up-day volume dominant"
    elif down_volume > up_volume * 1.10:
        price_volume_state = "down-day volume dominant"
    else:
        price_volume_state = "balanced"

    long_scenario, bearish_scenario = _build_scenarios(
        close,
        atr,
        support,
        resistance,
        (demand_low, demand_high),
        (supply_low, supply_high),
        swing_avwap,
        stress_avwap,
        config,
    )
    return DeskSnapshot(
        ticker=ticker.upper(),
        as_of=pd.Timestamp(features.index[-1]),
        last_close=round(close, 2),
        primary_trend=trend,
        trend_score=trend_score,
        market_structure=str(last["market_structure"]),
        structure_score=float(last["structure_score"]),
        momentum_21d=_finite(last["momentum_21d"], "21-day momentum"),
        momentum_126_21d=_finite(
            last["momentum_126_21d"], "medium momentum"
        ),
        near_52w_high=_finite(last["near_52w_high"], "52-week-high distance"),
        realized_vol_20=_finite(last["realized_vol_20"], "realized volatility"),
        downside_vol_60=_finite(last["downside_vol_60"], "downside volatility"),
        max_drawdown_126=_finite(last["max_drawdown_126"], "drawdown"),
        atr=round(atr, 4),
        volume_z_20=_finite(last["volume_z_20"], "volume z-score"),
        price_volume_state=price_volume_state,
        log_adv_20=_finite(last["log_adv_20"], "ADV"),
        amihud_20=_finite(last["amihud_20"], "Amihud illiquidity"),
        support=round(support, 2),
        resistance=round(resistance, 2),
        demand_zone_low=round(demand_low, 2),
        demand_zone_high=round(demand_high, 2),
        demand_zone_touches=demand_touches,
        supply_zone_low=round(supply_low, 2),
        supply_zone_high=round(supply_high, 2),
        supply_zone_touches=supply_touches,
        swing_avwap=round(swing_avwap, 2) if math.isfinite(swing_avwap) else math.nan,
        stress_avwap=round(stress_avwap, 2) if math.isfinite(stress_avwap) else math.nan,
        avwap_state=avwap_state,
        bars_since_swing_high=float(last["bars_since_swing_high"]),
        bars_since_swing_low=float(last["bars_since_swing_low"]),
        long_scenario=long_scenario,
        bearish_scenario=bearish_scenario,
    )


def analyze_ticker(
    ticker: str,
    config: DeskConfig | None = None,
    downloader: DataDownloader = download_ohlcv,
) -> DeskAnalysis:
    """Download once and produce a complete causal desk analysis."""

    desk_config = config or DeskConfig()
    raw = downloader(ticker, desk_config)
    history = compute_desk_features(raw, desk_config)
    return DeskAnalysis(
        snapshot=_build_snapshot(ticker, history, desk_config),
        history=history,
    )


def snapshot_frame(snapshot: DeskSnapshot) -> pd.DataFrame:
    """Return a compact one-row desk snapshot without nested scenarios."""

    values = asdict(snapshot)
    values.pop("long_scenario")
    values.pop("bearish_scenario")
    return pd.DataFrame([values])


def scenario_frame(snapshot: DeskSnapshot) -> pd.DataFrame:
    """Return the bullish and bearish desk scenarios as rows."""

    return pd.DataFrame(
        [asdict(snapshot.long_scenario), asdict(snapshot.bearish_scenario)]
    )


def factor_panel(snapshot: DeskSnapshot) -> pd.DataFrame:
    """Return observable factor state with explicit units and interpretation."""

    rows = [
        ("Trend", "Primary trend", snapshot.primary_trend, "EMA alignment"),
        ("Structure", "Confirmed structure", snapshot.market_structure, "Delayed pivot confirmation"),
        ("Momentum", "21-day return", snapshot.momentum_21d, "decimal return"),
        ("Momentum", "126-to-21-day return", snapshot.momentum_126_21d, "skip-recent momentum"),
        ("Price location", "Distance from 52-week high", snapshot.near_52w_high, "decimal distance"),
        ("Risk", "20-day realized volatility", snapshot.realized_vol_20, "annualized"),
        ("Risk", "60-day downside volatility", snapshot.downside_vol_60, "annualized"),
        ("Risk", "126-day drawdown", snapshot.max_drawdown_126, "decimal drawdown"),
        ("Liquidity", "Log 20-day ADV", snapshot.log_adv_20, "log dollars"),
        ("Liquidity", "20-day Amihud", snapshot.amihud_20, "absolute return / dollar volume"),
        ("Volume", "Volume surprise", snapshot.volume_z_20, "lagged 20-day z-score"),
        ("Reference", "AVWAP state", snapshot.avwap_state, "daily-bar proxy"),
    ]
    return pd.DataFrame(rows, columns=["sleeve", "metric", "value", "units_or_definition"])


def scan_universe(
    tickers: Sequence[str],
    config: DeskConfig | None = None,
    downloader: DataDownloader = download_ohlcv,
) -> pd.DataFrame:
    """Analyze tickers independently while preserving per-name errors."""

    desk_config = config or DeskConfig()
    rows: list[pd.DataFrame] = []
    for ticker in dict.fromkeys(value.strip().upper() for value in tickers if value.strip()):
        try:
            analysis = analyze_ticker(ticker, desk_config, downloader)
            rows.append(snapshot_frame(analysis.snapshot))
        except Exception as exc:  # scanner must preserve the rest of the universe
            rows.append(pd.DataFrame([{"ticker": ticker, "error": str(exc)}]))
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    ranked = numeric.rank(method="average", pct=True, ascending=higher_is_better)
    return ranked


def rank_universe(scan: pd.DataFrame) -> pd.DataFrame:
    """Add transparent cross-sectional factor sleeves and a diagnostic composite.

    The equal-weight composite is a research-priority rank, not a probability or
    a validated expected-return forecast.
    """

    if scan.empty:
        return scan.copy()
    required = {
        "momentum_21d",
        "momentum_126_21d",
        "near_52w_high",
        "trend_score",
        "structure_score",
        "realized_vol_20",
        "max_drawdown_126",
        "log_adv_20",
        "amihud_20",
    }
    missing = sorted(required.difference(scan.columns))
    if missing:
        raise ValueError(f"Scan is missing ranking columns: {missing}")
    ranked = scan.copy()
    valid = ranked.get("error", pd.Series(index=ranked.index, dtype=object)).isna()
    subset = ranked.loc[valid]
    ranked.loc[valid, "momentum_rank"] = pd.concat(
        [
            _percentile(subset["momentum_21d"]),
            _percentile(subset["momentum_126_21d"]),
            _percentile(subset["near_52w_high"]),
        ],
        axis=1,
    ).mean(axis=1)
    ranked.loc[valid, "structure_rank"] = pd.concat(
        [
            _percentile(subset["trend_score"]),
            _percentile(subset["structure_score"]),
        ],
        axis=1,
    ).mean(axis=1)
    ranked.loc[valid, "risk_rank"] = pd.concat(
        [
            _percentile(subset["realized_vol_20"], higher_is_better=False),
            _percentile(subset["max_drawdown_126"]),
        ],
        axis=1,
    ).mean(axis=1)
    ranked.loc[valid, "liquidity_rank"] = pd.concat(
        [
            _percentile(subset["log_adv_20"]),
            _percentile(subset["amihud_20"], higher_is_better=False),
        ],
        axis=1,
    ).mean(axis=1)
    sleeve_columns = [
        "momentum_rank",
        "structure_rank",
        "risk_rank",
        "liquidity_rank",
    ]
    ranked.loc[valid, "research_priority"] = ranked.loc[
        valid, sleeve_columns
    ].mean(axis=1)
    return ranked.sort_values(
        ["research_priority", "ticker"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def _add_level(
    fig: go.Figure,
    level: float,
    name: str,
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
    dash: str,
) -> None:
    if not math.isfinite(level):
        return
    fig.add_trace(
        go.Scatter(
            x=[first_date, last_date],
            y=[level, level],
            mode="lines",
            name=name,
            line={"width": 1, "dash": dash},
            hovertemplate=f"{name}<br>$%{{y:.2f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )


def create_desk_chart(
    analysis: DeskAnalysis,
    config: DeskConfig | None = None,
    show: bool = True,
) -> go.Figure:
    """Create an inline interactive price, volume, and risk dashboard."""

    desk_config = config or DeskConfig()
    snapshot = analysis.snapshot
    plot = analysis.history.tail(desk_config.chart_bars).copy()
    first_date = pd.Timestamp(plot.index[0])
    last_date = pd.Timestamp(plot.index[-1])
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.64, 0.18, 0.18],
        specs=[[{}], [{}], [{"secondary_y": True}]],
        subplot_titles=("Price, structure, and scenarios", "Volume", "Risk state"),
    )
    fig.add_trace(
        go.Candlestick(
            x=plot.index,
            open=plot["open"],
            high=plot["high"],
            low=plot["low"],
            close=plot["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    for column, label in (
        ("ema_fast", f"EMA {desk_config.ema_fast_span}"),
        ("ema_mid", f"EMA {desk_config.ema_mid_span}"),
        ("ema_slow", f"EMA {desk_config.ema_slow_span}"),
        ("swing_avwap", "Confirmed swing AVWAP"),
        ("stress_avwap", "Causal stress AVWAP"),
    ):
        fig.add_trace(
            go.Scatter(
                x=plot.index,
                y=plot[column],
                mode="lines",
                name=label,
                line={"width": 1.4},
            ),
            row=1,
            col=1,
        )
    confirmed_highs = plot[plot["confirmed_swing_high_event"].notna()]
    confirmed_lows = plot[plot["confirmed_swing_low_event"].notna()]
    fig.add_trace(
        go.Scatter(
            x=confirmed_highs.index,
            y=confirmed_highs["confirmed_swing_high_event"],
            mode="markers",
            marker={"symbol": "triangle-down", "size": 8},
            name="Confirmed swing highs",
            hovertemplate="Confirmed high<br>%{x}<br>$%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=confirmed_lows.index,
            y=confirmed_lows["confirmed_swing_low_event"],
            mode="markers",
            marker={"symbol": "triangle-up", "size": 8},
            name="Confirmed swing lows",
            hovertemplate="Confirmed low<br>%{x}<br>$%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hrect(
        y0=snapshot.demand_zone_low,
        y1=snapshot.demand_zone_high,
        fillcolor="rgba(46, 160, 67, 0.12)",
        line_width=0,
        annotation_text=f"Demand references ({snapshot.demand_zone_touches})",
        annotation_position="bottom left",
        row=1,
        col=1,
    )
    fig.add_hrect(
        y0=snapshot.supply_zone_low,
        y1=snapshot.supply_zone_high,
        fillcolor="rgba(248, 81, 73, 0.12)",
        line_width=0,
        annotation_text=f"Supply references ({snapshot.supply_zone_touches})",
        annotation_position="top left",
        row=1,
        col=1,
    )
    levels = (
        (snapshot.support, "Confirmed support", "dot"),
        (snapshot.resistance, "Confirmed resistance", "dot"),
        (snapshot.long_scenario.trigger, "Long trigger", "dash"),
        (snapshot.long_scenario.stop, "Long stop", "dot"),
        (snapshot.long_scenario.target_1, "Long target 1", "dashdot"),
        (snapshot.bearish_scenario.trigger, "Bearish trigger", "dash"),
        (snapshot.bearish_scenario.stop, "Bearish stop", "dot"),
        (snapshot.bearish_scenario.target_1, "Bearish target 1", "dashdot"),
    )
    for level, label, dash in levels:
        _add_level(fig, level, label, first_date, last_date, dash)

    fig.add_trace(
        go.Bar(x=plot.index, y=plot["volume"], name="Daily volume"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot.index,
            y=plot["volume"].rolling(desk_config.volume_window).mean(),
            mode="lines",
            name=f"{desk_config.volume_window}-day average volume",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot.index,
            y=plot["realized_vol_20"] * 100.0,
            mode="lines",
            name="Realized volatility",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=plot.index,
            y=plot["max_drawdown_126"] * 100.0,
            mode="lines",
            name="Rolling drawdown",
            fill="tozeroy",
            opacity=0.35,
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    summary = (
        f"Trend: {snapshot.primary_trend} | Structure: {snapshot.market_structure} | "
        f"AVWAP: {snapshot.avwap_state} | Volume: {snapshot.price_volume_state}"
    )
    fig.update_layout(
        title={
            "text": f"{snapshot.ticker} institutional research desk<br><sup>{summary}</sup>",
            "x": 0.02,
            "xanchor": "left",
        },
        height=980,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 70, "r": 70, "t": 125, "b": 45},
    )
    fig.update_yaxes(title_text="Price", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Volatility %", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1, secondary_y=True)
    for row in (1, 2, 3):
        fig.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}], row=row, col=1)
    if show:
        fig.show()
    return fig
