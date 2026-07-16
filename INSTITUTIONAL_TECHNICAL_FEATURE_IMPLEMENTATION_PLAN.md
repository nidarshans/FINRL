# Institutional Technical Feature Implementation Plan

## Status and authority

This is a code-derived implementation proposal. It is intentionally separate
from the existing Markdown architecture and planning documents, which do not
fully describe the executable system. It does not update or override
`ARCHITECTURE.MD`.

The plan is based on the current source code and test suite as inspected on
2026-07-15. The baseline test suite passes: 174 tests.

## Implemented research-desk baseline

The notebook-facing research layer is now implemented, but these additions are
not yet routed into the production DPO feature contract:

- `src/finrl/research/institutional_desk.py` provides validated daily OHLCV
  ingestion, gap-aware ATR, multi-horizon momentum, volatility, drawdown,
  volume/liquidity proxies, delayed pivot confirmation, causal daily-bar AVWAP,
  clustered supply/demand references, transparent scenario construction, and
  cross-sectional research ranking.
- `notebooks/institutional_ta_scanner_notebook.ipynb` consumes that reusable
  module as an inline desk with a single-name monitor, data-health controls,
  factor and scenario books, a three-panel Plotly dashboard, a universe
  blotter, and an unscaled DPO state hand-off.
- `tests/test_institutional_desk.py` verifies true range, pivot confirmation
  delay, future-mutation invariance, AVWAP arithmetic, scenario risk bounds,
  cross-sectional ranks, and chart composition.

This establishes one tested calculation path shared by research output and the
notebook. The production DPO integration should still follow the phased order
and preprocessing gates below rather than importing the notebook's scenario
levels or composite research-priority score as policy inputs.

## Objective

Add a compact, causal, economically interpretable set of institutional-style
technical features to the direct portfolio optimization (DPO) policy, then
measure their incremental out-of-sample value through strict walk-forward
ablations.

The goal is not to reproduce discretionary chart annotations inside the model.
The goal is to translate useful price, volume, volatility, liquidity, and
market-structure information into dimensionless decision-date features that the
shared per-stock allocation function can compare across securities.

## Current executable system

### Active data flow

The code currently implements this path:

```text
adjusted daily OHLCV
  -> trailing per-ticker asset features
  -> daily or weekly decision-date filtering
  -> chronological rolling preprocessing
  -> [time, asset, feature] panel
  -> explicit feature-name-to-index routing
  -> shared per-stock allocation head
  -> softmax or sparsemax across risky assets
  -> append cash fallback column
  -> differentiable chronological portfolio accounting
  -> negative net-return Sharpe loss
  -> frozen walk-forward test evaluation
```

Features are observed at the completed decision-date close. Returns are aligned
to the following execution-open-to-next-execution-open holding interval.

### Current routed features

`src/finrl/features/columns.py` currently routes 14 features:

1. `mr_ewma50_vol_gap`
2. `ewma50_slope`
3. `acc_macd_signal`
4. `acc_klinger_signal`
5. `macd_signal_strength`
6. `klinger_signal_strength`
7. `acc_momentum_quality`
8. `cmf`
9. `cmf_slope`
10. `cmf_cross_signal`
11. `cmf_days_since_cross`
12. `frog_in_the_pan`
13. `bollinger_bandwidth`
14. `fip_over_bollinger_bandwidth`

These features cover mean reversion, moving-average trend, MACD, signed-volume
trend, Chaikin money flow, momentum consistency, return-sign persistence, and
Bollinger compression. They do not provide a complete institutional feature
set.

### Important code observations

1. The policy is long-only and cross-sectionally normalized. A bearish feature
   can reduce a stock's relative weight, but it cannot create a short position.
2. Cash receives zero weight whenever at least one stock is tradable. Technical
   risk signals cannot move the portfolio defensively into cash under the
   current allocation head.
3. Each stock is scored independently by the same learned function. Cross-asset
   interaction occurs only through the final simplex normalization. Inputs
   should therefore be scale-free or explicitly cross-sectionally ranked.
4. The feature pipeline computes daily trailing values before selecting
   decision dates. This is the correct place for multi-day technical features.
5. The preprocessing path rolling-standardizes ordinary columns per ticker and
   preserves columns ending in `_percentile_rank`.
6. `finrl.features.relative` contains cross-sectional rank helpers, but the
   active feature pipeline does not call them.
7. The relative-feature helper expects return, dollar-volume, and Amihud
   columns that the active `compute_asset_features` output does not expose.
8. The feature function returns only routed columns. There is no current
   separation between computed diagnostic features and policy-routed features.
9. The environment can model a liquidity surcharge, but the DPO training loss
   currently models only a flat transaction-cost rate. Variable cost inputs
   must not create a training/evaluation accounting mismatch.
10. The production code has a 14-column routing contract, while older Markdown
    describes other feature counts and policy structures. New work must lock a
    code-based baseline before changing the routing contract.

## Preprocessing audit

### Verdict

The active production path is chronologically causal, but its preprocessing is
not yet statistically appropriate for every current or proposed feature.

What is correct:

- Asset preprocessing is grouped by ticker and sorted chronologically.
- Train rows precede test rows, and future test values cannot alter prior rows.
- Earlier test observations may update later test normalization state, which is
  valid for a sequential online transform.
- Cross-sectional percentile-rank columns are intentionally passed through.
- Non-finite values are prevented from reaching the JAX panel.
- The current tests verify chronological behavior, shape preservation, clipping,
  rank passthrough, and finite output.

What needs correction:

1. The same rolling z-score is applied to almost every feature, including
   bounded indicators, sparse event flags, age variables, already-normalized
   ratios, price-scale oscillators, volume-scale oscillators, and heavy-tailed
   products.
2. `rolling_window=252` counts decision rows, not trading sessions. It is about
   one year for daily rebalancing but about five years for weekly rebalancing.
   Configuration must specify decision periods or derive the intended number
   from rebalance frequency.
3. Rolling mean and standard deviation include the current feature value. This
   is causal, but it dampens current extremes. Features intended to measure a
   surprise should use statistics through `t-1`; slow state variables may use
   current-inclusive normalization if explicitly documented.
4. Early warm-up nulls are replaced by zero and then included in rolling
   statistics. Zero is not a valid neutral substitute for insufficient history.
5. Later missing values are forward-filled without a maximum staleness limit.
   A stale signal can therefore remain tradable indefinitely.
6. `build_asset_feature_panel` performs an additional blanket null-to-zero fill
   and sets every asset tradable. This can hide feature-readiness problems.
7. Clipping is global across all transformed columns. Different feature
   distributions require different bounds and transforms.
8. Default clipping is disabled. Heavy-tailed features such as momentum quality
   and FIP divided by narrow Bollinger bandwidth can remain finite but extreme,
   contaminating rolling statistics for many periods.
9. `fit_preprocessors` stores column metadata rather than fitted scale
   parameters. The safe production function combines train and test history,
   but calling `transform_features(test_only, fitted)` resets rolling history at
   the test boundary. The API name and guardrails do not make this risk obvious.
10. `PreprocessingConfig` does not validate window size, clip ordering, finite
    fill values, or other numerical constraints.
11. Per-ticker time-series standardization does not replace same-date
    cross-sectional ranking. The shared allocation head needs explicitly
    comparable cross-asset features for relative selection.

### Required preprocessing design

Replace suffix-based, one-transform-for-all behavior with explicit metadata per
feature:

```python
TransformKind = Literal[
    "passthrough",
    "clipped_passthrough",
    "rolling_zscore",
    "lagged_rolling_zscore",
    "log1p_rolling_zscore",
    "signed_log_rolling_zscore",
    "percentile_rank",
]

@dataclass(frozen=True, slots=True)
class FeatureTransformSpec:
    name: str
    transform: TransformKind
    rolling_periods: int | None
    min_periods: int
    clip_lower: float | None = None
    clip_upper: float | None = None
    maximum_forward_fill_periods: int = 0
```

The experiment artifact must serialize the transform specification in the same
order as the routed feature names.

### Transform policy for the current features

| Feature | Recommended treatment | Reason |
|---|---|---|
| `mr_ewma50_vol_gap` | clipped passthrough or percentile rank | Already dimensionless and volatility-normalized; avoid blind double standardization |
| `ewma50_slope` | clipped passthrough | Already normalized by trailing variation |
| `acc_macd_signal` | first divide by close or ATR, then clip/standardize | Raw MACD signal has dollar-price scale |
| `acc_klinger_signal` | first normalize by ADV or volume scale, then clip/standardize | Raw value depends on the ticker's trading volume scale |
| `macd_signal_strength` | signed-log or percentile rank | Product feature can be heavy-tailed |
| `klinger_signal_strength` | signed-log or percentile rank | Product feature can be heavy-tailed and volume-scale dependent |
| `acc_momentum_quality` | winsorized rank or clipped rolling z-score | Return divided by small variance can explode |
| `cmf` | clipped passthrough | Naturally bounded and zero has economic meaning |
| `cmf_slope` | clipped passthrough | Already normalized by trailing variation |
| `cmf_cross_signal` | passthrough | Sparse categorical event in `{-1, 0, +1}` |
| `cmf_days_since_cross` | capped `log1p`, then passthrough | Nonnegative age variable; monotonic meaning should be retained |
| `frog_in_the_pan` | scale by its theoretical window bound or clip/pass | Already normalized by square root of window length |
| `bollinger_bandwidth` | `log1p` then rolling z-score or rank | Positive and right-skewed |
| `fip_over_bollinger_bandwidth` | hard finite cap plus signed-log or rank | Ratio becomes unstable when bandwidth is close to zero |

### Transform policy for proposed institutional features

- Returns, log price ratios, and ATR-normalized distances: clipped raw value,
  lagged time-series z-score, or same-date percentile rank depending on the
  hypothesis.
- Realized volatility, downside volatility, ADV, Amihud, and spread estimates:
  log transform before time-series scaling or cross-sectional ranking.
- `confirmed_structure_score`: passthrough.
- Bars-since-event features: capped `log1p` passthrough.
- Percentile ranks: passthrough with explicit validation that values lie in
  `[0, 1]`.
- Binary readiness and tradability state: masks, never normalized alpha inputs.

### Preprocessing tests to add before expanding the feature set

1. Weekly and daily configurations map a requested calendar horizon to the
   intended number of decision periods.
2. Lagged normalization at `t` uses statistics ending at `t-1`.
3. Warm-up rows remain not-ready rather than becoming tradable zero signals.
4. Forward fill respects a configured staleness limit.
5. Sparse categorical signals retain their exact values.
6. Bounded features remain inside their contractual bounds.
7. Heavy-tailed finite inputs are clipped or transformed before rolling moments.
8. Test-only transformation cannot silently reset required train history.
9. Rank columns are finite and contained in `[0, 1]`.
10. Changing future rows cannot change prior transformed values for every
    transform kind.
11. Each routed feature has exactly one explicit transform specification.
12. Invalid windows, clip bounds, fill values, and transform combinations fail
    during configuration validation.

## What institutional technical features mean

There is no universal feature list used by all institutions. Systematic asset
managers, hedge funds, execution desks, and market makers solve different
problems. Their reusable technical feature families commonly include:

- Multi-horizon trend and momentum.
- Relative strength versus the cross-section, market, industry, or factor
  benchmark.
- Breakout and price-location features, including proximity to trailing highs.
- Realized volatility, downside risk, drawdown, gap risk, beta, and correlation.
- Volume surprise, signed-volume proxies, participation, and price-volume
  confirmation.
- Liquidity and trading-cost proxies, including dollar volume, price impact,
  and estimated spread.
- Market breadth, dispersion, and volatility-regime state.
- Intraday order flow, imbalance, depth, and realized execution costs when
  quote and trade data are available.

The repository currently has daily OHLCV and public proxy data. It can build
the first seven categories at low frequency, with limitations. It cannot
credibly produce order-book imbalance, queue position, dealer inventory,
buyer-initiated trade flow, or true intraday VWAP from daily bars.

## Evidence-based feature catalogue

### 1. Multi-horizon momentum and trend

Institutional implementations usually separate horizons instead of relying on
one oscillator. Candidate features include:

```text
mom_21d       = close_t / close_(t-21) - 1
mom_63d       = close_t / close_(t-63) - 1
mom_126_21d   = close_(t-21) / close_(t-126) - 1
mom_252_21d   = close_(t-21) / close_(t-252) - 1
trend_20_100  = log(EMA_20 / EMA_100)
trend_50_200  = log(EMA_50 / EMA_200)
```

The skip-month variants separate medium-term momentum from very recent
reversal. Momentum should also be ranked cross-sectionally on each date.

Research basis:

- Jegadeesh and Titman document intermediate-horizon cross-sectional momentum.
- Moskowitz, Ooi, and Pedersen document time-series trend across liquid
  instruments.
- Brock, Lakonishok, and LeBaron evaluate moving-average and trading-range
  rules, demonstrating why they require statistical testing rather than visual
  assertion.

### 2. Breakout and price-location features

Candidate features:

```text
near_52w_high       = close / rolling_max(close, 252) - 1
near_52w_low        = close / rolling_min(close, 252) - 1
donchian_position   = (close - rolling_low_N) / (rolling_high_N - rolling_low_N)
breakout_20d_atr    = (close - prior_20d_high) / ATR_20
overnight_gap_atr   = (open - previous_close) / ATR_20
close_location      = ((close - low) - (high - close)) / (high - low)
```

The prior rolling high must exclude the current bar when measuring a breakout.
The 52-week-high literature supports treating proximity to a long-run price
anchor as distinct from raw past return.

### 3. Volatility and downside risk

Candidate features:

```text
realized_vol_20     = std(daily_return, 20) * sqrt(252)
realized_vol_60     = std(daily_return, 60) * sqrt(252)
downside_vol_60     = sqrt(mean(min(return, 0)^2, 60) * 252)
natr_20             = ATR_20 / close
parkinson_vol_20    = range-based volatility from log(high / low)
max_drawdown_126    = close / rolling_max(close, 126) - 1
gap_vol_20          = std(log(open / previous_close), 20)
vol_of_vol_60       = std(realized_vol_20, 60)
```

ATR must use true range:

```text
TR_t = max(
    high_t - low_t,
    abs(high_t - close_(t-1)),
    abs(low_t - close_(t-1)),
)
```

Volatility features can be predictors, allocation inputs, or risk controls.
Those uses must be evaluated separately. Research on volatility-managed
portfolios motivates the family but does not guarantee out-of-sample benefit;
the feature must pass this repository's own walk-forward tests.

### 4. Liquidity and execution-cost proxies

Candidate daily-data features:

```text
dollar_volume       = close * volume
log_adv_20          = log(mean(dollar_volume, 20))
volume_z_20         = (volume - mean(volume, 20)) / std(volume, 20)
amihud_20           = mean(abs(return) / dollar_volume, 20)
turnover_shock_20   = dollar_volume / mean(dollar_volume, 20) - 1
zero_return_ratio   = mean(return == 0, 60)
corwin_schultz_spread = high-low-based low-frequency spread estimate
```

Amihud's measure is feasible with current daily data and estimates price impact
per unit of dollar volume. Corwin-Schultz provides a spread proxy from daily
highs and lows. These features are useful both for alpha conditioning and for
capacity/cost modeling, but those roles must remain separate.

### 5. Price-volume interaction

The current system already contains CMF and Klinger-style features. Incremental
candidates should target information not already represented:

```text
signed_dollar_volume_20 = sum(sign(return) * dollar_volume, 20)
volume_confirmed_mom    = mom_63d * volume_z_20
breakout_volume         = breakout_20d_atr * max(volume_z_20, 0)
down_up_volume_ratio    = log(mean(volume | down day) / mean(volume | up day))
```

The label `accumulation` should not be used as ground truth. Daily OHLCV does
not identify the initiating side of every trade. These are price-volume
proxies, not direct measurements of institutional accumulation.

### 6. Market-relative and industry-relative features

Candidate features:

```text
beta_252             = cov(stock_return, market_return) / var(market_return)
residual_mom_126_21  = momentum after removing rolling market exposure
relative_strength_63 = stock_return_63 - market_return_63
idio_vol_60          = std(rolling market-model residual, 60)
```

If reliable industry metadata is later added:

```text
industry_relative_momentum
industry_breadth
sector-neutral percentile ranks
```

Market-relative features can be built from the existing SPY series. Industry
features require a point-in-time classification source and should not be
invented from ticker names.

### 7. Causal market structure and anchored reference prices

The institutional scanner notebook can contribute features only after its
look-ahead and scaling problems are corrected.

Recommended translations:

```text
confirmed_structure_score    in {-1, 0, +1}
support_distance_atr         = (close - confirmed_support) / ATR
resistance_distance_atr      = (confirmed_resistance - close) / ATR
swing_avwap_distance_atr     = (close - causal_swing_AVWAP) / ATR
capitulation_avwap_distance  = (close - causal_capitulation_AVWAP) / ATR
bars_since_swing_high
bars_since_swing_low
```

A pivot at date `p` requiring `right` future bars becomes available only at
`p + right`. The pivot price may be stored and forward-filled beginning on the
confirmation date. It must never be published on the pivot date in the
historical feature table.

Anchored VWAP calculated from daily HLC3 and volume is a daily-bar proxy, not
trade-level VWAP. The name and documentation should say so.

Do not route these notebook outputs into DPO:

- Raw support or resistance prices.
- Symmetric supply/demand bands derived from one pivot.
- Long or short trigger prices.
- Stops and targets.
- Reward/risk calculated from those heuristic levels.
- Hand-written `medium` or `high` confidence.
- The hand-weighted bullish score.

### 8. Regime, breadth, and dispersion

Potential market-state features include:

```text
market_realized_vol_20
cross_sectional_return_dispersion
fraction_above_ema_200
fraction_at_20d_high
average_pairwise_correlation
advance_decline_breadth
```

These are date-level features shared by all stocks. The current DPO path routes
only per-asset inputs, so using shared regime features requires an explicit
policy-input design decision. HMM states, if added, must use filtering only.

## Recommended feature sets

Do not add every candidate to one policy. Correlated indicator proliferation
raises estimation error and makes attribution impossible.

### Baseline set

Freeze the current 14 routed features exactly as `baseline_current_14` for
reproducibility.

### Institutional core v1

Test these additions first because they are causal, dimensionless or readily
normalized, feasible from current data, and relatively complementary to the
existing features:

1. `mom_21d`
2. `mom_126_21d`
3. `near_52w_high`
4. `realized_vol_20`
5. `downside_vol_60`
6. `max_drawdown_126`
7. `log_adv_20`
8. `amihud_20`
9. `volume_z_20`

### Structure extension v1

Evaluate only after the institutional core is stable:

1. `confirmed_structure_score`
2. `support_distance_atr`
3. `resistance_distance_atr`
4. `swing_avwap_distance_atr`
5. `bars_since_swing_low`

### Market-relative extension v1

Evaluate separately:

1. `relative_strength_63`
2. `beta_252`
3. `residual_mom_126_21`
4. `idio_vol_60`

### Deferred features

Defer the following until the simpler groups demonstrate value:

- Capitulation-anchor AVWAP because the anchor can be unstable and repaint if
  implemented incorrectly.
- Chart-pattern classifiers because they add parameter and multiple-testing
  burden.
- Industry-relative signals until point-in-time industry metadata exists.
- Intraday VWAP, order flow, spread, and imbalance until timestamped trades and
  quotes exist.
- Learned cash timing, short positions, or leverage because they change the
  allocation contract.

## Proposed implementation design

### 1. Introduce named feature-set configuration

Replace the single global all-or-nothing feature tuple with explicit named
feature sets while preserving exact allowlisting.

Proposed types:

```python
@dataclass(frozen=True, slots=True)
class FeatureSetConfig:
    name: str
    routed_columns: tuple[str, ...]
```

Required named sets:

- `baseline_current_14`
- `baseline_plus_momentum`
- `baseline_plus_risk`
- `baseline_plus_liquidity`
- `baseline_plus_structure`
- `institutional_core_v1`
- `institutional_core_plus_structure_v1`

The exact ordered feature names must be serialized with every experiment.
Unknown or missing names must raise `ValueError`.

### 2. Separate calculation from routing

Refactor the feature layer into:

```text
compute candidate feature table
  -> validate causal daily feature schema
  -> add selected per-date relative ranks
  -> select named routed columns
  -> decision-date filtering
```

This allows diagnostics and ablations without forcing every computed column
into the learned policy.

Suggested modules:

- `src/finrl/features/momentum.py`
- `src/finrl/features/risk.py`
- `src/finrl/features/liquidity.py`
- `src/finrl/features/structure.py`
- `src/finrl/features/market_relative.py`

Keep each function pure, typed, deterministic, per-ticker, and free of hidden
state.

### 3. Extend feature configuration

Add validated parameters for:

- Momentum horizons and skip window.
- ATR and realized-volatility windows.
- Downside-risk and drawdown windows.
- ADV, volume-z, and Amihud windows.
- Swing left/right confirmation widths.
- Support/resistance lookback.
- AVWAP anchor policy.
- Minimum history required for each feature group.

All windows must be positive integers. Feature names should encode their default
horizon only when the horizon is contractually fixed; otherwise artifact
metadata must record the configured horizon.

### 4. Implement causal daily features

For ordinary trailing features, use only current and prior rows grouped by
ticker. For centered pivot candidates:

1. A candidate at `p` may inspect bars through `p + right`.
2. Shift the candidate flag and pivot price forward to the confirmation row.
3. Publish all derived structure state only from that confirmation row onward.
4. Add a future-mutation test proving that rows through date `t` cannot change
   when data after `t` changes.

Compute on the complete validated daily table, then sample decision dates. Do
not compute a weekly pivot algorithm on already-sampled weekly rows unless that
is a separately named feature definition.

### 5. Activate cross-sectional ranking deliberately

For features whose economic meaning is relative, add percentile ranks within
each date after raw trailing calculations and before decision-date filtering.

Initial rank candidates:

- `mom_126_21d_percentile_rank`
- `near_52w_high_percentile_rank`
- `realized_vol_20_percentile_rank`
- `log_adv_20_percentile_rank`
- `amihud_20_percentile_rank`

Ranks must be computed across the point-in-time eligible universe for that date.
Raw and ranked variants should not both be routed by default; test them as
separate ablations.

### 6. Handle warm-up and tradability explicitly

Long-horizon features require up to 252 completed sessions. A missing warm-up
value is not automatically equivalent to a neutral signal.

Add:

- Per-feature minimum-history validation.
- A row-level `feature_ready` diagnostic.
- A decision-date tradable mask based on valid feature history and valid
  execution/next-execution prices.
- Coverage reports by date and ticker.

Do not silently convert an insufficient-history asset into a fully tradable
zero-feature asset.

### 7. Preserve DPO accounting initially

The allocation policy and loss need no structural change for the first feature
experiments. The selected feature dimension is already inferred at runtime.

Keep unchanged initially:

- Shared per-stock scoring parameters.
- Long-only simplex allocation.
- Full chronological training path.
- Flat transaction-cost accounting.
- Frozen outer-test policy.

Before activating variable liquidity costs, make the same differentiable cost
function available to both DPO training and environment evaluation.

### 8. Keep alpha, risk, and execution roles distinct

Some inputs can serve multiple purposes, but experiments must not conflate
them:

- Momentum and relative strength are alpha candidates.
- Volatility, beta, and drawdown can be alpha conditioners or risk controls.
- ADV, Amihud, and spread can be alpha conditioners, eligibility filters, or
  transaction-cost inputs.
- Tradability and missing-price state belong in masks, not learned alpha.

Each role needs a separate experiment configuration and attribution report.

## Delivery phases

### Phase 0: Freeze the executable baseline

Work:

- Record the current 14 feature names and order.
- Add a named `baseline_current_14` routing configuration.
- Serialize feature names, feature parameters, preprocessing parameters,
  universe, date range, data fingerprint, seed, costs, and git revision.
- Save baseline per-split and aggregate metrics.
- Confirm the full suite remains green.

Acceptance criteria:

- Two runs with identical inputs produce identical feature panels and results.
- Baseline artifacts fully identify feature values and routing order.

### Phase 1: Institutional momentum and price location

Implement:

- `mom_21d`
- `mom_126_21d`
- `near_52w_high`
- Optional date-wise percentile ranks

Tests:

- Hand-calculated return horizons.
- Skip-window boundary correctness.
- Current bar excluded from the prior-high breakout denominator where required.
- Future data mutation does not alter prior values.
- Per-ticker independence.

Acceptance criteria:

- Baseline, raw-momentum, and ranked-momentum ablations run end to end.

### Phase 2: Volatility and downside risk

Implement:

- True range and `natr_20`
- `realized_vol_20`
- `downside_vol_60`
- `max_drawdown_126`

Tests:

- Gap-aware true range fixtures.
- Realized and downside volatility against hand calculations.
- Drawdown after new highs and partial recovery.
- Finite behavior for constant-price windows.

Acceptance criteria:

- Risk features can be ablated individually and as a group.
- No raw dollar price level reaches the policy.

### Phase 3: Liquidity and capacity proxies

Implement:

- `log_adv_20`
- `volume_z_20`
- `amihud_20`
- Cross-sectional liquidity ranks
- Coverage and minimum-liquidity diagnostics

Defer Corwin-Schultz until the simpler proxies are validated.

Tests:

- Dollar-volume and Amihud hand calculations.
- Zero-volume and non-finite input handling.
- Rank calculation within each date only.
- No cross-date or cross-ticker leakage.

Acceptance criteria:

- Liquidity can be evaluated separately as an alpha input and eligibility
  filter.
- Training and evaluation cost assumptions remain identical.

### Phase 4: Causal structure and AVWAP

Implement:

- Confirmed swing high and low events.
- `confirmed_structure_score`.
- ATR-normalized support and resistance distances.
- Causal swing-low daily-bar AVWAP distance.
- Bars since confirmed pivot.

Tests:

- Exact confirmation delay.
- Pivot chronological ordering.
- No label on the unconfirmed pivot date.
- AVWAP against a manual price-volume fixture.
- Anchor reset behavior.
- Future-mutation invariance.

Acceptance criteria:

- Structure features can be reproduced without the notebook.
- The visualization consumes production feature outputs rather than maintaining
  a second implementation.

### Phase 5: Market-relative features

Implement:

- SPY return alignment at daily feature timestamps.
- `relative_strength_63`.
- Rolling `beta_252`.
- `residual_mom_126_21`.
- `idio_vol_60`.

Tests:

- Beta and residuals against deterministic fixtures.
- Missing benchmark dates cannot be silently filled with zero.
- Benchmark values after date `t` cannot alter features through `t`.

Acceptance criteria:

- Raw and market-residual momentum are reported in separate ablations.

### Phase 6: Walk-forward feature selection and robustness

Run a locked experiment matrix:

| Experiment | Current 14 | Momentum | Risk | Liquidity | Structure | Market-relative |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | yes | no | no | no | no | no |
| Momentum | yes | yes | no | no | no | no |
| Risk | yes | no | yes | no | no | no |
| Liquidity | yes | no | no | yes | no | no |
| Structure | yes | no | no | no | yes | no |
| Core v1 | yes | yes | yes | yes | no | no |
| Core + structure | yes | yes | yes | yes | yes | no |
| Full candidate | yes | yes | yes | yes | yes | yes |

Use identical:

- Outer walk-forward splits.
- Universe and point-in-time eligibility rules.
- Training epochs and optimizer settings.
- Rebalance schedule.
- Seed set.
- Transaction costs.
- Evaluation and reporting path.

Report:

- Net annualized return and Sharpe.
- SPY-relative return and information ratio.
- Maximum drawdown and downside deviation.
- Turnover and transaction-cost drag.
- Weight concentration.
- Performance by outer test split.
- Performance contribution by ticker and sector where metadata exists.
- Stability across deterministic seeds.
- Feature coverage and distribution drift.

Stress:

- 0, 5, 10, and 20 bps transaction costs.
- One-session execution delay.
- Daily versus weekly rebalance where supported.
- Reasonable neighboring lookback windows.
- Removal of the best single ticker and best single test year.

Do not select features from aggregate Sharpe alone. Prefer additions that improve
several outer splits, remain useful after costs, and do not create unacceptable
turnover or concentration.

## Test plan

Every new production feature requires:

1. A hand-calculated unit fixture.
2. A future-mutation no-look-ahead test.
3. A per-ticker isolation test.
4. A missing-data and constant-series test.
5. A feature-schema and routing test.
6. An end-to-end feature-bundle test.
7. A decision-date alignment test.
8. A preprocessing finite-output test.
9. A DPO panel-shape test.
10. A small deterministic walk-forward smoke test.

Structure features additionally require explicit confirmation-date tests.
Cross-sectional features require same-date-only rank tests. Market-relative
features require benchmark-publication and date-alignment tests.

## Acceptance gate for a new routed feature

A feature may enter the preferred DPO configuration only if:

- Its economic hypothesis is written before running the outer tests.
- Its formula is deterministic and causal.
- It has sufficient point-in-time coverage.
- It adds information beyond close substitutes in correlation and ablation
  diagnostics.
- It improves net out-of-sample behavior across more than one split.
- Its benefit survives reasonable costs and parameter perturbations.
- It does not rely on one ticker, one year, or one seed.
- It has all required tests.
- Its exact definition and routing order are serialized.

Failure to improve does not justify silently retuning on the outer test set.
Rejected features should remain documented with their experiment identifiers.

## Non-goals

This plan does not authorize or include:

- Short selling or leverage.
- A learned cash allocation.
- Intraday order-book or trade classification from daily bars.
- Fundamental, analyst-estimate, options, news, or alternative-data features.
- Forward-backward HMM smoothing.
- Updating existing architecture documents.
- Treating technical-analysis terminology as evidence of predictive edge.

## Recommended first implementation slice

The first vertical delivery should be deliberately small:

1. Add named feature routing while preserving `baseline_current_14`.
2. Implement `mom_21d`, `mom_126_21d`, and `near_52w_high`.
3. Add optional same-date percentile ranks for those three features.
4. Add unit, causality, routing, panel, and walk-forward smoke tests.
5. Run baseline versus raw momentum versus ranked momentum under identical
   settings.
6. Save a comparison artifact before starting volatility or structure work.

This slice exercises data ingestion, feature engineering, preprocessing,
routing, DPO, and walk-forward evaluation without introducing pivot-state
complexity prematurely.

## Research references

- Jegadeesh, N., and Titman, S. (1993), [Returns to Buying Winners and Selling
  Losers](https://www.bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf).
- Moskowitz, T. J., Ooi, Y. H., and Pedersen, L. H. (2012), [Time Series
  Momentum](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf).
- George, T. J., and Hwang, C.-Y. (2004), [The 52-Week High and Momentum
  Investing](https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1540-6261.2004.00695.x).
- Brock, W., Lakonishok, J., and LeBaron, B. (1992), [Simple Technical Trading
  Rules and the Stochastic Properties of Stock
  Returns](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1992.tb04681.x).
- Lo, A. W., Mamaysky, H., and Wang, J. (2000), [Foundations of Technical
  Analysis](https://www.nber.org/papers/w7613).
- Amihud, Y. (2002), [Illiquidity and Stock Returns: Cross-Section and
  Time-Series Effects](https://www.sciencedirect.com/science/article/pii/S1386418101000246).
- Corwin, S. A., and Schultz, P. (2012), [A Simple Way to Estimate Bid-Ask
  Spreads from Daily High and Low
  Prices](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2012.01729.x).
- Moreira, A., and Muir, T. (2017), [Volatility-Managed
  Portfolios](https://www.nber.org/papers/w22208).
- Barroso, P., and Santa-Clara, P. (2015), [Momentum Has Its
  Moments](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2041429).
