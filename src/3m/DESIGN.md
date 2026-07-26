# 3M: Three-Model Buy, Hold, and Sell Policy

## Status

The initial vertical slice is implemented in `finrl.three_m`: 50-column feature
routing, labels, pooled scikit-learn classifiers, state gating, cash-aware
allocation, and a chronological frozen split runner. It is covered by focused
deterministic tests.

`ARCHITECTURE.MD` remains unchanged. 3M is an experimental policy path until a
broader architecture update formally makes it part of the active system.

## Goal

3M is a long-only portfolio policy for a configurable universe of approximately
50 stocks. It is intended to reduce unnecessary exits from persistent winners
while still learning distinct entry and exit conditions.

The policy has three gradient-boosted binary classifiers:

1. **Buy model**: estimates whether a currently unheld asset is an attractive
   new position.
2. **Hold model**: estimates whether a currently held asset still has an
   attractive forward return path.
3. **Sell model**: estimates whether a currently held asset should be exited.

Each classifier is one shared model trained on pooled `(date, asset)` rows. No
asset receives its own fitted model and ticker identity is not an input. At
inference, each asset is scored separately and the current portfolio state
determines which actions are eligible.

The core hypothesis is not merely that a model can predict positive returns.
It is that separate entry, continuation, and exit objectives plus explicit
decision hysteresis can increase average holding time, reduce turnover, and
retain exposure to persistent compounders.

## Library Choice

Use `sklearn.ensemble.HistGradientBoostingClassifier` for all three models.
Do not implement boosting, trees, split finding, or probability prediction in
this repository.

Reasons for selecting the scikit-learn implementation:

- mature, maintained CPU implementation;
- supports non-linear interactions without feature scaling;
- handles the pooled tabular dataset expected here;
- exposes `predict_proba` for threshold-based action decisions;
- has few operational dependencies and fits the single-Colab-machine target;
- supports deterministic fitting when configuration and input order are fixed.

Initial training should set `early_stopping=False`. Scikit-learn's automatic
validation split is not a substitute for a chronological financial validation
window. Hyperparameters and action thresholds must be selected with explicit
time-ordered splits.

The initial model configuration should expose:

- `learning_rate`
- `max_iter`
- `max_leaf_nodes`
- `max_depth`
- `min_samples_leaf`
- `l2_regularization`
- `max_bins`
- `random_state`

Class imbalance should be handled with train-only `sample_weight`, calculated
separately for each of the three targets. Default random cross-validation is
prohibited.

## Feature Contract

3M computes and routes the ordered, de-duplicated union of every named feature
column declared in `finrl.features.columns`. This currently produces 50 asset
features:

### Baseline

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

### Momentum and cross-sectional momentum

15. `mom_21d`
16. `mom_126_21d`
17. `near_52w_high`
18. `mom_21d_percentile_rank`
19. `mom_126_21d_percentile_rank`
20. `near_52w_high_percentile_rank`

### Liquidity and cross-sectional liquidity

21. `log_adv_20`
22. `volume_z_20`
23. `amihud_20`
24. `log_adv_20_percentile_rank`
25. `volume_z_20_percentile_rank`
26. `amihud_20_percentile_rank`

### Market structure

27. `confirmed_structure_score`
28. `support_distance_atr`
29. `resistance_distance_atr`
30. `swing_avwap_distance_atr`
31. `bars_since_swing_low`

### Market-relative

32. `relative_strength_63`
33. `beta_252`
34. `residual_mom_126_21`
35. `idio_vol_60`

### Risk

36. `natr_20`
37. `realized_vol_20`
38. `downside_vol_60`
39. `max_drawdown_126`

### Volume, price trend, and volatility

40. `close_vwap20_gap`
41. `close_ema20_gap`
42. `close_ema50_gap`
43. `close_ema200_gap`
44. `ema20_ema50_distance`
45. `ema50_ema200_distance`
46. `ema20_ema200_distance`
47. `ema20_slope`
48. `ema50_slope`
49. `ema200_slope`
50. `realized_vol_126`

`volume_z_20`, `cmf`, and `cmf_slope` occur in more than one existing group but
must appear only once in the 3M matrix.

### Reuse of the existing feature pipeline

The current implementation already computes the underlying raw columns through:

- `finrl.features.asset.compute_asset_features`
- `finrl.features.market_relative.compute_market_relative_features`
- `finrl.features.relative.cross_sectional_percentile_rank`
- `finrl.features.pipeline.build_feature_bundle`

Implementation should add one explicit ordered constant, tentatively
`THREE_M_FEATURE_COLUMNS`, and one named feature set,
`three_m_all_v1`, to `finrl.features.columns`. The feature set must be built
from explicit column names, not prefixes.

Selecting `three_m_all_v1` must cause the existing pipeline to:

1. compute the trailing per-asset features;
2. join market-relative features by actual date;
3. compute momentum and liquidity ranks independently within each date;
4. select the exact 50-column ordered allowlist;
5. fail before fitting if a column is absent;
6. preserve the fixed ticker order in the panel.

The longest current lookback is 252 trading sessions. Warm-up rows with
incomplete information must not silently turn into apparently valid signals.
The implementation should expose an explicit feature-valid mask and train only
on rows where every required feature is available. Forward filling may use
past observations only; backfilling is prohibited.

Trees do not require z-score scaling. Existing causal preprocessing may still
be used for finite-value handling and train-configured clipping, but it must
not fit statistics on test data. Cross-sectional ranks remain date-local.

## Inputs and Outputs

For `T` decision dates, `N` assets, and `F=50` features:

```text
asset feature panel:       (T, N, 50)
pooled training matrix:    (T_valid * N_valid, 50)
buy probabilities:         (T, N)
hold probabilities:        (T, N)
sell probabilities:        (T, N)
portfolio weights:         (T, N + 1)
```

The last portfolio column is cash. Unlike the active DPO path, 3M needs usable
cash so that existing holdings are not trimmed merely to fund every new buy.

## Target Construction

Features at decision date `t` may use only information available at that
completed close. Labels may inspect future returns during training, but labels
must never be included as features and must never cross from a training window
into validation or test.

All target returns must follow the environment's execution timing:

```text
decision at completed close t
trade at next session open
measure subsequent open-to-open returns
```

Use causal crossover events plus a cost-aware forward outcome. Let:

- `R_h(t, i)` be the compounded forward return for asset `i` over `h` sessions;
- `DD_h(t, i)` be the minimum cumulative return within the forward horizon;
- `c` be estimated round-trip transaction cost;
- `g_20`, `g_50`, and `g_vwap` be decision-close distances to EMA-20, EMA-50,
  and 20-session VWAP.

All thresholds are configuration values and are fitted or selected using
training data only.

### Buy target

`y_buy=1` when price crosses above EMA-20, EMA-50, or VWAP-20 at `t`, and
`R_20(t, i)` is at least the configured buy threshold plus `c`. The initial
threshold is 5%.

### Hold target

`y_hold=1` when price is above EMA-20 or VWAP-20, or is no more than the
configured epsilon below EMA-50. The initial EMA-50 mean-reversion tolerance
is 1% of decision-close price.

### Sell target

`y_sell=1` when price crosses below EMA-20, EMA-50, or VWAP-20 at `t`, and
the following 20-session path has `DD_20(t, i)` at or below the configured
sell drawdown threshold plus `c`. The initial threshold is 5%.

These are three related binary tasks, not a single three-class target. Buy and
hold can both be favorable for the same market row because eligibility is
resolved by actual portfolio state. Hold and sell disagreements are resolved
by the deterministic policy described below.

The default forward outcome horizon is 20 trading sessions. The 5/20/60
horizons remain available for rolling-barrier diagnostics, not event labels.

The final `outcome_horizon - 1` training rows cannot have complete labels and must be
dropped. Walk-forward splits must purge every training observation whose label
window overlaps validation or test. An optional embargo starts after this
purge, not instead of it.

## Training

One outer walk-forward split performs:

1. Slice the raw train and test dates.
2. Fit or configure preprocessing from the train window only.
3. Compute labels only from returns wholly contained in the train window.
4. Purge incomplete and boundary-crossing labels.
5. Flatten valid train rows in date-major, asset-minor order.
6. Fit independent buy, hold, and sell classifiers with fixed seeds.
7. Freeze all three estimators and thresholds.
8. Run the test period chronologically without model updates.

Every asset from a decision date must stay in the same temporal fold. Splitting
the flattened matrix with a random row splitter would leak date-specific market
conditions between train and validation.

Initial probability thresholds should be tuned on a trailing chronological
validation slice inside the training window. Probability calibration is
deferred until the uncalibrated benchmark works. If calibration is added, it
must also use an explicit chronological calibration slice.

## Per-Asset Decision State Machine

Portfolio state gates the model outputs:

```text
if asset is not held:
    BUY  if p_buy >= buy_threshold
    FLAT otherwise

if asset is held:
    SELL if p_sell >= sell_threshold
         or p_hold < hold_threshold
    HOLD otherwise
```

An optional confirmation count may require a sell condition on more than one
decision date. The first implementation should keep this at one and add it only
as a tested configuration option.

This gate intentionally creates hysteresis:

- the buy model cannot repeatedly resize an existing position;
- the sell model cannot create a short position;
- a held asset remains untouched unless continuation evidence weakens or exit
  evidence becomes strong.

Position state is used by the deterministic gate, not as a model feature in
version 1. Adding holding age, cost basis, current weight, or unrealized return
to the training matrix requires causal out-of-fold policy rollouts. Constructing
those fields from future-informed training labels would leak future information.

## Portfolio Construction

Model decisions are per asset, but cash and portfolio limits are global. Use
this deterministic order at each rebalance:

1. Execute sells and release their capital.
2. Carry held positions at their current drifted weights without rebalancing.
3. Rank eligible buys by `p_buy - buy_threshold`.
4. Admit candidates until the maximum position count or available cash is
   exhausted.
5. Size each new position using a configured entry weight, capped by the
   position limit.
6. Leave unused capital in cash.

Do not proportionally rescale held positions every period. That would turn
`HOLD` into a rebalance and recreate avoidable turnover. If cash is
insufficient, skip lower-ranked buys rather than trimming holds.

The initial benchmark should use fixed entry weights and a maximum number of
positions. Volatility targeting or risk-parity sizing can be a later ablation.
Position caps apply at entry; a separate, explicit risk rule is required if
natural appreciation above the cap must force a trim.

The portfolio transition should be a pure deterministic function of:

```text
current drifted weights
tradable mask
buy/hold/sell probabilities
policy thresholds
portfolio constraints
```

It should return target weights, action codes, and reasons for rejected buys.
Suspended or otherwise non-tradable assets must not be sold or bought until
tradable; their last valid position remains accounted for.

## Backtest and Evaluation

Use the existing daily or holiday-adjusted weekly decision schedule and
next-session-open execution. Start with the existing default walk-forward
protocol: ten years of training, one year of frozen evaluation, and annual
retraining.

Always compare 3M against:

- SPY;
- equal weight;
- risk parity;
- momentum top-K;
- the existing single pooled GBT policy;
- a 3M ablation with immediate sell behavior and no hold model.

In addition to return, Sharpe, Sortino, and drawdown, report:

- turnover and transaction-cost drag;
- average and median holding period;
- percentage of positions held beyond 20 and 60 sessions;
- contribution to P&L by holding-period bucket;
- win rate and payoff ratio by completed trade;
- buy precision, hold continuation precision, and sell precision;
- action counts and rejected-buy reasons;
- cash utilization and average number of positions;
- performance by walk-forward year and asset.

The primary success criterion is better out-of-sample net performance with
lower turnover and longer profitable holding periods. Longer holding time by
itself is not success.

## Proposed Modules

The requested `src/3m` directory is not a valid normal Python package name
because Python identifiers cannot begin with a digit. Keep this directory for
the design, or explicitly map it to a valid import name during implementation.
The lower-risk repository convention is to place executable modules under
`src/finrl/three_m`:

```text
src/3m/
    DESIGN.md

src/finrl/three_m/
    __init__.py
    config.py          # frozen dataclass configuration
    features.py        # exact 50-column routing and validity mask
    labels.py          # causal-timing, purged target construction
    dataset.py         # pooled matrices and sample weights
    model.py           # three sklearn estimator wrappers
    policy.py          # per-asset state gate
    allocation.py      # cash-aware deterministic portfolio transition
    serialization.py   # estimators, metadata, thresholds, schema version
    runner.py          # walk-forward fit and frozen inference
```

Corresponding tests should live under `tests/three_m/`.

## Serialization and Reproducibility

Each saved 3M artifact must include:

- all three fitted scikit-learn estimators;
- scikit-learn version;
- ordered feature names and a schema hash;
- target definition and horizons;
- action thresholds;
- portfolio construction parameters;
- universe and ticker order;
- training date range and purge boundary;
- preprocessing metadata;
- random seeds;
- code or experiment version.

Inference must fail on feature-order, schema-version, or ticker-order mismatch.
Use a library-supported serializer such as `joblib`; never deserialize
untrusted model files.

## Testing Requirements

### Features

- the 3M allowlist is exactly ordered and contains 50 unique columns;
- every declared column is produced;
- ranks use only same-date rows;
- market-relative features use actual aligned dates;
- a future-row mutation cannot alter earlier features;
- incomplete warm-up rows are masked.

### Labels and splits

- barrier ordering and transaction-cost adjustment;
- buy, hold, and sell target boundary cases;
- the last horizon rows are removed;
- no label window crosses a split;
- all assets for one date remain in one fold.

### Models

- exactly three `HistGradientBoostingClassifier` estimators are fitted;
- fixed data and seeds produce identical probabilities;
- feature-order mismatch fails;
- each classifier uses only train rows and its own sample weights.

### Policy and accounting

- flat assets can only buy or remain flat;
- held assets can only hold or sell;
- a hold decision produces zero trade for that asset;
- sells occur before buys;
- insufficient cash rejects buys without trimming holds;
- position and portfolio limits hold;
- cash accounting, turnover, transaction costs, and value evolution are exact;
- daily, weekly, and holiday-adjusted weekly transitions remain correct.

### Walk-forward

- estimators are frozen during each test window;
- preprocessing is train-only;
- target purge and optional embargo are respected;
- serialization round-trips without changing predictions.

## Vertical Implementation Order

1. Add `three_m_all_v1` feature routing and its tests.
2. Add purged buy/hold/sell labels and deterministic fixtures.
3. Fit the three scikit-learn classifiers on a small pooled dataset.
4. Add the per-asset state gate.
5. Add cash-aware portfolio construction and accounting tests.
6. Run one deterministic benchmark backtest.
7. Integrate full walk-forward evaluation and artifact serialization.
8. Tune thresholds and model parameters only after the baseline is reproducible.

## Remaining Decisions

1. Decide whether to formalize 3M in `ARCHITECTURE.MD` as an active policy.
2. Select production decision frequency, entry weight, maximum position count,
   and target thresholds through walk-forward validation.
3. Integrate the split runner into an experiment configuration and artifact
   serialization path after the first benchmark backtest is reviewed.
