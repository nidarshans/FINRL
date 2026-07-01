# AI-Native Quantitative Trading Framework

## Authority

This document is the authoritative specification. If code conflicts with this
document, update the code. Architecture changes require explicit user approval.

---

# Objective

Build a JAX-native research framework for configurable daily or weekly long-only portfolio allocation
over a configurable stock universe plus cash.

The active system combines:

* Public OHLCV-style market data
* Split-safe feature engineering
* JAX-native daily and weekly portfolio accounting
* Learned direct-feature allocation head
* Direct portfolio optimization
* Strict walk-forward evaluation

Default deployment assumptions:

* Local CPU for tests, accounting checks, and toy fixtures
* Google Colab single GPU for full training runs
* 100 risky assets plus one cash asset by default

---

# Repository Layers

* `finrl.data`: market data sources, storage, calendars, universes, validation
* `finrl.features`: asset, macro, relative, Hawkes, preprocessing, feature schema
* `finrl.env`: accounting, rewards, schedules, trading environment
* `finrl.dpo_jax`: differentiable direct portfolio optimization
* `finrl.backtest`: walk-forward split generation and result containers
* `finrl.experiments`: experiment configuration, runners, artifacts, reporting

Core financial logic should remain deterministic, functional, JAX-compatible,
CPU-testable, and free of hidden mutable state.

---

# Data and Features

The stock universe is user-defined and fixed for an experiment. The code should
support configurable `N`; `100` is a default, not a universal constant.

Market data must be chronological, calendar-aligned, and validated before
feature generation. Feature generation and preprocessing are split-safe:

1. Fit preprocessing only on the train window.
2. Transform train and test with the train-fitted preprocessing object.
3. Never fit on the test window or full dataset.
4. Compute cross-sectional ranks independently per date.

Asset-level feature groups include momentum, liquidity, volume, relative, and
Hawkes features. Macro features may be used in offline features or diagnostics.
Spectral artifacts are not part of the active DPO state.

---

# Direct-Feature Allocation Policy

The active policy consumes decision-date asset features:

```text
(B, N, F_asset)
```

Defaults:

* `N = 100` risky assets
* `F_asset` comes from the feature schema

The policy path:

1. Select direct-allocation inputs using an explicit feature allowlist.
2. Feed the selected per-asset features to the shared allocation head.
3. Apply the configured simplex activation across stock logits.
4. Append an exact zero cash weight.

Input and allocation shapes:

```text
direct features: (B, N, 3)
weights:         (B, N + 1)
```

There is no intermediate learned layer, recurrent encoder, or model lookback
window. Trailing temporal information belongs in split-safe feature calculations.

## Direct-Allocation Routing

The allocation head receives:

1. `mr_ewma50_vol_gap`
2. `acc_macd_signal`
3. `acc_klinger_signal`

Missing required columns raise `ValueError`. Feature names are converted to
integer indices before entering JAX/Flax modules. Prefix-only routing is
prohibited.

---

# Direct Portfolio Optimization

`finrl.dpo_jax` implements differentiable direct portfolio optimization and
trains a `DirectAllocationHead` over explicitly routed asset features.

The active DPO data flow is:

```text
asset_features: (T, N, F)
  -> select direct allocation features: (T, N, 3)
  -> shared per-stock DirectAllocationHead with optional hidden dims
  -> logits: (T, N + 1)
  -> simplex activation
  -> weights: (T, N + 1)
  -> dpo_loss

weights: (T, N + 1)
asset_returns: (T, N)
```

Allocation logits remain unrestricted real values. Softmax jointly normalizes
all stock logits to strictly positive stock weights summing to one. Before
softmax, each stock is processed independently by the same shared allocation
function; no stock-specific parameters or cross-asset mixing are used. Cash is
excluded from the projection and its appended portfolio weight is always zero.
The `N + 1` output shape is retained for environment and accounting compatibility.

The DPO objective is a scan-based differentiable backtest. It accounts for:

* Stock returns
* SPY benchmark returns
* Cash weight
* Turnover from previous weights
* Transaction costs from configured bps
* Equity curve updates
* Running maximum equity
* Drawdown
* Concentration diagnostics

Current loss behavior follows the code in `finrl.dpo_jax.losses`: it optimizes
mean active log return relative to SPY with a drawdown penalty. Turnover and
concentration are reported as metrics and have configuration fields, but they
are not currently included in the loss expression.

Experiment configuration can disable DPO to run the deterministic
environment-only equal-weight path.

---

# Trading Environment

Rebalance frequency is configurable as `weekly` or `daily`.

Weekly mode uses the final trading session of each week as the decision date,
including Thursday when Friday is a market holiday. Daily mode uses every
completed trading session as a decision date.

One weekly environment step represents:

1. Friday close feature/state/action generation
2. Monday open rebalance to target weights
3. Transaction cost charge
4. Hold until the next Monday open

No trade is executed at Friday close.

One daily environment step uses the same causal ordering:

1. Trading-day close feature/state/action generation
2. Next-session open rebalance
3. Transaction cost charge
4. Hold until the following session open

Annualized metrics use 52 periods for weekly mode and 252 periods for daily
mode unless an explicit test or experiment override is supplied.

Accounting:

```text
TO_t = sum(abs(w_target - w_current))
TC_t = transaction_cost_rate * TO_t
R_gross = dot(w_executed, r_assets)
R_net = R_gross - TC_t
V_(t+1) = V_t * (1 + R_net)
DD_t = 1 - V_t / V_peak
```

Default transaction cost rate is `0.001`, equivalent to 10 bps.

Default reward:

```text
r_t = log(1 + R_net) - log(1 + R_SPY)
```

Optional configured penalties include drawdown, Sortino downside, turnover, and
active risk.

---

# Benchmarks and Walk-Forward Evaluation

Required benchmark strategies:

* SPY
* Equal Weight
* Risk Parity
* Momentum Top-K

Required learned-policy ablations:

* Without Hawkes features
* Without selected accumulation inputs
* Without selected liquidity-exit inputs

Default walk-forward setup:

* 10-year train window
* 1-year test window
* Annual rolling retraining

Evaluation uses train-fitted preprocessing and train-fitted policy parameters.
No model updates occur during evaluation.

---

# Regime Diagnostics

Gaussian HMM regime diagnostics may be used after the core data, feature,
environment, and benchmark paths are validated.

HMM inference must use filtering only:

```text
P(k_t | x_1:t)
```

Forward-backward smoothing is prohibited for evaluation features.

Regime outputs are diagnostics unless the code explicitly wires them into a
learned policy path.

---

# Testing Requirements

All core financial calculations require unit tests, including:

* Turnover
* Transaction costs
* Portfolio value evolution
* Drawdown
* Benchmark returns
* Cash accounting
* Weekly and daily rebalance logic
* Holiday-adjusted weekly decision logic

Additional coverage should include JAX `jit` and `scan` compatibility,
long-only normalization, split-safe preprocessing, decision-date panel
alignment, direct-feature routing, allocation, DPO losses, and
walk-forward boundaries.

Tests should use small deterministic arrays and pass on CPU-only JAX.
