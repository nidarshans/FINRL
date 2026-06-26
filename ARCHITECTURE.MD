# AI-Native Quantitative Trading Framework

## Authority

The implementation is now the source of truth.

This document summarizes the current architecture and intended boundaries. When
the code and this document disagree, update this document to match the code
unless the user explicitly requests an implementation change.

---

# Objective

Build a JAX-native research framework for weekly long-only portfolio allocation
over a configurable stock universe plus cash.

The active system combines:

* Public OHLCV-style market data
* Split-safe feature engineering
* JAX-native weekly portfolio accounting
* Asset-only temporal representation learning
* Learned accumulation and liquidity-exit score heads
* PPO portfolio optimization
* Optional direct portfolio optimization experiments
* Strict walk-forward evaluation

Default deployment assumptions:

* Local CPU for tests, accounting checks, and toy fixtures
* Google Colab single GPU for full training runs
* 100 risky assets plus one cash asset by default

---

# Repository Layers

* `finrl.data`: market data sources, storage, calendars, universes, validation
* `finrl.features`: asset, macro, relative, Hawkes, preprocessing, feature schema
* `finrl.env`: accounting, rewards, schedules, weekly trading environment
* `finrl.models`: asset-only encoder, score heads, window construction
* `finrl.ppo`: actor-critic models, simplex distribution, rollout, losses, GAE
* `finrl.dpo_jax`: differentiable direct portfolio optimization
* `finrl.backtest`: walk-forward split generation and result containers
* `finrl.experiments`: experiment configuration, runners, artifacts, reporting
* `finrl.logging`: TensorBoard integration

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
Spectral artifacts are not part of the active PPO or DPO state.

---

# Asset Encoder

The active encoder consumes asset feature windows:

```text
(B, L, N, F_asset)
```

Defaults:

* `L = 60` trading days
* `N = 100` risky assets
* `F_asset` comes from the feature schema

The encoder path:

1. Select score-head inputs using explicit feature allowlists.
2. Compute accumulation and liquidity-exit scalar scores.
3. Concatenate the two scores onto the raw asset feature window.
4. Apply a shared LSTM across each asset's time sequence.
5. Optionally normalize embeddings.

Default embedding shape:

```text
(B, N, 64)
```

## Score-Head Routing

The accumulation head receives:

1. `acc_price_drift`
2. `acc_liquidity_growth`
3. `acc_vol_compression`
4. `acc_low_vol`
5. `acc_macd_improvement`
6. `acc_klinger_improvement`
7. `acc_macd_bullish_hist`
8. `acc_klinger_bullish_hist`
9. `acc_macd_early`
10. `acc_klinger_early`

The liquidity-exit head receives:

1. `liq_amihud_trend`
2. `liq_liquidity_deterioration`
3. `liq_klinger_deterioration`
4. `liq_vol_expansion`
5. `liq_liquidity_shock`

Missing required columns raise `ValueError`. Feature names are converted to
integer indices before entering JAX/Flax modules. Prefix-only routing is
prohibited.

---

# PPO Portfolio Optimization

PPO is the main reinforcement-learning path.

State:

```text
s_t = E_t
```

Where `E_t` is the per-asset embedding matrix with shape `(N, 64)`.

The PPO actor and critic are intentionally independent at each decision step.
They do not consume previous portfolio weights, current drawdown, or previous
turnover. Those accounting values still exist in the environment and rollout
diagnostics for transaction costs, metrics, and GAE targets.

The actor emits `N + 1` logits, including cash. Risky asset logits are computed
from per-asset embeddings. The cash logit is computed from pooled current asset
embeddings. The policy samples or evaluates long-only simplex weights.
Deterministic evaluation uses the distribution mean.

The critic estimates scalar value from pooled current asset embeddings only.

The PPO update path includes clipped policy loss, value loss, GAE, gradient
clipping, optional entropy terms, and target-KL diagnostics.

---

# Direct Portfolio Optimization

`finrl.dpo_jax` implements a differentiable direct portfolio optimization path.
It is an alternate learned policy mode, not a replacement for PPO.

The DPO path reuses the LSTM-based asset-only encoder and trains:

* Encoder parameters
* A `DirectAllocationHead`

The active DPO data flow is:

```text
asset_windows: (T, L, N, F)
  -> score heads:
       accumulation MLP with configurable hidden dims and optional LayerNorm
       liquidity-exit MLP with configurable hidden dims and optional LayerNorm
  -> concatenate scores onto the asset feature window
  -> shared LSTM with configurable hidden size from AssetOnlyEncoderConfig
  -> asset_embeddings: (T, N, D)
  -> DirectAllocationHead with configurable hidden dims and optional LayerNorm
  -> logits: (T, N + 1)
  -> sparsemax/softmax
  -> weights: (T, N + 1)
  -> dpo_loss

weights: (T, N + 1)
asset_returns: (T, N)
```

Supported allocation activations:

* `sparsemax`
* `softmax`

`sparsemax` projects logits onto the simplex and may produce exact zero weights.

The DPO objective is a scan-based differentiable backtest. It accounts for:

* Stock returns
* Cash weight
* Turnover from previous weights
* Transaction costs from configured bps
* Equity curve updates
* Running maximum equity
* Drawdown
* Concentration diagnostics

Current loss behavior follows the code in `finrl.dpo_jax.losses`: it optimizes
mean log return with a drawdown penalty. Turnover and concentration are reported
as metrics and have configuration fields, but they are not currently included in
the loss expression.

Experiment configuration enforces that PPO and DPO are not enabled at the same
time.

---

# Trading Environment

One environment step represents:

1. Friday close feature/state/action generation
2. Monday open rebalance to target weights
3. Transaction cost charge
4. Hold until the next Monday open

No trade is executed at Friday close.

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

Required PPO ablations:

* Without Hawkes features
* Without learned score heads
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
* Weekly rebalance logic

Additional coverage should include JAX `jit` and `scan` compatibility,
long-only normalization, split-safe preprocessing, no-look-ahead windows,
score-head routing, PPO losses, DPO losses, and walk-forward boundaries.

Tests should use small deterministic arrays and pass on CPU-only JAX.
