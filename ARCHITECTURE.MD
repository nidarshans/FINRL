# AI-Native Quantitative Trading Framework

## Objective

Develop an open-source, AI-native quantitative trading framework that combines:

* JAX-native GPU execution
* Asset-only reinforcement learning
* LSTM temporal representation learning
* Hawkes-process features
* Public market data
* PPO portfolio optimization

Target deployment:

* Google Colab
* Single GPU
* User-defined 100-stock universe

---

# Data

## Asset Features

### Momentum

* Returns
* RSI
* MACD
* Trend slope

### Liquidity

* Amihud illiquidity
* Dollar volume
* Turnover
* Liquidity deterioration
* Liquidity shock

### Volume

* Volume momentum
* Volume acceleration

### Relative Features

* Cross-sectional percentile ranks

### Hawkes Features

* Branching ratio (BR)
* Intensity (λ)
* Intensity acceleration (Δλ)
* Half-life (HL)
* Endogenous fraction (ENDO)
* Hawkes surprise residual

---

## Macro Features

* VIX
* Oil
* Fed Funds Rate
* 10Y Treasury Yield
* Gold
* Copper

---

## Spectral Features

Spectral feature artifacts are not part of the active PPO state.

Compatibility tables may retain empty or placeholder spectral outputs for reporting
interfaces, but PPO must not depend on spectral features.

---

# Representation Learning

Input:

Shape:

(60, 100, F_asset)

60 trading-day lookback.

---

## Explicit Score-Head Feature Routing

The learned score heads must use explicit allowlists, not prefix matching.

The accumulation MLP receives exactly these columns, in this order:

1. acc_price_drift
2. acc_liquidity_growth
3. acc_vol_compression
4. acc_low_vol
5. acc_macd_improvement
6. acc_klinger_improvement
7. acc_macd_bullish_hist
8. acc_klinger_bullish_hist
9. acc_macd_early
10. acc_klinger_early

Shape entering the accumulation MLP:

(B, L, N, 10)

The liquidity-exit MLP receives exactly these columns, in this order:

1. liq_amihud_trend
2. liq_liquidity_deterioration
3. liq_klinger_deterioration
4. liq_vol_expansion
5. liq_liquidity_shock

Shape entering the liquidity-exit MLP:

(B, L, N, 5)

Routing rules:

* Feature names are selected from explicit allowlists.
* Missing required columns raise ValueError.
* Selected feature names are converted to integer indices before entering JAX/Flax.
* Diagnostic metadata stores both selected feature names and selected indices.
* Prefix-only routing is prohibited for score-head inputs.

---

## Asset Encoder

Shared Asset LSTM

Hidden size:

64

Output:

(100, 64)

---

## Score Heads

Two learned MLP score heads operate on explicit asset feature slices:

* Accumulation score head
* Liquidity-exit score head

The scalar scores are concatenated back onto the full asset feature window before
the shared LSTM.

The active PPO pipeline has a single trainable encoder path. PPO stores raw asset
windows in rollouts and recomputes encoder outputs during updates so gradients
flow through:

* Score heads
* Shared LSTM
* Actor
* Critic

---

# PPO

## State

s_t = (E_t, C_t)

E_t is the per-asset embedding matrix from the asset encoder:

(100, 64)

Portfolio context C_t contains:

* Current portfolio weights
* Current drawdown
* Previous turnover

---

## Action Space

Target portfolio weights.

The policy outputs:

w_target

including cash.

Universe:

* 100 stocks
* 1 cash asset

Total assets:

101

Constraints:

Σw = 1

w ≥ 0

Long-only.

---

## Actor

Input:

* Per-asset embeddings
* Portfolio context

Architecture:

Configurable MLP over asset embeddings and portfolio context.

Output:

101 logits

Portfolio construction:

Long-only simplex action distribution.

---

## Critic

Input:

* Mean-pooled asset embeddings
* Max-pooled asset embeddings
* Portfolio context

Output:

Scalar value estimate

---

# Trading Environment

## Observation Schedule

Friday close:

* Compute features
* Build state
* Generate action

No trading occurs.

---

## Execution Schedule

Monday open:

* Rebalance to target weights

Hold until next Monday open.

---

## Transaction Costs

Turnover:

TO_t = Σ |w_target - w_current|

Transaction cost:

TC_t = c × TO_t

Default:

c = 10 bps

---

## Cash Asset

Cash is included as a tradable asset.

Cash return may be linked to the risk-free rate.

---

## Portfolio Return

Gross return:

R_gross

Net return:

R_net = R_gross - TC_t

Portfolio value update:

V_(t+1) = V_t × (1 + R_net)

---

## Drawdown

Peak value:

V_peak

Drawdown:

DD_t = 1 - V_t / V_peak

---

# Reward Function

Default reward:

r_t = log(1 + R_net) - log(1 + R_SPY)

Optional penalties may be enabled by configuration:

* Drawdown penalty
* Sortino downside penalty
* Turnover penalty
* Active risk penalty

Default penalty weights are zero.

Reward functions must be fully pluggable.

Benchmark:

SPY

The primary objective is risk-adjusted alpha generation relative to SPY.

---

# Walk-Forward Training

Training window:

10 years

Testing window:

1 year

Annual rolling retraining.

Example:

Train: 2010-2019
Test: 2020

Train: 2011-2020
Test: 2021

Evaluation uses the train-fitted policy parameters for each split.

No online learning or parameter updates occur during evaluation.

---

# JAX Stack

Libraries:

* JAX
* Flax
* Optax
* Distrax
* Orbax

Execution principles:

* jit
* scan
* vmap

GPU-first design.

---

# Benchmark Suite

Required benchmarks:

* SPY
* Equal Weight
* Risk Parity
* Momentum Top-K

Required ablations:

* PPO without Hawkes features
* PPO without learned score heads
* PPO without selected accumulation inputs
* PPO without selected liquidity-exit inputs

---

# Current Development Strategy

Implementation order:

1. Data layer
2. Feature engine
3. Trading environment
4. Benchmark strategies
5. Encoder
6. PPO
7. Walk-forward engine

Do not change architectural assumptions without explicit approval.
