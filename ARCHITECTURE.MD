# AI-Native Quantitative Trading Framework

## Objective

Develop an open-source, AI-native quantitative trading framework that combines:

* JAX-native GPU execution
* Regime-aware reinforcement learning
* LSTM temporal representation learning
* Cross-asset attention
* Hawkes-process features
* Spectral market features
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

* Volume eigenspectrum
* Liquidity eigenspectrum
* Sector flow indicators

---

# Representation Learning

Input:

Shape:

(60, 100, F_asset)

60 trading-day lookback.

---

## Asset Encoder

Shared Asset LSTM

Hidden size:

64

Output:

(100, 64)

---

## Cross-Asset Modeling

Cross-Asset Self-Attention

Output:

(100, 64)

---

## Aggregation

Attention Pooling

Output:

(64)

---

## Macro Encoder

Macro LSTM

Hidden size:

16

Output:

(16)

---

## Fusion

Inputs:

* Asset embedding (64)
* Macro embedding (16)
* Spectral features (20)

Concatenate:

(100)

Fusion MLP:

100
→ 64
→ 32

Market state:

φ_t ∈ R^32

---

# Regime Detection

Model:

Gaussian Hidden Markov Model

Configuration:

* Diagonal covariance
* Default: 4 states
* Configurable

Input:

φ_t

Output:

Posterior regime probabilities:

p(k_t)

Training schedule:

* Annual retraining
* 10-year rolling window

Only filtering probabilities may be used:

P(k_t | x_1:t)

Forward-backward smoothing is prohibited.

---

# PPO

## State

s_t = (φ_t, p(k_t), C_t)

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

State vector

Architecture:

140
→ 128
→ 128
→ 101 logits

Portfolio construction:

Temperature Softmax

Temperature is configurable.

---

## Critic

Architecture:

140
→ 128
→ 64
→ 1

---

# Trading Environment

## Observation Schedule

Friday close:

* Compute features
* Build state
* Infer regime probabilities
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

r_t = log(1 + R_net) - log(1 + R_SPY) - λ_DD max(0, DD_t - DD_max) - λ_TO TO_t

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

Policy is frozen during test periods.

No online learning during evaluation.

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

* PPO without HMM
* PPO without Hawkes features
* PPO without Spectral features
* PPO without Cross-Asset Attention

---

# Current Development Strategy

Implementation order:

1. Data layer
2. Feature engine
3. Trading environment
4. Benchmark strategies
5. HMM
6. Encoder
7. PPO
8. Walk-forward engine

Do not change architectural assumptions without explicit approval.
