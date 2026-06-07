# Project Context

This project is an AI-native quantitative trading framework for regime-aware PPO portfolio optimization.

## Objective

Build a JAX-native trading system for a user-defined 100-stock universe plus cash.

The system should combine:

- LSTM temporal representation learning
- Cross-asset attention
- Macro features
- Hawkes-process behavioral/liquidity features
- Spectral market features
- Gaussian HMM regime detection
- PPO portfolio optimization
- SPY-relative reward optimization
- Walk-forward backtesting

Target environment:

- Google Colab
- Single GPU
- Open-source codebase

---

# Data

## Universe

The stock universe is a fixed user-selected set of 100 stocks.

The portfolio includes:

- 100 stocks
- 1 cash asset

Total allocation dimension:

\[
N + 1 = 101
\]

---

## Asset Features

For each stock \(i\) at date \(t\), compute asset-level features:

### Momentum

- Returns
- RSI
- MACD
- Trend slope

### Liquidity

- Amihud illiquidity
- Dollar volume
- Turnover

### Volume

- Volume momentum
- Volume acceleration

### Relative

- Cross-sectional percentile ranks

### Hawkes

- Branching ratio \(BR\)
- Intensity \(\lambda\)
- Intensity acceleration \(\Delta \lambda\)
- Half-life \(HL\)
- Endogenous fraction \(ENDO\)
- Hawkes surprise residual

Asset feature tensor:

\[
X^{asset}_t \in \mathbb{R}^{100 \times F_{asset}}
\]

Historical tensor:

\[
X^{asset} \in \mathbb{R}^{T \times 100 \times F_{asset}}
\]

---

## Macro Features

Macro features include:

- VIX
- Oil
- Fed Funds Rate
- 10Y Treasury Yield
- Gold
- Copper

Macro tensor:

\[
X^{macro} \in \mathbb{R}^{T \times F_{macro}}
\]

---

## Spectral Features

Spectral features include:

- Volume eigenspectrum
- Liquidity eigenspectrum
- Sector flow indicators

Spectral tensor:

\[
X^{spectral} \in \mathbb{R}^{T \times 20}
\]

---

# Preprocessing

sklearn pipelines may be used for offline preprocessing only.

Allowed sklearn usage:

- imputation
- scaling
- clipping / winsorization
- normalization
- train-window-only fitting

Forbidden sklearn usage:

- inside JAX environment
- inside PPO training loop
- inside rollout generation
- inside jit / scan logic

Walk-forward preprocessing rule:

\[
\text{fit only on train window}
\]

Then transform train and test:

\[
X_{train}^{scaled} = f_{train}(X_{train})
\]

\[
X_{test}^{scaled} = f_{train}(X_{test})
\]

Never fit preprocessing on the full dataset.

Cross-sectional percentile ranks are computed per date, not globally fit.

---

# Representation Learning

Input lookback window:

\[
L = 60
\]

Asset input:

\[
X^{asset}_{t-L+1:t}
\in
\mathbb{R}^{60 \times 100 \times F_{asset}}
\]

Macro input:

\[
X^{macro}_{t-L+1:t}
\in
\mathbb{R}^{60 \times F_{macro}}
\]

---

## Asset Encoder

Use a shared LSTM across assets.

For each asset \(i\):

\[
h_{i,t} = \text{LSTM}_{asset}(X^{asset}_{t-L+1:t,i})
\]

Hidden size:

\[
64
\]

Output:

\[
H_t \in \mathbb{R}^{100 \times 64}
\]

---

## Cross-Asset Attention

Apply self-attention across assets:

\[
\tilde{H}_t = \text{Attention}(H_t)
\]

Output:

\[
\tilde{H}_t \in \mathbb{R}^{100 \times 64}
\]

Purpose:

- model cross-asset relationships
- detect sector/cluster behavior
- allow flow and momentum propagation across related names

---

## Attention Pooling

Aggregate asset embeddings into one market embedding:

\[
e^{asset}_t = \text{AttentionPool}(\tilde{H}_t)
\]

Output:

\[
e^{asset}_t \in \mathbb{R}^{64}
\]

---

## Macro Encoder

Macro LSTM:

\[
e^{macro}_t =
\text{LSTM}_{macro}(X^{macro}_{t-L+1:t})
\]

Hidden size:

\[
16
\]

Output:

\[
e^{macro}_t \in \mathbb{R}^{16}
\]

---

## Fusion

Concatenate:

\[
z_t =
[
e^{asset}_t,
e^{macro}_t,
X^{spectral}_t
]
\]

Dimensions:

\[
64 + 16 + 20 = 100
\]

Fusion MLP:

\[
100 \rightarrow 64 \rightarrow 32
\]

Market state:

\[
\phi_t \in \mathbb{R}^{32}
\]

---

# Regime Detection

Use Gaussian HMM on market state:

\[
\phi_t
\]

Default:

- 4 hidden states
- diagonal covariance
- configurable state count

Output:

\[
p(k_t)
\]

where:

\[
p(k_t) \in \mathbb{R}^{4}
\]

Only filtering is allowed:

\[
p(k_t \mid x_{1:t})
\]

Smoothing is not allowed:

\[
p(k_t \mid x_{1:T})
\]

because smoothing leaks future information.

Retraining:

- annual
- 10-year rolling window

---

# PPO State

The PPO state is:

\[
s_t = (\phi_t, p(k_t), C_t)
\]

where:

\[
\phi_t \in \mathbb{R}^{32}
\]

\[
p(k_t) \in \mathbb{R}^{4}
\]

Portfolio context:

\[
C_t = (w_t, DD_t, TO_{t-1})
\]

where:

- \(w_t \in \mathbb{R}^{101}\)
- \(DD_t\) is current drawdown
- \(TO_{t-1}\) is previous turnover

Approximate state dimension:

\[
32 + 4 + 101 + 1 + 1 = 139
\]

---

# PPO Action

The actor outputs logits:

\[
z_t \in \mathbb{R}^{101}
\]

Temperature-softmax portfolio construction:

\[
w_{i,t}^{target}
=
\frac{\exp(z_{i,t}/T)}
{\sum_j \exp(z_{j,t}/T)}
\]

Constraints:

\[
\sum_i w_{i,t}^{target} = 1
\]

\[
w_{i,t}^{target} \ge 0
\]

Action interpretation:

The action is the target portfolio allocation.

The environment trades from current weights to target weights.

---

# Actor-Critic Architecture

## Actor

Input:

\[
s_t
\]

Architecture:

\[
139 \rightarrow 128 \rightarrow 128 \rightarrow 101
\]

Output:

\[
101
\]

target-weight logits.

---

## Critic

Input:

\[
s_t
\]

Architecture:

\[
139 \rightarrow 128 \rightarrow 64 \rightarrow 1
\]

Output:

\[
V(s_t)
\]

---

# Trading Environment

## Timeline

Use Option A execution timing.

At Friday close:

1. Compute all features using data available through Friday close.
2. Build market state \(\phi_t\).
3. Infer regime probabilities \(p(k_t)\).
4. Build PPO state \(s_t\).
5. Actor emits target weights \(w_t^{target}\).

No execution occurs Friday.

At Monday open:

1. Rebalance portfolio to target weights.
2. Pay transaction costs.
3. Hold portfolio until next Monday open.

One RL step equals one weekly portfolio decision.

---

## Returns

For stock \(i\), holding-period return:

\[
r_{i,t}
=
\frac{P^{open}_{i,t+1}}
{P^{open}_{i,t}}
-1
\]

Cash return:

\[
r_{cash,t}
\]

Can be zero initially or derived from Fed Funds:

\[
r_{cash,t}
=
\frac{r^{annual}_{ff,t}}{252}
\times d_t
\]

where \(d_t\) is number of trading days in the holding period.

---

## Turnover

Current weights before rebalance:

\[
w_t^{current}
\]

Target weights:

\[
w_t^{target}
\]

Turnover:

\[
TO_t =
\sum_i
|w_{i,t}^{target} - w_{i,t}^{current}|
\]

---

## Transaction Costs

Simple transaction cost model:

\[
TC_t = c \cdot TO_t
\]

Default:

\[
c = 10 \text{ bps} = 0.001
\]

---

## Portfolio Return

Executed weights:

\[
w_t^{exec} = w_t^{target}
\]

Gross portfolio return:

\[
R_t^{gross}
=
\sum_i
w_{i,t}^{exec} r_{i,t}
\]

Net return:

\[
R_t^{net}
=
R_t^{gross}
-
TC_t
\]

Portfolio value update:

\[
V_{t+1}
=
V_t(1 + R_t^{net})
\]

---

## Drawdown

Running peak:

\[
V_t^{peak}
=
\max(V_0,\ldots,V_t)
\]

Drawdown:

\[
DD_t
=
1 -
\frac{V_t}
{V_t^{peak}}
\]

---

# Benchmark

Benchmark is SPY.

SPY return over same Monday-open to Monday-open interval:

\[
R_t^{SPY}
=
\frac{P^{open}_{SPY,t+1}}
{P^{open}_{SPY,t}}
-1
\]

Benchmark timing must exactly match portfolio holding timing.

---

# Reward

Default reward:

\[
r_t =
\log(1 + R_t^{net})
-
\log(1 + R_t^{SPY})
-
\lambda_{DD}
\max(0, DD_t - DD_{max})
-
\lambda_{TO}TO_t
\]

Interpretation:

- positive reward means outperformance vs SPY after costs
- negative reward means underperformance vs SPY
- drawdown penalty discourages uncontrolled losses
- turnover penalty discourages unnecessary trading

Reward function must be fully pluggable.

---

# PPO Training

Use a single historical trajectory due to Colab constraints.

No parallel overlapping episodes in v1.

For each walk-forward train window:

1. Build states.
2. Run policy through historical environment.
3. Collect trajectory:

\[
(s_t, a_t, r_t, s_{t+1}, \log \pi(a_t), V(s_t))
\]

4. Compute GAE.
5. Update PPO for multiple epochs.
6. Save policy checkpoint.
7. Freeze policy for test window.

---

# Walk-Forward Protocol

Training window:

\[
10 \text{ years}
\]

Test window:

\[
1 \text{ year}
\]

Roll forward:

\[
1 \text{ year}
\]

Example:

Train 2010-2019, test 2020.

Train 2011-2020, test 2021.

Train 2012-2021, test 2022.

No learning occurs during test years.

The policy is frozen during test windows.

All preprocessing, HMM fitting, encoder training, and PPO training must use only training-window data.

---

# Implementation Order

Do not start with PPO.

Implement in this order:

1. Data layer
2. Offline feature pipeline
3. JAX trading environment
4. Benchmark strategies
5. Walk-forward splitter
6. HMM regime detector
7. Encoder
8. PPO trainer
9. Full experiment runner

---

# First Implementation Milestone

Build the JAX trading environment.

Required files:

- src/env/accounting.py
- src/env/trading_env.py
- tests/test_accounting.py
- tests/test_trading_env.py

Required functions:

- calculate_turnover
- calculate_transaction_cost
- calculate_portfolio_return
- update_portfolio_value
- calculate_drawdown
- calculate_spy_relative_reward

Acceptance criteria:

- pytest passes
- environment supports 100 stocks + cash
- target weights sum to 1
- turnover is correct
- transaction cost is correct
- portfolio value updates correctly
- drawdown updates correctly
- reward is SPY-relative
- implementation is compatible with JAX jit

---

# Development Rule

Correctness comes before sophistication.

The first working system should be:

Data
→ Environment
→ Equal Weight Backtest
→ SPY-relative performance

Only then add:

- HMM
- Encoder
- PPO