# AI-Native Quantitative Trading Framework

An open-source quantitative trading framework combining:

* JAX
* Reinforcement Learning
* Regime Detection
* Hawkes Processes
* Cross-Asset Attention
* Spectral Market Features

## Goals

Build a research-grade portfolio allocation system capable of:

* Learning market representations
* Detecting market regimes
* Allocating capital dynamically
* Generating alpha relative to SPY

## Execution Targets

### Local Development

Local development targets a MacBook CPU environment.

Use local runs for:

* Small toy datasets
* Unit tests
* Trading environment debugging
* Accounting validation
* Benchmark validation

Do not assume a local GPU is available. All tests must pass on CPU-only JAX and should use small deterministic arrays.

### Colab Training

Google Colab is the target for large training and experiment runs.

Use Colab for:

* Full 100-stock universe runs
* Market encoder training
* PPO training
* Full walk-forward experiments

GPU-only work must stay isolated to experiment scripts or notebooks. Core package modules and tests must remain CPU-compatible.

## Core Components

### Feature Engine

Generates:

* Technical indicators
* Liquidity features
* Volume features
* Hawkes-process features
* Spectral market features
* Macro features

### Market Encoder

Transforms high-dimensional market observations into:

φ_t ∈ R^32

using:

* LSTM
* Cross-Asset Attention
* Attention Pooling

### Regime Detection

Gaussian Hidden Markov Model

Outputs:

p(k_t)

for regime-aware decision making.

### PPO Portfolio Manager

Produces long-only target allocations across:

* 100 stocks
* Cash

using temperature-softmax portfolio construction.

### Trading Environment

Features:

* Weekly rebalancing
* Transaction costs
* Cash accounting
* SPY-relative rewards
* Drawdown penalties

### Walk-Forward Evaluation

* 10-year training window
* 1-year test window
* Annual retraining
* Strict out-of-sample evaluation

## TensorBoard Logging

PPO training can write TensorBoard event files under `runs/`:

```python
from dataclasses import replace

from finrl.experiments import ExperimentConfig
from finrl.ppo import PPOConfig

config = ExperimentConfig(
    ppo=replace(
        PPOConfig(),
        enable_tensorboard=True,
        log_dir="runs",
        log_frequency=1,
    )
)
```

Launch TensorBoard from the repository root:

```bash
tensorboard --logdir runs
```

Example dashboard structure:

```text
runs/
└── split_0/
    ├── ppo/
    │   ├── policy_loss
    │   ├── value_loss
    │   ├── total_loss
    │   ├── entropy
    │   ├── approx_kl
    │   ├── clip_fraction
    │   ├── grad_norm
    │   └── explained_variance
    ├── portfolio/
    │   ├── reward
    │   ├── alpha_vs_spy
    │   ├── turnover
    │   ├── transaction_cost
    │   ├── max_drawdown
    │   ├── cash_weight
    │   ├── max_position_weight
    │   └── effective_positions
    ├── regime/
    │   ├── state_0_probability
    │   ├── state_1_probability
    │   └── state_0_asset_0_allocation
    └── evaluation/
        ├── train_return
        ├── test_return
        ├── train_alpha
        ├── test_alpha
        ├── train_max_drawdown
        └── test_max_drawdown
```

The TensorBoard Scalars view should show PPO update metrics at every configured
`log_frequency`. Regime allocation tags expand by HMM state and asset index, so a
100-stock-plus-cash run will include one allocation scalar per state per asset.

## Repository Structure

src/
├── data/
├── features/
├── env/
├── regimes/
├── models/
├── ppo/
├── backtest/
└── experiments/

## Status

Current phase:

Environment and infrastructure implementation.

PPO training begins only after environment validation and benchmark verification.
