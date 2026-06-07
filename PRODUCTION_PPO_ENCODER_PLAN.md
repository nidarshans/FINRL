# Production PPO and Encoder Implementation Plan

This plan is separate from `IMPLEMENTATION_PLAN.md`. The current encoder and PPO modules are useful smoke-test scaffolding, but production training should move to a Flax/JAX stack with Optax-style optimization, rigorous rollout handling, and experiment-grade logging.

Do not change `ARCHITECTURE.MD` without explicit approval. The production implementation must preserve the existing research architecture:

- Market encoder emits `phi_t in R^32`.
- HMM consumes `phi_t` and emits filtering-only regime probabilities.
- PPO state is `(phi_t, p(k_t), portfolio_context)`.
- PPO action is a long-only target allocation over `N + 1` assets including cash.
- Walk-forward evaluation fits on train only and freezes all artifacts during test.

## Current Implementation

The repository currently has a functional end-to-end scaffold, but the encoder and PPO pieces are not yet production-grade research models.

### Already Implemented

#### Data, Features, and Splits

- yfinance/Polars data ingestion.
- Configurable `N`-stock universe.
- Weekly rebalance calendar.
- Open-to-open holding-period returns.
- Asset, macro, Hawkes, relative, and spectral feature scaffolding.
- Chronological rolling preprocessing.
- Walk-forward train/test splitter.
- No-look-ahead tests for preprocessing and splits.

#### Trading Environment

- JAX-compatible weekly environment step.
- Long-only weight normalization.
- Turnover calculation.
- Transaction cost calculation.
- Gross and net portfolio return.
- Portfolio value update.
- Running peak and drawdown.
- SPY-relative reward.
- Environment scan.
- Accounting and invariant tests.

#### HMM Regime Detector

- `hmmlearn` Gaussian HMM fitting on train-window `phi_t`.
- Project-owned filtering-only probability path.
- Explicit tests that filtering probabilities match `hmmlearn.predict_proba(prefix)[-1]`.
- No smoothing probabilities in evaluation.

#### Current Encoder Scaffold

Files:

- `src/finrl/models/encoder.py`
- `src/finrl/models/attention.py`
- `src/finrl/models/windows.py`

Implemented:

- Pure JAX hand-written LSTM-like encoder.
- Shared asset recurrent parameters.
- Single-head cross-asset attention.
- Attention pooling.
- Macro recurrent encoder.
- Fusion MLP.
- `phi_t in R^32`.
- JIT, `vmap`, `scan`, shape, and no-look-ahead window tests.

Current limitations:

- Not Flax.
- No Optax training state.
- No production encoder training objective.
- No multi-task or self-supervised pretraining.
- No dropout, checkpointed training loop, or experiment-grade metrics.
- The current encoder is suitable for pipeline smoke tests, not final research training.

#### Current PPO Scaffold

Files:

- `src/finrl/ppo/policy.py`
- `src/finrl/ppo/value.py`
- `src/finrl/ppo/distributions.py`
- `src/finrl/ppo/gae.py`
- `src/finrl/ppo/losses.py`
- `src/finrl/ppo/trainer.py`

Implemented:

- Pure JAX actor and critic MLPs.
- PPO state construction:

  \[
  s_t = [\phi_t, p(k_t), w_t, DD_t, TO_{t-1}]
  \]

- Dirichlet simplex action distribution for long-only portfolio weights.
- Exact Dirichlet log-probability and entropy.
- GAE.
- PPO clipped loss.
- Value loss.
- Single-trajectory rollout through the tested JAX environment.
- Simple SGD update.
- Frozen-policy evaluation.
- Checkpoint save/load.
- PPO no-look-ahead tests.

Current limitations:

- Not Flax.
- Not Optax.
- No minibatching.
- No proper rollout buffer.
- No multiple PPO epochs over shuffled minibatches.
- No gradient clipping.
- No KL early stopping.
- No explained variance, clip fraction, or production diagnostics.
- No deterministic mean-action evaluation mode formalized as a separate API.
- No full Colab-grade training loop.
- The current PPO is suitable for smoke-mode correctness and integration tests, not final research training.

#### Current Walk-Forward Runner and Notebook

Files:

- `src/finrl/experiments/run_walk_forward.py`
- `src/finrl/experiments/reporting.py`
- `notebooks/colab_walk_forward.ipynb`

Implemented:

- Strict walk-forward orchestration.
- Train-only preprocessing, HMM, and optional PPO.
- Frozen test evaluation.
- SPY/S&P 500 benchmark comparison.
- Plotly performance chart.
- Plotly spectral feature evolution chart.
- Real yfinance notebook path.
- Synthetic smoke-test path.

Current limitations:

- Production Flax encoder/PPO path is not yet integrated.
- Remaining Phase 4 benchmarks are still incomplete:
  - equal weight
  - cash only
  - momentum top-K
  - risk parity
- Current notebook can run real data, but model quality depends on replacing the scaffold encoder/PPO with the production implementation in this plan.

### What We Will Implement Next

The future production implementation will replace the scaffold encoder/PPO with:

- Flax `nn.Module` encoder architecture matching `ARCHITECTURE.MD`.
- Explicit encoder pretraining objective.
- Flax actor and critic networks.
- Optax optimizer stack.
- Proper rollout buffer.
- Minibatched PPO updates.
- Multiple PPO epochs.
- Gradient clipping.
- KL monitoring and early stopping.
- Deterministic frozen evaluation mode.
- TensorBoard or structured logging.
- Full walk-forward integration behind explicit production config flags.
- Benchmark gate before research claims.

## Production Architecture Summary

The production stack will build two trainable model families:

1. A **Flax market encoder**:

   \[
   f_\theta:
   \left(
   X^{asset}_{t-L+1:t},
   X^{macro}_{t-L+1:t},
   X^{spectral}_t
   \right)
   \mapsto
   \phi_t \in \mathbb{R}^{32}
   \]

2. A **Flax PPO actor-critic**:

   \[
   \pi_\psi(a_t \mid s_t),
   \quad
   V_\omega(s_t)
   \]

   where:

   \[
   s_t =
   [
   \phi_t,
   p(k_t),
   w_t,
   DD_t,
   TO_{t-1}
   ]
   \]

   and:

   \[
   a_t = w_t^{target} \in \Delta^{N}
   \]

   with:

   \[
   \Delta^{N}
   =
   \left\{
   w \in \mathbb{R}^{N+1}
   :
   \sum_{i=1}^{N+1} w_i = 1,
   \quad
   w_i \ge 0
   \right\}
   \]

The production code should treat the current hand-written JAX PPO/encoder as a smoke-test baseline, not the final research model.

## Encoder Model Specification

### Inputs

For each decision date \(t\):

Asset window:

\[
X^{asset}_t
\in
\mathbb{R}^{L \times N \times F_a}
\]

Macro window:

\[
X^{macro}_t
\in
\mathbb{R}^{L \times F_m}
\]

Spectral state:

\[
X^{spectral}_t
\in
\mathbb{R}^{20}
\]

Default:

\[
L = 60
\]

The full encoder output is:

\[
\phi_t \in \mathbb{R}^{32}
\]

### Shared Asset LSTM

For each asset \(i\), use the same recurrent parameters \(\theta_a\):

\[
h_{i,t}^{asset}
=
\operatorname{LSTM}_{\theta_a}
\left(
X^{asset}_{t-L+1:t,i,:}
\right)
\]

where:

\[
h_{i,t}^{asset} \in \mathbb{R}^{64}
\]

Stacking assets:

\[
H_t
=
\begin{bmatrix}
h_{1,t}^{asset} \\
\vdots \\
h_{N,t}^{asset}
\end{bmatrix}
\in
\mathbb{R}^{N \times 64}
\]

Implementation requirement:

- One `nn.OptimizedLSTMCell` or equivalent recurrent cell is parameter-shared across assets.
- The asset dimension is handled by `jax.vmap`, not by constructing `N` separate LSTMs.

### Cross-Asset Self-Attention

Use multi-head self-attention over the asset dimension:

\[
Q = H_t W_Q,
\quad
K = H_t W_K,
\quad
V = H_t W_V
\]

For each head \(h\):

\[
A_h
=
\operatorname{softmax}
\left(
\frac{Q_h K_h^\top}{\sqrt{d_h}}
\right)
\]

\[
\tilde{H}_{t,h}
=
A_h V_h
\]

Concatenate heads and project:

\[
\tilde{H}_t
=
\operatorname{Concat}
(\tilde{H}_{t,1}, \ldots, \tilde{H}_{t,H})
W_O
\]

Shape:

\[
\tilde{H}_t \in \mathbb{R}^{N \times 64}
\]

Default production hyperparameters:

- hidden dimension: 64
- attention heads: 4
- head dimension: 16
- dropout: configurable, default 0.0 for deterministic evaluation

### Attention Pooling

Learn a global query vector:

\[
q_p \in \mathbb{R}^{64}
\]

Pooling logits:

\[
\ell_i = q_p^\top \tanh(W_p \tilde{h}_{i,t} + b_p)
\]

Asset attention weights:

\[
\alpha_i
=
\frac{\exp(\ell_i)}
{\sum_{j=1}^{N} \exp(\ell_j)}
\]

Pooled asset embedding:

\[
e_t^{asset}
=
\sum_{i=1}^{N}
\alpha_i \tilde{h}_{i,t}
\in
\mathbb{R}^{64}
\]

### Macro LSTM

Macro sequence:

\[
X^{macro}_{t-L+1:t}
\in
\mathbb{R}^{L \times F_m}
\]

Macro encoder:

\[
e_t^{macro}
=
\operatorname{LSTM}_{\theta_m}
\left(
X^{macro}_{t-L+1:t}
\right)
\in
\mathbb{R}^{16}
\]

### Fusion MLP

Concatenate:

\[
z_t =
\left[
e_t^{asset},
e_t^{macro},
X_t^{spectral}
\right]
\in
\mathbb{R}^{64 + 16 + 20}
=
\mathbb{R}^{100}
\]

Fusion network:

\[
u_t
=
\operatorname{LayerNorm}
\left(
\operatorname{GELU}
(z_t W_1 + b_1)
\right)
\in
\mathbb{R}^{64}
\]

\[
\phi_t
=
u_t W_2 + b_2
\in
\mathbb{R}^{32}
\]

Optional final normalization:

\[
\phi_t
\leftarrow
\operatorname{LayerNorm}(\phi_t)
\]

This should be configurable because HMM fitting may behave differently with normalized versus unnormalized embeddings.

## Encoder Training Objective Specification

The encoder objective must be explicit. The default production proposal is a **multi-task self-supervised prediction objective** trained only inside each walk-forward train split.

### Prediction Targets

At decision date \(t\), predict train-window labels available after the holding period:

1. Equal-weight next holding-period market return:

\[
y^{mkt}_{t+1}
=
\frac{1}{N}
\sum_{i=1}^{N}
r_{i,t+1}
\]

2. Cross-sectional normalized next returns:

\[
\tilde{r}_{i,t+1}
=
\frac{
r_{i,t+1} - \mu_{t+1}
}{
\sigma_{t+1} + \epsilon
}
\]

where:

\[
\mu_{t+1}
=
\frac{1}{N}
\sum_i r_{i,t+1}
\]

3. Optional volatility target:

\[
y^{vol}_{t+1}
=
\sqrt{
\frac{1}{N}
\sum_i
(r_{i,t+1} - \mu_{t+1})^2
}
\]

### Encoder Heads

Use small heads on top of \(\phi_t\):

\[
\hat{y}^{mkt}_{t+1}
=
g_m(\phi_t)
\]

\[
\hat{y}^{vol}_{t+1}
=
g_v(\phi_t)
\]

\[
\hat{\tilde{r}}_{t+1}
=
g_x(\phi_t, H_t)
\in
\mathbb{R}^{N}
\]

The cross-sectional head may combine the global market state \(\phi_t\) with per-asset embeddings \(\tilde{H}_t\):

\[
\hat{\tilde{r}}_{i,t+1}
=
g_x([\tilde{h}_{i,t}, \phi_t])
\]

### Encoder Loss

Default:

\[
\mathcal{L}_{encoder}
=
\lambda_m
\operatorname{Huber}
(\hat{y}^{mkt}_{t+1}, y^{mkt}_{t+1})
+
\lambda_v
\operatorname{Huber}
(\hat{y}^{vol}_{t+1}, y^{vol}_{t+1})
+
\lambda_x
\frac{1}{N}
\sum_{i=1}^{N}
\operatorname{Huber}
(\hat{\tilde{r}}_{i,t+1}, \tilde{r}_{i,t+1})
+
\lambda_{L2}
\|\theta\|_2^2
\]

Default weights:

- \(\lambda_m = 1.0\)
- \(\lambda_v = 0.25\)
- \(\lambda_x = 0.5\)
- \(\lambda_{L2} = 10^{-5}\)

Important no-look-ahead rule:

- Labels may use \(t+1\) only when both \(t\) and \(t+1\) are inside the train split.
- Test labels must never be used to train encoder parameters.
- During test, encoder parameters are frozen and only features through date \(t\) are used to compute \(\phi_t\).

## PPO Model Specification

### PPO State

Let:

\[
K = \text{number of HMM regimes}
\]

\[
w_t \in \mathbb{R}^{N+1}
\]

Then:

\[
s_t
=
[
\phi_t,
p(k_t),
w_t,
DD_t,
TO_{t-1}
]
\]

Shape:

\[
\dim(s_t)
=
32 + K + (N+1) + 1 + 1
=
35 + K + N
\]

For \(N=100\), \(K=4\):

\[
\dim(s_t)=139
\]

The architecture document also mentions 140 in one location; implementation should compute the dimension from components and add a test documenting the exact value.

### Actor Network

Actor:

\[
\pi_\psi(a_t \mid s_t)
\]

MLP trunk:

\[
h_1 = \operatorname{Tanh}(s_t W_1 + b_1)
\quad
h_1 \in \mathbb{R}^{128}
\]

\[
h_2 = \operatorname{Tanh}(h_1 W_2 + b_2)
\quad
h_2 \in \mathbb{R}^{128}
\]

Policy logits:

\[
z_t = h_2 W_3 + b_3
\quad
z_t \in \mathbb{R}^{N+1}
\]

Softmax mean allocation:

\[
\mu_t
=
\operatorname{softmax}
\left(
\frac{z_t}{\tau}
\right)
\in
\Delta^N
\]

### Action Distribution

Use a Dirichlet policy on the simplex:

\[
a_t
\sim
\operatorname{Dirichlet}
(\alpha_t)
\]

Concentration:

\[
\alpha_t
=
\alpha_0 \mu_t + \alpha_{\min}
\]

where:

- \(\alpha_0 > 0\) controls exploration around the mean.
- \(\alpha_{\min} > 0\) prevents invalid concentration values.

Log probability:

\[
\log \pi_\psi(a_t \mid s_t)
=
\log \Gamma
\left(
\sum_i \alpha_{t,i}
\right)
-
\sum_i
\log \Gamma(\alpha_{t,i})
+
\sum_i
(\alpha_{t,i} - 1)
\log a_{t,i}
\]

Entropy:

\[
\mathcal{H}
(\operatorname{Dir}(\alpha))
=
\sum_i \log \Gamma(\alpha_i)
-
\log \Gamma(\alpha_0)
+
(\alpha_0 - d)
\psi(\alpha_0)
-
\sum_i
(\alpha_i - 1)
\psi(\alpha_i)
\]

where:

\[
\alpha_0 = \sum_i \alpha_i
\]

and \(d=N+1\).

Evaluation mode should use:

\[
a_t^{eval} = \mu_t
\]

not a random Dirichlet sample.

### Critic Network

Critic:

\[
V_\omega(s_t)
\]

Architecture:

\[
h_1^V
=
\operatorname{Tanh}(s_t W_1^V + b_1^V)
\in
\mathbb{R}^{128}
\]

\[
h_2^V
=
\operatorname{Tanh}(h_1^V W_2^V + b_2^V)
\in
\mathbb{R}^{64}
\]

\[
\hat{V}_t
=
h_2^V W_3^V + b_3^V
\in
\mathbb{R}
\]

### Environment Transition

The actor action is a target allocation:

\[
a_t = w_t^{target}
\]

The environment computes:

\[
TO_t
=
\sum_i
|w_{i,t}^{target} - w_{i,t}^{current}|
\]

\[
TC_t
=
c \cdot TO_t
\]

\[
R_t^{gross}
=
\sum_i
w_{i,t}^{target}
r_{i,t}
\]

\[
R_t^{net}
=
R_t^{gross} - TC_t
\]

\[
V_{t+1}
=
V_t(1 + R_t^{net})
\]

Reward:

\[
reward_t
=
\log(1 + R_t^{net})
-
\log(1 + R_t^{SPY})
-
\lambda_{DD}
\max(0, DD_t - DD_{max})
-
\lambda_{TO}TO_t
\]

PPO must call the existing environment accounting path. It must not duplicate these formulas in the trainer.

## PPO Optimization Math

### Advantage Estimation

TD residual:

\[
\delta_t
=
r_t
+
\gamma(1-d_t)V(s_{t+1})
-
V(s_t)
\]

GAE:

\[
\hat{A}_t
=
\sum_{\ell=0}^{T-t-1}
(\gamma\lambda)^\ell
\left(
\prod_{j=0}^{\ell}
(1-d_{t+j})
\right)
\delta_{t+\ell}
\]

Return target:

\[
\hat{R}_t
=
\hat{A}_t + V(s_t)
\]

Normalize advantages within the train rollout:

\[
\bar{A}_t
=
\frac{
\hat{A}_t - \mu_A
}{
\sigma_A + \epsilon
}
\]

### PPO Clipped Objective

Probability ratio:

\[
\rho_t(\psi)
=
\frac{
\pi_\psi(a_t \mid s_t)
}{
\pi_{\psi_{old}}(a_t \mid s_t)
}
=
\exp
\left(
\log\pi_\psi(a_t \mid s_t)
-
\log\pi_{\psi_{old}}(a_t \mid s_t)
\right)
\]

Actor objective:

\[
\mathcal{L}^{CLIP}_t(\psi)
=
\min
\left(
\rho_t(\psi)\bar{A}_t,
\operatorname{clip}
(\rho_t(\psi), 1-\epsilon, 1+\epsilon)
\bar{A}_t
\right)
\]

Actor loss to minimize:

\[
\mathcal{L}_{actor}
=
-
\frac{1}{T}
\sum_t
\mathcal{L}^{CLIP}_t(\psi)
\]

Critic loss:

\[
\mathcal{L}_{critic}
=
\frac{1}{T}
\sum_t
\left(
V_\omega(s_t) - \hat{R}_t
\right)^2
\]

Entropy regularization:

\[
\mathcal{L}_{entropy}
=
-
\frac{1}{T}
\sum_t
\mathcal{H}(\pi_\psi(\cdot \mid s_t))
\]

Total loss:

\[
\mathcal{L}_{PPO}
=
\mathcal{L}_{actor}
+
c_v
\mathcal{L}_{critic}
+
c_e
\mathcal{L}_{entropy}
\]

where entropy coefficient \(c_e\) is typically negative if written as a bonus, or the sign is handled explicitly in code:

\[
\mathcal{L}
=
\mathcal{L}_{actor}
+
c_v\mathcal{L}_{critic}
-
c_{ent}\mathcal{H}
\]

### Production PPO Hyperparameters

Initial defaults:

- \(\gamma = 0.99\)
- \(\lambda_{GAE} = 0.95\)
- clip epsilon \(= 0.2\)
- value coefficient \(= 0.5\)
- entropy coefficient \(= 0.001\)
- max gradient norm \(= 0.5\)
- learning rate \(= 3 \times 10^{-4}\)
- PPO epochs per rollout \(= 5\)
- minibatch size \(= 32\), configurable
- Dirichlet concentration scale \(\alpha_0 = 50\), configurable
- softmax temperature \(\tau = 1.0\), configurable

### KL Monitoring

Approximate KL:

\[
\widehat{KL}
=
\frac{1}{T}
\sum_t
\left(
\log\pi_{old}(a_t \mid s_t)
-
\log\pi_{new}(a_t \mid s_t)
\right)
\]

Clip fraction:

\[
f_{clip}
=
\frac{1}{T}
\sum_t
\mathbf{1}
\left[
|\rho_t - 1| > \epsilon
\right]
\]

Use early stopping if:

\[
\widehat{KL} > KL_{target}
\]

Default:

\[
KL_{target}=0.03
\]

## Phase A: Dependencies and Boundaries

### Objective

Promote the production model stack to Flax/JAX while keeping the existing smoke-test modules available until the replacement is fully validated.

### Files to Create or Modify

- `pyproject.toml`
- `src/finrl/models/flax_encoder.py`
- `src/finrl/ppo/flax_policy.py`
- `src/finrl/ppo/flax_value.py`
- `src/finrl/ppo/opt_state.py`
- `tests/test_flax_imports.py`

### Required Functions or Classes

- `ProductionEncoderConfig`
- `ProductionPPOConfig`
- `TrainState` or project wrapper around `flax.training.train_state.TrainState`
- explicit module exports that do not break current tests

### Tests to Write

- Flax imports work on CPU-only JAX.
- Production modules initialize on CPU with tiny deterministic arrays.
- Existing smoke-test PPO/encoder tests still pass.

### Acceptance Criteria

- `pytest` passes on CPU-only JAX.
- No local GPU assumption is introduced.
- Existing Phase 10/11 APIs are not silently broken.

### Risks or Failure Modes

- Mixing smoke-test and production APIs creates ambiguous behavior.
- Flax dependency version conflicts with installed JAX.
- Production code accidentally imports Colab-only packages in core modules.

## Phase B: Production Flax Market Encoder

### Objective

Replace the hand-written pure JAX encoder with a Flax module that matches the architecture: shared asset LSTM, cross-asset attention, attention pooling, macro LSTM, and fusion MLP.

### Files to Create or Modify

- `src/finrl/models/flax_encoder.py`
- `src/finrl/models/encoder_state.py`
- `tests/test_flax_encoder_shapes.py`
- `tests/test_no_lookahead_flax_encoder.py`

### Required Functions or Classes

- `AssetLSTMEncoder(nn.Module)`
- `CrossAssetSelfAttention(nn.Module)`
- `AttentionPool(nn.Module)`
- `MacroLSTMEncoder(nn.Module)`
- `MarketEncoderFlax(nn.Module)`
- `init_encoder_train_state(rng, config)`
- `encode_market_state_flax(params, feature_window, config)`

### Tests to Write

- Asset input `(L, N, F_asset)` produces `(N, 64)`.
- Macro input `(L, F_macro)` produces `(16,)`.
- Spectral input dimension must be `20`.
- Fused market state has shape `(32,)`.
- Encoder works under `jax.jit`.
- Encoder works under `jax.vmap` over windows.
- Lookback windows include only `t-L+1:t`.
- Same PRNG seed initializes identical parameters.

### Acceptance Criteria

- Encoder produces `phi_t in R^32`.
- Shared asset LSTM uses shared parameters across assets, not one LSTM per ticker.
- No future observations enter a feature window.
- All tests use small arrays and pass on CPU.

### Risks or Failure Modes

- Flax recurrent API shape conventions are misused.
- Per-asset parameter sharing is accidentally broken.
- Attention pooling collapses the wrong axis.
- Spectral features are accidentally treated as a sequence instead of current-date features.

## Phase C: Encoder Training Objective

### Objective

Define and implement a real training objective for the market encoder. This must be approved before implementation if the objective changes the research architecture.

### Candidate Objectives

- Self-supervised next-period market return prediction.
- Cross-sectional return prediction.
- Contrastive regime-aware representation learning.
- Auxiliary reconstruction/prediction of spectral or macro state.

### Files to Create or Modify

- `src/finrl/models/encoder_training.py`
- `src/finrl/models/encoder_losses.py`
- `src/finrl/models/checkpoints.py`
- `tests/test_encoder_training.py`
- `tests/test_no_lookahead_encoder_training.py`

### Required Functions or Classes

- `EncoderTrainingConfig`
- `EncoderBatch`
- `make_encoder_batches(...)`
- `encoder_loss(...)`
- `train_encoder_epoch(...)`
- `fit_encoder_on_train_split(...)`
- `save_encoder_checkpoint(...)`
- `load_encoder_checkpoint(...)`

### Tests to Write

- Batches are generated from train split only.
- Targets use future returns only when they are train-window labels.
- Test split never influences encoder training.
- One training step changes parameters.
- Loss is finite on deterministic synthetic data.
- Checkpoint round-trip preserves parameters.

### Acceptance Criteria

- Training objective is explicitly documented.
- Encoder training is train-window-only.
- Full training remains isolated to Colab scripts/notebooks.
- Local tests use tiny deterministic arrays.

### Risks or Failure Modes

- Objective introduces look-ahead by using future test labels.
- Encoder overfits tiny train windows.
- Unclear objective causes misleading downstream PPO behavior.

## Phase D: Production PPO Actor-Critic Modules

### Objective

Implement Flax actor and critic networks that match `ARCHITECTURE.MD`, while supporting a valid portfolio action distribution on the simplex.

### Files to Create or Modify

- `src/finrl/ppo/flax_policy.py`
- `src/finrl/ppo/flax_value.py`
- `src/finrl/ppo/simplex_distribution.py`
- `tests/test_flax_policy.py`
- `tests/test_simplex_distribution.py`

### Required Functions or Classes

- `PortfolioActorFlax(nn.Module)`
- `PortfolioCriticFlax(nn.Module)`
- `DirichletPortfolioDistribution`
- `build_ppo_state(...)`
- `sample_action(...)`
- `action_log_prob(...)`
- `policy_entropy(...)`

### Tests to Write

- Actor input dimension is `32 + K + N + 1 + 1`.
- Actor outputs `N + 1` allocation logits.
- Critic outputs scalar value.
- Sampled action is long-only and sums to `1`.
- Dirichlet log-probability matches SciPy reference on small arrays.
- Deterministic evaluation mode uses mean allocation.
- Sampling is reproducible for fixed PRNG keys.

### Acceptance Criteria

- Actor and critic initialize and run under `jax.jit`.
- No invalid weights can reach the environment.
- Evaluation can be deterministic and frozen.

### Risks or Failure Modes

- Dirichlet concentration becomes numerically unstable.
- Deterministic target weights conflict with PPO log-probability requirements.
- Action distribution entropy is miscomputed.

## Phase E: Production Rollout Buffer

### Objective

Create a rollout buffer for historical single-trajectory PPO that stores all quantities needed for minibatched PPO updates without recomputing environment accounting incorrectly.

### Files to Create or Modify

- `src/finrl/ppo/rollout.py`
- `src/finrl/ppo/batches.py`
- `tests/test_ppo_rollout_buffer.py`
- `tests/test_no_lookahead_ppo_rollout.py`

### Required Functions or Classes

- `RolloutBatch`
- `RolloutBuffer`
- `collect_rollout(...)`
- `make_minibatches(...)`
- `shuffle_rollout_indices(...)`

### Tests to Write

- Rollout calls the existing `environment_step`.
- Rewards match environment accounting.
- Stored log-probs match the policy used during rollout.
- Minibatches preserve state/action/reward alignment.
- No test observations are used in train rollout collection.

### Acceptance Criteria

- Rollout collection works under `jax.lax.scan`.
- Buffer supports deterministic shuffling with PRNG keys.
- All rollout arrays have consistent leading time dimension.

### Risks or Failure Modes

- Stored old log-probs are recomputed after policy update.
- Minibatch shuffling breaks temporal alignment.
- Environment state leaks between splits.

## Phase F: Production PPO Optimization Loop

### Objective

Implement a mature PPO update loop with Optax, minibatches, multiple epochs, gradient clipping, advantage normalization, KL monitoring, and deterministic frozen evaluation.

### Files to Create or Modify

- `src/finrl/ppo/flax_trainer.py`
- `src/finrl/ppo/losses.py`
- `src/finrl/ppo/metrics.py`
- `src/finrl/ppo/checkpoints.py`
- `tests/test_flax_ppo_trainer.py`
- `tests/test_no_lookahead_flax_ppo.py`

### Required Functions or Classes

- `ProductionPPOTrainState`
- `PPOTrainMetrics`
- `compute_gae(...)`
- `ppo_actor_loss(...)`
- `critic_loss(...)`
- `ppo_total_loss(...)`
- `update_minibatch(...)`
- `train_epoch(...)`
- `train_ppo_on_split(...)`
- `evaluate_frozen_policy(...)`

### Tests to Write

- GAE matches hand-computed fixtures.
- One minibatch update changes actor and critic parameters.
- Value loss decreases on a trivial supervised fixture.
- PPO losses are finite for synthetic trajectories.
- Evaluation does not update parameters or optimizer state.
- Checkpoint save/load preserves params and optimizer state.
- Train artifacts record split train window only.

### Acceptance Criteria

- PPO supports minibatch updates.
- PPO supports multiple epochs per rollout.
- Gradient clipping is applied.
- Approximate KL, entropy, value loss, actor loss, and explained variance are reported.
- Frozen evaluation is deterministic for fixed seed/config.

### Risks or Failure Modes

- PPO updates overfit a single historical trajectory.
- Minibatch updates use stale or mismatched advantages.
- Optimizer state is accidentally updated in evaluation.
- KL blows up without early stopping.

## Phase G: Walk-Forward Integration

### Objective

Replace the current smoke PPO/encoder path inside the walk-forward runner with production Flax artifacts while preserving strict train/test boundaries.

### Files to Create or Modify

- `src/finrl/experiments/run_walk_forward.py`
- `src/finrl/experiments/artifacts.py`
- `src/finrl/experiments/reporting.py`
- `notebooks/colab_walk_forward.ipynb`
- `tests/test_production_walk_forward.py`
- `tests/test_no_lookahead_production_experiment.py`

### Required Functions or Classes

- `fit_encoder_train_artifacts(...)`
- `fit_hmm_train_artifacts(...)`
- `fit_ppo_train_artifacts(...)`
- `evaluate_frozen_production_policy(...)`
- `ProductionExperimentArtifacts`

### Tests to Write

- Each split has independent train-fitted encoder, HMM, and PPO artifacts.
- Test split uses frozen artifacts only.
- Result dates align with exact holding-period returns.
- SPY benchmark uses identical decision dates.
- Re-running with fixed seed returns identical local synthetic results.

### Acceptance Criteria

- Full synthetic walk-forward run passes locally on CPU.
- Colab notebook can run real yfinance data path.
- Reports include portfolio vs SPY, drawdown, turnover, costs, and spectral evolution.
- Production results remain blocked from research claims until Phase 4 benchmark suite is complete.

### Risks or Failure Modes

- Artifact reuse across splits leaks train/test information.
- Encoder checkpoint from one split is accidentally reused in another.
- Runner becomes too monolithic to debug.

## Phase H: Logging, Monitoring, and Diagnostics

### Objective

Add experiment-grade logging and diagnostics for production PPO and encoder training.

### Files to Create or Modify

- `src/finrl/logging/tensorboard.py`
- `src/finrl/ppo/metrics.py`
- `src/finrl/models/encoder_metrics.py`
- `tests/test_training_metrics.py`

### Required Metrics

- Actor loss
- Critic loss
- Entropy
- Approximate KL
- Clip fraction
- Explained variance
- Mean turnover
- Mean transaction cost
- Mean drawdown
- Portfolio return
- SPY return
- SPY-relative alpha
- Encoder loss
- Gradient norm

### Tests to Write

- Metrics are finite.
- Metrics shapes are scalar or documented.
- TensorBoard logging can be disabled in local tests.

### Acceptance Criteria

- Colab runs produce per-split logs.
- Logs are optional in local tests.
- No logging call mutates training semantics.

### Risks or Failure Modes

- Logging slows down training.
- Metrics are computed from test data during training.
- Run artifacts pollute the repository.

## Phase I: Benchmark Gate Before Research Claims

### Objective

Ensure production PPO is not treated as valid until Phase 4 benchmark strategies are implemented and pass.

### Required Benchmarks

- SPY
- Equal weight
- Cash only
- Momentum top-K
- Risk parity

### Tests to Write

- PPO and benchmarks use identical holding-period returns.
- Benchmark policies are deterministic.
- Momentum and risk parity use past information only.
- Final report contains benchmark comparisons.

### Acceptance Criteria

- Any production experiment report includes benchmark table.
- PPO-only charts are labeled preliminary until benchmark gate passes.

### Risks or Failure Modes

- PPO performance is interpreted without baseline comparisons.
- Benchmark date alignment differs from PPO evaluation dates.

## Implementation Order

1. Add Flax/Optax dependency boundary and import tests.
2. Implement production Flax encoder forward pass.
3. Decide and implement encoder training objective.
4. Implement Flax actor/critic and simplex distribution.
5. Implement rollout buffer and minibatching.
6. Implement Optax PPO update loop.
7. Integrate production artifacts into walk-forward runner behind config flags.
8. Add logging and diagnostics.
9. Re-run Colab real-data notebook with production path.
10. Complete Phase 4 benchmark gate before drawing research conclusions.

## Non-Negotiable Constraints

- Do not fit encoder, HMM, PPO, or preprocessing on full data.
- Do not update policy during test evaluation.
- Do not use HMM smoothing probabilities.
- Do not require GPU for unit tests.
- Do not hard-code 100 stocks; support configurable `N`.
- Do not change `ARCHITECTURE.MD` without explicit approval.
- Do not claim production research validity before benchmark strategies pass.
