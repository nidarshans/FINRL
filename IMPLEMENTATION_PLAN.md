# Implementation Plan

This plan follows `ARCHITECTURE.MD`, `PROJECT_CONTEXT.md`, `TESTING.md`, `README.md`, and `AGENTS.md`. The architecture remains the source of truth; if implementation conflicts with it, update code rather than architecture. Do not begin production PPO training or trust PPO evaluation results until the JAX environment, accounting tests, and benchmark backtests are correct.

The numbered phase sections below preserve the requested planning breakdown. Implementation order is now: Phase 1, Phase 2, Phase 3, Phase 5, Phase 6, Phase 7, Phase 8, Phase 9, Phase 10, Phase 11, Phase 12, and Phase 4 last. Early environment phases must use small synthetic fixtures only; integration with real market data begins after the data ingestion and feature engineering phases.

## Execution Targets

### Local Development

- Target: MacBook CPU.
- Use small toy datasets and deterministic arrays.
- Run unit tests, environment debugging, accounting validation, and benchmark validation locally.
- Do not assume a local GPU is available.
- All tests must pass on CPU-only JAX.

### Colab Training

- Target: Google Colab with a single GPU when available.
- Use Colab for the full configured `N`-stock universe, encoder training, PPO training, and full walk-forward experiments.
- Keep GPU-only assumptions isolated to experiment scripts or notebooks.
- Core package modules and tests must remain CPU-compatible.

## Global Constraints

- Keep core trading environment logic deterministic, functional, and JAX-compatible.
- Prefer pure functions, typed dataclasses, focused modules, and clear docstrings.
- Use `jax.numpy` in environment/accounting/reward logic.
- Support `jax.jit`, `jax.lax.scan`, float32 inputs, and deterministic outputs for environment functions.
- Tests must use small deterministic arrays and pass on CPU-only JAX.
- Use chronological rolling preprocessing only; do not fit full-window scalers.
- Avoid look-ahead bias in every split, feature, benchmark, HMM, encoder, and PPO path.
- Keep the v1 training loop Colab-friendly: single historical trajectory, modest memory use, minimal dependencies.
- Keep full-universe training and GPU-only work out of unit tests and isolated to experiment scripts or notebooks.
- Treat the number of stocks as configurable `N`; do not hard-code 100 stocks except as an optional default.
- Use filtering-only HMM probabilities: `P(k_t | x_1:t)`. Never use forward-backward smoothing for evaluation features.
- Follow `TESTING.md` strictly. Financial correctness gates model sophistication.

## Phase 1: Repository Setup

### Objective

Create a minimal, testable Python package structure that supports JAX, pytest, type hints, and future Colab use without adding model complexity.

### Files to Create or Modify

- `pyproject.toml`
- `src/finrl/__init__.py`
- `src/finrl/config.py`
- `src/finrl/types.py`
- `src/finrl/data/__init__.py`
- `src/finrl/features/__init__.py`
- `src/finrl/env/__init__.py`
- `src/finrl/backtest/__init__.py`
- `src/finrl/regimes/__init__.py`
- `src/finrl/models/__init__.py`
- `src/finrl/ppo/__init__.py`
- `src/finrl/experiments/__init__.py`
- `tests/conftest.py`
- `.gitignore`
- `README.md`

### Required Functions or Classes

- `ProjectConfig` dataclass for global defaults:
  - `num_stocks: int = 100`
  - `include_cash: bool = True`
  - `rebalance_frequency: str = "W-FRI-to-MON"`
  - `transaction_cost_bps: float = 10.0`
  - `lookback_days: int = 60`
  - `hmm_states: int = 4`
  - `train_years: int = 10`
  - `test_years: int = 1`
- Shared type aliases for arrays, dates, tickers, and path-like inputs.

### Tests to Write

- `tests/test_config.py`
  - default asset dimension is 101 when cash is included
  - transaction cost bps converts consistently to decimal cost
  - configuration values are immutable or treated as value objects

### Acceptance Criteria

- `pytest` discovers tests.
- Package imports from `src/finrl`.
- No trading logic is implemented yet.
- Dependency set is minimal and compatible with Colab.
- Tests run on CPU-only JAX without requiring GPU availability.
- `.gitignore` does not ignore source code or tests that must be committed.

### Dependencies on Prior Phases

- None.

### Risks or Failure Modes

- Adding too many dependencies before they are needed.
- Naming the package ambiguously relative to the existing `FINRL` repository folder.
- Accidentally ignoring `tests/` or important markdown files in `.gitignore`.

## Phase 2: JAX-Native Trading Environment

### Objective

Implement the pure JAX accounting and weekly trading step required by the first milestone. The environment should operate on synthetic arrays and not depend on real data ingestion yet.

### Files to Create or Modify

- `src/finrl/env/accounting.py`
- `src/finrl/env/rewards.py`
- `src/finrl/env/trading_env.py`
- `src/finrl/env/schedules.py`
- `src/finrl/types.py`
- `tests/test_accounting.py`
- `tests/test_trading_env.py`

### Required Functions or Classes

- `EnvState` dataclass or chex-compatible dataclass:
  - `weights`
  - `portfolio_value`
  - `peak_value`
  - `drawdown`
  - `previous_turnover`
  - optional step index
- `EnvConfig` dataclass:
  - `transaction_cost_rate`
  - `drawdown_limit`
  - `drawdown_penalty`
  - `turnover_penalty`
  - `cash_index`
- `StepResult` dataclass:
  - `state`
  - `reward`
  - `gross_return`
  - `net_return`
  - `transaction_cost`
  - `turnover`
- `calculate_turnover(current_weights, target_weights)`
- `calculate_transaction_cost(turnover, cost_rate)`
- `calculate_gross_portfolio_return(weights, asset_returns)`
- `calculate_net_portfolio_return(gross_return, transaction_cost)`
- `update_portfolio_value(portfolio_value, net_return)`
- `update_running_peak(previous_peak, portfolio_value)`
- `calculate_drawdown(portfolio_value, peak_value)`
- `calculate_spy_relative_reward(net_return, spy_return, drawdown, turnover, config)`
- `normalize_long_only_weights(raw_weights)` if needed for tests and policy integration
- `environment_step(state, target_weights, asset_returns, spy_return, config)`
- `scan_environment(initial_state, target_weights, returns, spy_returns, config)`

### Tests to Write

- Unit tests for:
  - turnover
  - transaction cost
  - gross portfolio return
  - net portfolio return
  - portfolio value update
  - running peak update
  - drawdown
  - SPY-relative reward
  - cash return contribution
  - weight normalization checks
- Environment step tests that verify all fields from input state through next state.
- Invariant tests:
  - weights sum to 1
  - weights are nonnegative
  - portfolio value remains positive for valid returns
  - drawdown is in `[0, 1]`
  - turnover and transaction costs are nonnegative
  - reward is finite
- JAX compatibility tests:
  - `jax.jit(environment_step)`
  - `jax.lax.scan` over multiple weekly steps
  - float32 inputs and deterministic outputs
- Numerical edge-case tests:
  - zero returns
  - negative returns
  - high turnover
  - all-cash allocation
  - all-stock allocation
  - zero and nonzero transaction cost

### Acceptance Criteria

- `pytest` passes with strict tolerances from `TESTING.md`.
- Tests use small deterministic arrays and pass on CPU-only JAX.
- Environment supports configured `N` stocks plus cash, `N + 1` allocation weights total.
- One step represents Friday decision, Monday-open rebalance, Monday-open to next Monday-open hold.
- Transaction costs are charged from turnover exactly as specified.
- Reward is SPY-relative and pluggable.
- No hidden mutable state or Python-side state changes inside JAX step logic.

### Dependencies on Prior Phases

- Phase 1 package structure and test setup.

### Risks or Failure Modes

- Accidentally mixing pre-rebalance and post-rebalance weights.
- Charging transaction cost in bps instead of decimal rate.
- Allowing invalid target weights to silently pass into accounting.
- Breaking JAX compatibility with regular Python dataclasses or control flow.
- Using Friday close to Friday close returns instead of Monday open to Monday open returns.

## Phase 3: Accounting and Reward Tests

### Objective

Expand financial correctness coverage until accounting and reward behavior are trustworthy enough to support benchmarks and later PPO.

### Files to Create or Modify

- `tests/test_accounting.py`
- `tests/test_rewards.py`
- `tests/test_trading_env.py`
- `tests/test_env_invariants.py`
- `tests/fixtures/accounting_cases.py`
- `src/finrl/env/accounting.py`
- `src/finrl/env/rewards.py`

### Required Functions or Classes

- `RewardConfig` dataclass with:
  - `drawdown_penalty`
  - `drawdown_limit`
  - `turnover_penalty`
  - reward mode identifier if needed
- `spy_relative_reward(...)`
- Optional `RewardFn` protocol or callable type for pluggable rewards.

### Tests to Write

- Table-driven accounting cases with hand-computed expected values.
- Reward cases:
  - portfolio beats SPY
  - portfolio trails SPY
  - SPY up period
  - SPY down period
  - drawdown penalty below threshold is zero
  - drawdown penalty above threshold is positive
  - turnover penalty scales linearly
- Cash accounting cases:
  - all-cash earns only cash return
  - mixed stock/cash uses cash return in portfolio return
- Regression-style tests for exact weekly rebalance logic.

### Acceptance Criteria

- All mandatory tests in `TESTING.md` exist.
- Accounting tests are readable enough to audit by hand.
- Accounting and reward tests use small deterministic arrays and pass on CPU-only JAX.
- Reward function accepts alternate implementations without changing environment step.
- Every core financial calculation has a direct unit test.

### Dependencies on Prior Phases

- Phase 2 environment and accounting functions.

### Risks or Failure Modes

- Tests duplicate implementation formulas without independent expected values.
- Reward tests fail to cover drawdown and turnover penalties together.
- Floating-point tolerances are too loose to catch accounting mistakes.

## Phase 4: Benchmark Strategies (Implement Last)

### Objective

Implement deterministic benchmark strategies and a simple benchmark backtest path after the main data, feature, model, PPO, and experiment scaffolding phases are in place. This phase is deliberately deferred to the end, but production PPO results are not considered valid until this benchmark suite passes.

### Files to Create or Modify

- `src/finrl/backtest/benchmarks.py`
- `src/finrl/backtest/engine.py`
- `src/finrl/backtest/metrics.py`
- `src/finrl/backtest/results.py`
- `tests/test_benchmarks.py`
- `tests/test_backtest_engine.py`
- `tests/test_metrics.py`

### Required Functions or Classes

- `BenchmarkPolicy` protocol:
  - `target_weights(context) -> Array`
- `cash_only_policy(num_assets, cash_index)`
- `equal_weight_policy(num_stocks, cash_index)`
- `spy_benchmark_returns(spy_open_prices)`
- `momentum_top_k_policy(past_returns, k, include_cash)`
- `risk_parity_policy(past_returns, lookback, include_cash)`
- `run_policy_backtest(policy, returns, spy_returns, initial_state, env_config)`
- Metrics:
  - cumulative return
  - annualized return
  - volatility
  - max drawdown
  - turnover summary
  - alpha relative to SPY

### Tests to Write

- Cash-only strategy earns cash return only.
- Equal-weight assigns equal stock weights and expected cash weight.
- SPY benchmark return matches SPY open-to-open return.
- Momentum Top-K uses only historical returns available before the rebalance date.
- Risk parity uses only trailing-window covariance.
- Benchmark backtest uses the same holding period as portfolio returns.
- Weekly rebalance updates weights deterministically.

### Acceptance Criteria

- Equal-weight and cash-only backtests run end-to-end on synthetic data.
- Local benchmark validation uses small deterministic datasets and CPU-only JAX.
- SPY-relative performance is computed over the exact same Monday-open holding windows.
- Benchmark policies are deterministic and reproducible.
- No benchmark uses future returns, full-sample covariance, or full-period ranking.
- Final PPO and walk-forward reports include benchmark comparisons before any research conclusions are drawn.

### Dependencies on Prior Phases

- Phase 2 environment step.
- Phase 3 accounting/reward test coverage.
- Phase 5 data ingestion and calendar.
- Phase 8 walk-forward splitter.
- Phase 12 experiment runner integration points.

### Risks or Failure Modes

- Momentum or risk parity accidentally uses data from the rebalance interval.
- Benchmark and portfolio returns use different calendars.
- Risk parity becomes numerically unstable with singular covariance.
- Benchmark code drifts away from the JAX environment accounting path.

## Phase 5: Data Ingestion

### Objective

Build a reproducible public-market-data layer for a configurable `N`-stock universe, SPY, cash/risk-free data, macro series, and raw OHLCV inputs. Use `yfinance` for ticker OHLCV data and Polars DataFrames for ingestion, validation, alignment, and cached tabular storage.

### Files to Create or Modify

- `src/finrl/data/universe.py`
- `src/finrl/data/sources.py`
- `src/finrl/data/download.py`
- `src/finrl/data/calendar.py`
- `src/finrl/data/schema.py`
- `src/finrl/data/storage.py`
- `src/finrl/data/validation.py`
- `tests/test_data_schema.py`
- `tests/test_calendar.py`
- `tests/test_data_validation.py`

### Required Functions or Classes

- `UniverseConfig`
- `MarketDataConfig`
- `MarketDataBundle`
- `load_universe(path)`
- `validate_universe(tickers, expected_count: int | None = None)`
- `download_ohlcv_yfinance(tickers, start, end, source_config) -> pl.DataFrame`
- `download_ohlcv(tickers, start, end, source_config) -> pl.DataFrame`
- `download_macro_series(start, end, source_config)`
- `align_to_trading_calendar(data, calendar)`
- `build_weekly_rebalance_calendar(daily_prices)`
- `compute_open_to_open_returns(open_prices, rebalance_calendar)`
- `load_or_cache_raw_data(config)`

### Tests to Write

- Universe contains the configured `N` stocks before cash is added.
- `N` may be small for local tests and larger for Colab runs.
- Ticker OHLCV download code uses `yfinance`.
- Ingestion APIs return Polars DataFrames for tabular data.
- SPY is available for benchmark returns.
- Open prices produce correct Monday-open to Monday-open returns.
- Weekly calendar maps Friday decision dates to next Monday execution dates.
- Missing values are detected and reported before feature generation.
- Data cache reads and writes preserve schema.

### Acceptance Criteria

- Raw data can be loaded from cache without network access.
- Data bundle contains aligned stock OHLCV, SPY OHLCV, macro series, and calendar metadata.
- Local ingestion tests use tiny cached fixtures rather than full-universe downloads.
- Polars is used for ingestion and cached tabular data; conversion to NumPy/JAX arrays happens only at explicit model or environment boundaries.
- The universe size is parameterized by `N`; no production code hard-codes 100 stocks.
- No preprocessing or feature fitting occurs in ingestion.
- Date alignment makes look-ahead boundaries explicit.

### Dependencies on Prior Phases

- Phase 1 package structure.
- Phase 2 defines environment return timing requirements that ingestion must satisfy.
- Phase 4 is deferred until last and consumes the aligned returns produced here.

### Risks or Failure Modes

- Public data source changes format or rate limits downloads.
- Survivorship bias in user-selected universe.
- Missing Monday opens due to holidays.
- yfinance data can be adjusted, delayed, missing, or revised; schema validation must make these issues visible.
- Timezone or date-index mismatch between assets and macro data.

## Phase 6: Feature Engineering

### Objective

Generate asset, macro, spectral, and optional Hawkes features using only information available at or before each feature date.

### Files to Create or Modify

- `src/finrl/features/asset.py`
- `src/finrl/features/macro.py`
- `src/finrl/features/spectral.py`
- `src/finrl/features/hawkes.py`
- `src/finrl/features/relative.py`
- `src/finrl/features/pipeline.py`
- `src/finrl/features/schema.py`
- `tests/test_asset_features.py`
- `tests/test_macro_features.py`
- `tests/test_spectral_features.py`
- `tests/test_hawkes_features.py`
- `tests/test_feature_pipeline.py`

### Required Functions or Classes

- `FeatureConfig`
- `FeatureBundle`
- Asset feature functions:
  - `compute_returns`
  - `compute_rsi`
  - `compute_macd`
  - `compute_trend_slope`
  - `compute_amihud_illiquidity`
  - `compute_dollar_volume`
  - `compute_turnover_feature`
  - `compute_volume_momentum`
  - `compute_volume_acceleration`
- Relative features:
  - `cross_sectional_percentile_rank(values_by_date)`
- Spectral features:
  - `compute_volume_eigenspectrum`
  - `compute_liquidity_eigenspectrum`
  - `compute_sector_flow_indicators`
- Hawkes features:
  - `compute_hawkes_features`
  - with TODO or issue if exact model details require user clarification.
- `build_feature_bundle(raw_data, config)`

### Tests to Write

- Each feature uses only trailing windows.
- Cross-sectional ranks are computed per date, not globally.
- Feature tensors have expected shapes:
  - asset: `(T, N, F_asset)`
  - macro: `(T, F_macro)`
  - spectral: `(T, 20)`
- Missing initial lookback periods are handled consistently.
- Hawkes features do not use future events.
- Feature dates align to Friday close decision dates.

### Acceptance Criteria

- Feature pipeline returns arrays and metadata with explicit date indexes.
- Local feature tests run on small deterministic fixtures.
- Features can be computed for train and test periods without fitting on full data.
- Output is suitable for offline preprocessing and later encoder input windows.
- Any underspecified Hawkes detail is documented as TODO rather than invented.

### Dependencies on Prior Phases

- Phase 5 data bundle and calendar.

### Risks or Failure Modes

- Look-ahead leakage from centered rolling windows.
- Cross-sectional ranks accidentally fit globally.
- Spectral features use future covariance windows.
- Hawkes implementation becomes too slow for Colab.

## Phase 7: Rolling Preprocessing Pipeline

### Objective

Implement offline preprocessing that preserves strict chronological realism during training and evaluation. Do not fit full train-window scalers. Standardize features with rolling statistics computed per ticker for asset features and over time for macro/spectral features.

### Files to Create or Modify

- `src/finrl/features/preprocessing.py`
- `src/finrl/features/splitsafe.py`
- `tests/test_preprocessing.py`
- `tests/test_no_lookahead_preprocessing.py`

### Required Functions or Classes

- `PreprocessingConfig`
- `FittedPreprocessor`
- `build_asset_preprocessor(config)`
- `build_macro_preprocessor(config)`
- `build_spectral_preprocessor(config)`
- `fit_preprocessors(train_features, config)`
- `transform_features(features, fitted_preprocessors)`
- `fit_transform_train_transform_test(train_features, test_features, config)`
- Optional components:
  - causal forward fill / default fill
  - rolling standardization
  - clipping or winsorization

### Tests to Write

- Fit metadata records train dates only.
- Train and test data are transformed chronologically with no future rows.
- Full-dataset fitting and full train-window standardization are impossible through production API.
- Shapes and date indexes are preserved.
- Cross-sectional ranks are not standardized in a way that changes their intended meaning.
- No sklearn object is used inside preprocessing, environment step, or PPO rollout functions.

### Acceptance Criteria

- Production preprocessing API requires explicit train/test inputs or explicit train date range.
- Fitted transformer metadata records fit start/end dates.
- Tests fail if future train/test values change earlier transformed rows.
- Preprocessing tests use small deterministic arrays and do not require GPU.
- No JAX environment dependency imports sklearn.
- No preprocessing code imports sklearn.

### Dependencies on Prior Phases

- Phase 6 feature bundle.
- Phase 8 walk-forward splitter may refine train/test API, but preprocessing must already enforce split safety.

### Risks or Failure Modes

- Convenience APIs accidentally fit on all data.
- Flattening asset tensors loses asset/date alignment.
- Rolling statistics accidentally include future rows.

## Phase 8: Walk-Forward Splitter

### Objective

Create deterministic 10-year train, 1-year test, annual rolling splits that control all later fitting boundaries.

### Files to Create or Modify

- `src/finrl/backtest/walk_forward.py`
- `src/finrl/data/calendar.py`
- `tests/test_walk_forward.py`
- `tests/test_no_lookahead_splits.py`

### Required Functions or Classes

- `WalkForwardConfig`
- `WalkForwardSplit`
- `generate_walk_forward_splits(dates, config)`
- `slice_feature_bundle(features, split)`
- `slice_returns(returns, spy_returns, split)`
- `validate_split_boundaries(split)`

### Tests to Write

- Example splits:
  - train 2010-2019, test 2020
  - train 2011-2020, test 2021
  - train 2012-2021, test 2022
- Train end precedes test start.
- Test window is exactly one year where calendar permits.
- Annual retraining advances by one year.
- Preprocessing, HMM, encoder, and PPO receive only train slices for fitting.
- Policy evaluation receives frozen fitted artifacts and test slices only.

### Acceptance Criteria

- Split objects carry train/test dates, decision dates, execution dates, and holding-period dates.
- All future phases consume split objects rather than ad hoc date masks.
- Splitter tests use deterministic toy calendars locally.
- No split includes overlapping train/test samples.

### Dependencies on Prior Phases

- Phase 5 calendar.
- Phase 7 preprocessing API.

### Risks or Failure Modes

- Off-by-one errors around Friday decisions and Monday executions.
- Holidays shortening or shifting holding periods.
- Feature lookback windows accidentally using test data for initial test states.

## Phase 9: HMM Regime Detector

### Objective

Fit a Gaussian HMM on train-window market states and produce filtering-only regime probabilities for train/test periods.

### Files to Create or Modify

- `src/finrl/regimes/hmm.py`
- `src/finrl/regimes/filtering.py`
- `src/finrl/regimes/schema.py`
- `tests/test_hmm.py`
- `tests/test_no_lookahead_hmm.py`

### Required Functions or Classes

- `HMMConfig`
- `FittedHMM`
- `fit_hmm(train_phi, config)`
- `filter_regime_probabilities(fitted_hmm, phi_sequence)`
- `annual_hmm_refit(train_phi_by_split, config)`
- `validate_filtering_only(probabilities, metadata)`

### Tests to Write

- HMM fit receives train market states only.
- Filtering probabilities at time `t` are unchanged by modifying future observations.
- Output probabilities sum to 1 and are finite.
- Default state count is 4 and configurable.
- Diagonal covariance configuration is honored.
- No forward-backward smoothing API is used for evaluation features.

### Acceptance Criteria

- Regime probabilities have shape `(T, 4)` by default.
- Fitted HMM metadata records train window.
- HMM unit tests use small synthetic `phi_t` arrays and require no GPU.
- Filtering is usable for both train rollout and frozen test evaluation.
- If the selected HMM library lacks filtering-only behavior, implement or wrap a forward filter explicitly.

### Dependencies on Prior Phases

- Phase 8 walk-forward splitter.
- Phase 10 encoder eventually produces `phi_t`; interim tests may use synthetic `phi_t`.

### Risks or Failure Modes

- Library defaults return smoothed probabilities.
- HMM fitting is numerically unstable on short or low-variance sequences.
- Regime labels switch across annual refits; downstream code must treat labels as latent probabilities, not stable semantic names.

## Phase 10: Market Encoder

### Objective

Implement the JAX/Flax encoder that maps 60-day asset, macro, and spectral inputs to `phi_t in R^32`.

### Files to Create or Modify

- `src/finrl/models/encoder.py`
- `src/finrl/models/attention.py`
- `src/finrl/models/windows.py`
- `src/finrl/models/training.py`
- `src/finrl/models/checkpoints.py`
- `tests/test_encoder_shapes.py`
- `tests/test_encoder_windows.py`
- `tests/test_no_lookahead_encoder.py`

### Required Functions or Classes

- `EncoderConfig`
- `AssetEncoder`
- `CrossAssetAttention`
- `AttentionPooling`
- `MacroEncoder`
- `FusionMLP`
- `MarketEncoder`
- `build_lookback_windows(features, lookback=60)`
- `encode_market_state(params, feature_window)`
- Optional pretraining/training utilities only after objective is specified:
  - `train_encoder`
  - `save_encoder_checkpoint`
  - `load_encoder_checkpoint`

### Tests to Write

- Input/output shape tests:
  - asset input `(60, 100, F_asset)`
  - asset encoder output `(100, 64)`
  - pooled asset embedding `(64,)`
  - macro embedding `(16,)`
  - fusion output `(32,)`
- Batch and scan compatibility where practical.
- Lookback window for date `t` includes only `t-L+1:t`.
- Encoder initialization and forward pass are deterministic for fixed PRNG keys.
- Spectral feature dimension is 20.

### Acceptance Criteria

- Market encoder forward pass works under `jax.jit`.
- Encoder produces `phi_t` with dimension 32.
- No PPO dependency is introduced here.
- Shape and window tests use small deterministic arrays on CPU-only JAX.
- Full encoder training is isolated to Colab experiment scripts or notebooks.
- Training objective is not invented if unspecified; add TODO or request clarification before supervised/self-supervised training design.

### Dependencies on Prior Phases

- Phase 6 feature tensors.
- Phase 7 preprocessing.
- Phase 8 split windows.

### Risks or Failure Modes

- Memory use too high for Colab when windowing full configured `N`-stock history.
- Shared LSTM accidentally becomes per-asset independent parameters.
- Attention pooling masks or dimensions are incorrect.
- Unspecified encoder training objective leads to invented architecture.

## Phase 11: PPO Trainer

### Objective

Implement PPO after environment, accounting, rewards, data, features, preprocessing, walk-forward splitting, HMM, and encoder scaffolding are correct. PPO should train on a single historical trajectory per walk-forward train split and freeze policy for test evaluation. Because Phase 4 benchmark strategies are now implemented last, PPO implementation may proceed before final benchmark validation, but production PPO claims remain blocked until Phase 4 passes.

### Files to Create or Modify

- `src/finrl/ppo/policy.py`
- `src/finrl/ppo/value.py`
- `src/finrl/ppo/distributions.py`
- `src/finrl/ppo/gae.py`
- `src/finrl/ppo/losses.py`
- `src/finrl/ppo/trainer.py`
- `src/finrl/ppo/checkpoints.py`
- `tests/test_policy.py`
- `tests/test_gae.py`
- `tests/test_ppo_losses.py`
- `tests/test_ppo_rollout.py`
- `tests/test_no_lookahead_ppo.py`

### Required Functions or Classes

- `PPOConfig`
- `PortfolioActor`
- `PortfolioCritic`
- `ActorCriticState`
- `temperature_softmax(logits, temperature)`
- `build_ppo_state(phi, regime_probs, portfolio_context)`
- `sample_action(params, state, rng)`
- `evaluate_action_logprob(params, state, action)`
- `compute_gae(rewards, values, dones, gamma, lambda_)`
- `ppo_clip_loss(...)`
- `value_loss(...)`
- `entropy_bonus(...)`
- `collect_train_trajectory(policy, env, train_data)`
- `train_ppo_on_split(train_artifacts, config)`
- `evaluate_frozen_policy(policy_checkpoint, test_artifacts)`

### Tests to Write

- Actor outputs `N + 1` logits and valid long-only weights after temperature softmax.
- Weights sum to 1 and are nonnegative.
- PPO state dimension matches architecture-derived context:
  - `phi_t` 32
  - regime probabilities 4
  - weights `N + 1`
  - drawdown 1
  - previous turnover 1
- GAE is correct on small hand-computed trajectories.
- PPO loss remains finite for synthetic trajectories.
- Rollout uses environment accounting path from Phase 2.
- PPO fit occurs only on train split.
- Policy parameters do not update during test evaluation.

### Acceptance Criteria

- PPO runs on synthetic and small real prepared data without breaking accounting tests.
- Policy checkpoint can be saved and loaded.
- Frozen test evaluation produces deterministic metrics for a fixed seed.
- PPO implementation is blocked unless Phase 2 and Phase 3 acceptance criteria are met.
- Production PPO evaluation and research conclusions are blocked until deferred Phase 4 benchmark acceptance criteria are met.
- Local PPO tests use tiny deterministic trajectories on CPU-only JAX.
- Full PPO training is isolated to Colab experiment scripts or notebooks.

### Dependencies on Prior Phases

- Phase 2 JAX environment.
- Phase 3 accounting/reward tests.
- Phase 8 walk-forward splits.
- Phase 9 regime probabilities.
- Phase 10 market states.
- Phase 4 benchmark backtests for final validation only, because Phase 4 is implemented last.

### Risks or Failure Modes

- PPO complexity hides accounting regressions.
- Action distribution/logprob design conflicts with deterministic target-weight interpretation.
- Training loop exceeds Colab memory or runtime.
- Evaluation accidentally updates policy or normalization statistics.

## Phase 12: Full Walk-Forward Experiment Runner

### Objective

Tie data, features, preprocessing, encoder, HMM, PPO, benchmark integration hooks, and reporting into a strict out-of-sample annual walk-forward experiment. Benchmark strategy implementations themselves are deferred to Phase 4, which is implemented last.

### Files to Create or Modify

- `src/finrl/experiments/run_walk_forward.py`
- `src/finrl/experiments/config.py`
- `src/finrl/experiments/artifacts.py`
- `src/finrl/experiments/reporting.py`
- `src/finrl/backtest/results.py`
- `notebooks/colab_walk_forward.ipynb` or `examples/walk_forward.py`
- `tests/test_experiment_runner.py`
- `tests/test_no_lookahead_experiment.py`

### Required Functions or Classes

- `ExperimentConfig`
- `ExperimentArtifacts`
- `WalkForwardResult`
- `run_walk_forward_experiment(config)`
- `run_split(split, raw_data, config)`
- `fit_train_artifacts(split, features, config)`
- `evaluate_test_split(split, frozen_artifacts, config)`
- `run_benchmark_suite(split, returns, spy_returns, config)`
- `aggregate_walk_forward_results(results)`
- `write_report(results, output_dir)`

### Tests to Write

- End-to-end synthetic walk-forward run with at least two splits.
- Artifacts fitted on each train split are reused frozen for its test split.
- Preprocessing, HMM, encoder, and PPO fit metadata match train window only.
- Benchmark integration points and PPO use identical holding-period returns and SPY returns once Phase 4 is implemented.
- Results aggregation preserves split boundaries.
- Runner can disable PPO and run data/features/preprocessing/environment flow on synthetic fixtures before benchmark strategies exist.

### Acceptance Criteria

- Full pipeline can run in environment-only smoke mode first:
  - Data
  - Features
  - Preprocessing
  - Environment
- PPO mode is available only after environment-only smoke mode and no-look-ahead checks are correct.
- Local experiment tests use tiny synthetic walk-forward runs on CPU-only JAX.
- Full configured `N`-stock walk-forward experiments are isolated to Colab scripts or notebooks.
- Output includes per-split and aggregate metrics:
  - cumulative return
  - annualized return
  - volatility
  - max drawdown
  - turnover
  - transaction costs
  - SPY-relative alpha
- Experiment is reproducible for fixed config and seed.
- Colab path is documented and avoids heavy local-only assumptions.
- Final benchmark comparisons are added after deferred Phase 4 is implemented.

### Dependencies on Prior Phases

- Phases 1-3 and 5-11.
- Phase 4 is intentionally deferred until after this runner exists; Phase 12 should expose benchmark integration hooks but not require benchmark strategy implementations yet.

### Risks or Failure Modes

- The runner becomes a monolith instead of composing tested components.
- Artifacts from one split leak into another.
- Reporting hides per-split failures behind aggregate performance.
- PPO mode is treated as research-valid before deferred benchmark validation.

## Milestone Gates

1. Repository imports and tests run.
2. JAX accounting and environment tests pass.
3. Accounting and reward tests cover hand-computed fixtures and edge cases.
4. Data ingestion with yfinance and Polars produces aligned train/test-ready data for configured `N`.
5. Feature generation produces aligned tensors for configured `N`.
6. Preprocessing and splitter pass no-look-ahead tests.
7. HMM filtering outputs pass no-look-ahead tests.
8. Encoder produces `phi_t` under JAX with correct windows.
9. PPO trains and evaluates on a frozen test split in synthetic/smoke mode.
10. Full walk-forward runner completes with per-split reporting hooks.
11. Phase 4 benchmark strategies are implemented last and pass synthetic plus walk-forward validation.
12. Final reports include benchmark comparisons before any research conclusions are drawn.

## Non-Negotiable Stop Conditions

- Do not implement PPO before the JAX environment and accounting/reward tests are correct.
- Do not treat PPO results as production-valid before deferred Phase 4 benchmark backtests are correct.
- Do not fit preprocessing, HMM, encoder, PPO, or benchmark statistics on full data for production paths.
- Do not use HMM smoothing probabilities in evaluation.
- Do not require a local GPU for tests or core package imports.
- Do not put full-universe training assumptions into unit tests.
- Do not hard-code a 100-stock universe; use configured `N` stocks plus cash.
- Do not change `ARCHITECTURE.MD` without explicit user approval.
- Do not invent unspecified research architecture; add a TODO or request clarification.
