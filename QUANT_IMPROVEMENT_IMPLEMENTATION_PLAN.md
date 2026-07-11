# Quantitative Strategy Improvement Plan

## Objective

Turn the current deterministic walk-forward research pipeline into a credible,
continuous, cost-aware out-of-sample portfolio evaluation system before adding
model complexity.

This plan is based on the executable code and tests. It does not modify
`ARCHITECTURE.MD`. Any item that changes an architectural contract must receive
explicit approval before implementation.

## Guiding Rules

- Preserve deterministic, functional, JAX-compatible environment logic.
- Use only information available by each decision timestamp.
- Complete each phase end to end before starting the next phase.
- Add unit tests for every financial calculation and backtest state transition.
- Compare every learned policy with simple, reproducible benchmarks.
- Do not claim performance improvement without frozen out-of-sample evidence.

## Phase 0: Establish a Reproducible Baseline

### Goal

Create a single command that records the current strategy's behavior before any
financial or modeling changes.

### Work

- Add an experiment entry point with a fully serialized configuration.
- Record the input-data fingerprint, ticker universe, date range, seed, package
  versions, and git revision.
- Save per-period returns, gross returns, allocations, turnover, transaction
  costs, and benchmark returns.
- Save per-split and aggregate metrics.
- Add benchmark policies:
  - SPY buy-and-hold.
  - Equal-weight risky assets.
  - Equal-weight with cash.
  - Previous-weight/no-trade policy.
- Add a small deterministic synthetic-data smoke experiment for CI.

### Primary Code Areas

- `src/finrl/experiments/run_walk_forward.py`
- `src/finrl/experiments/artifacts.py`
- `src/finrl/experiments/reporting.py`
- `src/finrl/backtest/results.py`

### Acceptance Criteria

- Two runs with the same inputs and seed produce identical artifacts.
- Every learned-policy result is presented beside all required benchmarks.
- The complete test suite remains green.

## Phase 1: Correct Market Data and Return Construction

### Goal

Remove corporate-action, missing-data, and tradability distortions before
changing the model.

### Work

1. Replace zero-filled missing OHLCV values with explicit missing observations.
2. Build a consistent adjusted OHLC series:
   - Derive an adjustment factor from adjusted and unadjusted close.
   - Apply the factor to open, high, low, and close.
   - Keep volume adjustment semantics explicit.
3. Calculate features and open-to-open returns from the same adjusted price
   basis.
4. Add a decision-date tradability mask requiring valid execution and next
   execution prices.
5. Prevent the allocator from assigning weight to non-tradable assets.
6. Reject or quarantine implausible prices and returns instead of replacing
   them with zero.
7. Add data-coverage reporting by ticker and date.

### Tests

- Split-adjusted prices do not create artificial returns or momentum signals.
- A missing execution price cannot become a zero return.
- Non-tradable assets receive zero target weight.
- A ticker/date gap cannot silently shorten or reorder the return panel.
- Adjusted open-to-open returns match hand-calculated fixtures.

### Acceptance Criteria

- Known split fixtures have continuous adjusted price and return series.
- All portfolio return rows have an auditable price pair and tradability state.
- No ingestion path fills a missing market price with zero.

## Phase 2: Complete the Existing Portfolio Objective

### Goal

Make every active DPO configuration parameter affect the loss as advertised.

### Work

- Add the configured turnover penalty to `dpo_loss`.
- Add the configured concentration penalty to `dpo_loss`.
- Keep transaction costs in net returns; document that the turnover penalty is
  an additional regularizer.
- Decide and document whether turnover means full L1 traded notional or
  one-way turnover (`0.5 * L1`). Use one convention consistently in training,
  execution, reports, and tests.
- Validate all penalty coefficients as finite and non-negative.
- Remove or deprecate environment reward parameters that remain intentionally
  inactive, or activate them with tests.
- Add numerical guards for returns at or below `-100%` before `log1p`.

### Tests

- Increasing `lambda_turnover` increases loss for a higher-turnover path.
- Increasing `lambda_concentration` increases loss for a more concentrated
  path.
- Training-loss accounting matches environment accounting.
- Loss and gradients remain finite on stressed but valid return paths.
- Cost convention fixtures cover buys, sells, cash deployment, and drift.

### Acceptance Criteria

- No public objective parameter is ignored.
- DPO and environment accounting agree within numerical tolerance.
- Existing ignored-penalty tests are replaced with behavioral tests.

## Phase 3: Make Walk-Forward Evaluation Continuous

### Goal

Produce one investable out-of-sample path while still reporting independent
split diagnostics.

### Work

- Separate model refitting state from portfolio accounting state.
- At an adjacent split boundary, carry forward:
  - Drifted portfolio weights.
  - Portfolio value.
  - Running peak.
  - Drawdown.
  - Previous turnover.
- Refit only model and preprocessing parameters at the boundary.
- Charge turnover from actual carried holdings to the new policy target.
- Detect and reject overlapping or duplicate out-of-sample dates before
  aggregation.
- Preserve an optional independent-split evaluation mode for diagnostics, but
  do not label it a continuous equity curve.

### Tests

- Adjacent splits do not reset to cash.
- Boundary turnover uses the previous split's drifted terminal weights.
- Continuous aggregate equity equals sequential environment evolution.
- Overlapping test windows cannot be concatenated silently.
- Gapped test windows are reported explicitly.

### Acceptance Criteria

- The headline walk-forward curve is generated by one uninterrupted accounting
  state.
- Split-level and continuous-path metrics are both available and clearly named.

## Phase 4: Strengthen Reporting and Statistical Validation

### Goal

Measure whether performance is economically meaningful, risk-adjusted, and
statistically credible.

### Work

- Add CAGR, annualized volatility, Sharpe, Sortino, Calmar, hit rate, best/worst
  period, and downside deviation.
- Add active return, tracking error, information ratio, beta, and regression
  alpha versus SPY.
- Rename the current cumulative-return difference so it is not described as
  statistical alpha.
- Add annualized turnover, gross-versus-net performance, and cost drag.
- Add rolling 13-, 26-, and 52-week diagnostics.
- Add drawdown duration and recovery time.
- Add stationary/block-bootstrap confidence intervals.
- Add parameter and execution stress tables:
  - Transaction costs at 0.5x, 1x, 2x, and 3x.
  - One-session signal delay.
  - Daily versus weekly rebalance where supported.
  - Multiple deterministic seeds.

### Tests

- Metrics match hand-calculated fixtures.
- Alpha/beta regression matches a reference calculation.
- Annualization respects the configured rebalance frequency.
- Bootstrap results are reproducible for a fixed seed.

### Acceptance Criteria

- Every report distinguishes gross, net, benchmark-relative, and risk-adjusted
  performance.
- Headline results include uncertainty and cost sensitivity.

## Phase 5: Integrate Benchmark Predictive Models

### Goal

Establish strong tabular and heuristic baselines before expanding the neural
policy.

### Work

1. Integrate the existing LightGBM modules into the walk-forward runner.
2. Construct schedule-aligned forward-return targets explicitly.
3. Purge observations whose label horizon crosses a train/validation or
   train/test boundary.
4. Add an inner chronological validation window for early stopping and
   hyperparameter selection.
5. Add simple signal benchmarks:
   - Medium-term momentum.
   - Short-term reversal.
   - Volatility-scaled momentum.
   - Equal-weight top-quantile signal portfolio.
6. Convert model scores to weights using the same costs, constraints, and
   accounting path as DPO.
7. Save feature importance and prediction diagnostics per split.

### Tests

- No training target uses returns after the allowed label boundary.
- LightGBM receives only training and inner-validation observations.
- Predictions preserve decision-date and ticker order.
- All policy types use identical execution return tables and cost accounting.

### Acceptance Criteria

- DPO, LightGBM, heuristic signals, equal weight, and SPY run through the same
  frozen out-of-sample harness.
- Model selection never accesses the outer test window.

## Phase 6: Improve Portfolio Construction

### Goal

Convert forecasts into stable, diversified, capacity-aware allocations.

### Work That Fits the Current Structure

- Add per-name maximum weights.
- Add minimum signal thresholds and no-trade bands.
- Smooth targets toward drifted current weights.
- Add cross-sectional feature ranks or robust z-scores calculated at each
  decision date.
- Add rolling covariance estimation with shrinkage.
- Add portfolio volatility and beta diagnostics.
- Add liquidity-aware transaction costs using spread/ADV proxies when data is
  available.

### Architecture Approval Required Before Implementation

The following materially change the allocation contract and must be approved
before coding:

- A learned cash logit or explicit equity/cash risk gate.
- Sector, industry, beta, and tracking-error constraints.
- A forecast-plus-constrained-optimizer architecture.
- A policy that consumes previous weights, drawdown, or other portfolio state.
- Short selling or leverage.

### Candidate Constrained Objective

For predicted return vector `mu`, covariance `Sigma`, previous weights
`w_previous`, and optional benchmark weights `w_benchmark`:

```text
maximize:
    mu.T @ w
    - risk_aversion * w.T @ Sigma @ w
    - turnover_aversion * trading_cost(w - w_previous)
    - active_risk_aversion
      * (w - w_benchmark).T @ Sigma @ (w - w_benchmark)
```

Subject to long-only, fully invested or cash-enabled, position, turnover,
liquidity, and exposure constraints.

### Acceptance Criteria

- Constraints hold on every evaluated date.
- Net performance improvements survive doubled transaction costs.
- Concentration, turnover, and tail risk improve without test-set tuning.

## Phase 7: Add Regime and Macro Information

### Goal

Use state-dependent information only after the baseline pipeline is trustworthy.

### Work

- Route the already-computed macro features into an explicitly tested model
  input path.
- Replace dummy spectral columns only after defining causal spectral features.
- Implement regime inference using filtering only; never use smoothed posterior
  states in features or allocation.
- Fit every regime model inside each training split.
- Update regime probabilities sequentially through validation and test periods.
- Compare regime-conditioned policies with the same policy without regime
  inputs.

### Tests

- Changing future observations cannot alter past regime probabilities.
- Regime parameters are fitted only on the training window.
- Macro release timing is lagged to actual availability where applicable.
- Removing regime inputs produces a valid ablation run.

### Acceptance Criteria

- Regime and macro features add stable outer-test value across multiple splits,
  not only aggregate backtest value.
- Filtering-only behavior is demonstrated by a future-perturbation test.

## Phase 8: Robustness, Capacity, and Release Gate

### Goal

Determine whether the strategy is robust enough for paper trading.

### Work

- Add point-in-time universe membership and delisting-return support.
- Measure performance by subperiod, volatility regime, sector, and market trend.
- Run feature and model ablations.
- Estimate capacity using ADV participation and nonlinear impact assumptions.
- Run parameter-neighborhood tests instead of reporting only the best setting.
- Add probability-of-backtest-overfitting or deflated-Sharpe diagnostics when
  comparing many configurations.
- Freeze one final configuration before the last untouched holdout evaluation.

### Release Gate

Proceed to paper trading only if all conditions hold:

- Positive net excess return in a majority of outer test splits.
- No single split or small group of names explains most total profit.
- Performance remains acceptable at 2x modeled transaction costs.
- Drawdown and turnover stay within predefined limits.
- Results remain directionally stable under execution delay and reasonable
  parameter perturbations.
- The final untouched holdout is evaluated exactly once after configuration
  freeze.

## Recommended Delivery Sequence

| Milestone | Phases | Deliverable |
|---|---:|---|
| M1: Trustworthy data | 0-1 | Reproducible baseline with adjusted, validated returns |
| M2: Trustworthy accounting | 2-3 | Complete objective and continuous OOS simulation |
| M3: Trustworthy evidence | 4 | Risk, active-performance, and uncertainty report |
| M4: Strong baselines | 5 | Integrated LightGBM and heuristic comparison |
| M5: Better allocation | 6 | Stable constrained portfolio construction |
| M6: Additional signals | 7 | Causal macro/regime ablation results |
| M7: Paper-trading gate | 8 | Robustness, capacity, and frozen holdout report |

## Definition of Done for Every Phase

- Implementation is type-annotated and documented.
- Core financial calculations have unit tests.
- Look-ahead and boundary tests are included.
- Existing tests pass.
- A deterministic experiment demonstrates the phase end to end.
- Gross and net effects are reported separately.
- Results and configuration are serialized for reproduction.
- Any architectural change has explicit approval before implementation.
