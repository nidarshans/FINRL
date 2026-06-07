# Testing Rules

## Core Principle

Financial correctness is more important than model sophistication.

No model component should be trusted until the environment, accounting, and benchmark logic are tested.

---

# Required Test Categories

## 1. Unit Tests

Every pure calculation must have direct unit tests.

Required:

- turnover
- transaction costs
- gross portfolio return
- net portfolio return
- portfolio value update
- running peak update
- drawdown
- SPY-relative reward
- cash return
- weight normalization checks

---

## 2. Environment Step Tests

The environment step must be tested end-to-end.

Each test should verify:

- input state
- target weights
- asset returns
- SPY return
- transaction cost
- new portfolio value
- new peak value
- drawdown
- reward
- next weights
- turnover

---

## 3. Invariant Tests

These must always hold:

\[
\sum_i w_i = 1
\]

\[
w_i \ge 0
\]

\[
V_t > 0
\]

\[
DD_t \in [0, 1]
\]

\[
TO_t \ge 0
\]

\[
TC_t \ge 0
\]

Reward must be finite.

---

## 4. No Look-Ahead Tests

Any train/test split must verify:

- preprocessing fit dates are train-only
- test data is transformed using train-fitted transformers
- HMM is fit on train only
- PPO is fit on train only
- policy is frozen during test
- benchmark return uses the exact same holding period as the portfolio

No function may fit on full data unless explicitly marked as exploratory.

---

## 5. Benchmark Tests

Benchmark policies must be deterministic and reproducible.

Required benchmark tests:

- cash-only strategy earns cash return only
- equal-weight strategy assigns equal stock weights
- SPY benchmark return matches SPY open-to-open return
- momentum strategy does not use future returns
- risk parity uses only past-window covariance

---

## 6. JAX Compatibility Tests

Core environment functions must support:

- `jax.jit`
- `jax.lax.scan`
- float32 inputs
- deterministic outputs

No hidden mutation.

No Python-side state changes inside environment step.

---

## 7. Numerical Stability Tests

Test edge cases:

- zero returns
- negative returns
- high turnover
- all-cash allocation
- all-stock allocation
- SPY down period
- SPY up period
- drawdown breach
- zero transaction cost
- nonzero transaction cost

All outputs must remain finite.

---

# Required Testing Tools

Use:

- `pytest`
- `chex`
- `jax.numpy`
- `numpy.testing`

Optional later:

- `hypothesis`

---

# Tolerances

Use strict tolerances for accounting:

```python
rtol = 1e-6
atol = 1e-8