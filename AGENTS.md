# Instructions for AI Coding Agents

## Source of Truth

ARCHITECTURE.md is the authoritative specification.

If code conflicts with ARCHITECTURE.md, update the code.

Do not update the architecture without explicit user approval.

---

## Development Philosophy

Build vertically.

Prioritize working end-to-end functionality over premature optimization.

A working benchmark backtest is more valuable than partially implemented PPO.

---

## Implementation Order

1. Data ingestion
2. Feature engineering
3. Trading environment
4. Benchmark strategies
5. HMM
6. Encoder
7. PPO
8. Walk-forward testing

Do not skip ahead.

---

## Testing Requirements

All core financial calculations require unit tests.

Mandatory tests:

* Turnover calculation
* Transaction cost calculation
* Portfolio value evolution
* Drawdown calculation
* Benchmark return calculation
* Cash accounting
* Weekly rebalance logic

No new environment logic may be merged without tests.

---

## Environment Principles

The trading environment must remain:

* Deterministic
* Functional
* JAX-compatible

Avoid hidden mutable state.

Prefer pure functions.

---

## Research Constraints

Avoid look-ahead bias.

Never use future information.

HMM inference must use filtering only.

No forward-backward smoothing.

---

## Performance Constraints

Target platform:

* Google Colab
* Single GPU

Favor simplicity and reliability over maximum complexity.

---

## Coding Style

* Type hints required
* Dataclasses preferred
* Small focused modules
* Clear docstrings
* Minimal dependencies

---

## When Uncertain

Do not invent architecture.

Create a GitHub issue or TODO comment and request clarification.
