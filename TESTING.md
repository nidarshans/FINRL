# Testing

Run the CPU test suite with:

```bash
python -m pytest -q
```

Mandatory financial tests cover turnover, transaction costs, portfolio value,
drawdown, benchmark returns, cash accounting, daily/weekly rebalancing, and
holiday-adjusted weekly decisions.

The learned path must additionally verify:

* explicit accumulation and liquidity-exit feature routing;
* allocation-head input width is exactly two;
* unrouted raw features cannot affect allocations;
* future feature changes cannot affect earlier decisions;
* gradients reach both score heads and the allocation head;
* weights are finite, nonnegative, and sum to one;
* deterministic JAX/JIT execution;
* train-only preprocessing and fitting;
* frozen walk-forward evaluation and exact date/return alignment.
