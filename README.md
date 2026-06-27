# FINRL Score Allocation

JAX-native research framework for long-only allocation across a
configurable stock universe plus cash.

Rebalancing can be configured as `weekly` or `daily`; weekly mode uses the last
trading session of each week when Friday is a market holiday.

The learned policy is intentionally small:

```text
decision-date asset features
  -> accumulation and liquidity-exit score heads
  -> two scores per asset
  -> direct allocation head
  -> stock and cash weights
```

Raw features are used only by the score heads. The allocation head never sees
them directly, and the model has no recurrent encoder. Trailing information is
computed by the split-safe feature pipeline.

## Development

Install the package with development dependencies, then run:

```bash
python -m pytest -q
```

See `ARCHITECTURE.MD` for the authoritative design and `TESTING.md` for the
required financial and research-correctness checks.
