# FINRL Direct Allocation

JAX-native research framework for long-only allocation across a
configurable stock universe plus cash.

Rebalancing can be configured as `weekly` or `daily`; weekly mode uses the last
trading session of each week when Friday is a market holiday.

The learned policy is intentionally small:

```text
decision-date asset features
  -> explicit direct-feature routing
  -> direct allocation head
  -> stock and cash weights
```

The allocation head sees only the explicitly routed feature tensor and has no
recurrent encoder. Trailing information is computed by the split-safe feature
pipeline.

## Development

Install the package with development dependencies, then run:

```bash
python -m pytest -q
```

See `ARCHITECTURE.MD` for the authoritative design and `TESTING.md` for the
required financial and research-correctness checks.
