# Implementation Plan

The active vertical path is:

1. Ingest and validate chronological market data.
2. Generate trailing, split-safe asset features.
3. Validate deterministic JAX portfolio accounting and configurable daily/weekly scheduling.
4. Run benchmark strategies.
5. Route approved feature columns into accumulation and liquidity-exit heads.
6. Feed only the resulting two scores per asset into the allocation head.
7. Optimize the score heads and allocation head through the differentiable
   portfolio objective.
8. Fit preprocessing and policy parameters on each train split and evaluate
   frozen artifacts on its test split.

Do not add model complexity before the end-to-end benchmark and walk-forward
paths are correct. Architecture changes require explicit user approval.
