# Risk layer

`finrl.risk` provides causal NumPy risk-feature estimation and a pure JAX risk
layer for DPO.  The layer receives raw target weights and the drifted holdings
inside the chronological accounting scan, then enforces long-only tradability,
minimum trade, ADV, turnover, position, volatility, drawdown, and cash limits.

Risk estimates at a date are trailing only. Missing ADV is represented as zero,
so it never grants unlimited liquidity; nontradable holdings are frozen against
increases. Asset-level spread and square-root impact costs are charged exactly
once from executed trades. `RiskConfig` is shared by experiment training and
evaluation configuration.
