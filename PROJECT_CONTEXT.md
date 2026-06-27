# Project Context

FINRL is a JAX-native daily or weekly portfolio-allocation research framework. It
combines public market data, split-safe trailing features, learned accumulation
and liquidity-exit scores, direct portfolio optimization, deterministic
portfolio accounting, benchmark comparison, and strict walk-forward testing.

The learned policy receives one feature panel per decision date with shape
`(N, F)`. Explicit allowlists route inputs to two independent score MLPs. Their
outputs are stacked to `(N, 2)` and are the allocation head's complete input.
The head emits long-only weights over `N` stocks plus cash.

There is no recurrent model or reinforcement-learning path. All temporal signal
must come from trailing, no-look-ahead feature calculations. Preprocessing and
policy fitting happen on train data only; test evaluation uses frozen artifacts.

`ARCHITECTURE.MD` is the authoritative specification.
