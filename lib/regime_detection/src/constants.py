# ==============================================================================
# CONFIGURATION
# ==============================================================================

SECTORS = [
    "XLK",   # Technology
    "XLV",   # Health Care
    "XLY",   # Consumer Discretionary
    "XLC",   # Communication Services
    "XLE",   # Energy
    "XLU",   # Utilities
    "GLD",   # Gold
    "XSD",   # Semiconductors
    "XAR",   # Defense
    "BIL",   # Money Market / Treasuries
]

BENCHMARK   = "SPYG"
TRAIN_START = "2019-01-01"
TRAIN_END   = "2022-12-31"   # Train window  (used in simple train/test mode)
TEST_START  = "2023-01-01"   # Test / backtest window
TEST_END    = "2026-05-14"

# Walk-forward mode ──────────────────────────────────────────────────────
# Set WALK_FORWARD=True to run rolling/anchored WF instead of single train/test
WALK_FORWARD        = True

# "rolling"  → fixed-size train window slides forward each step
# "anchored" → train window always starts at WF_FULL_START, expands each step
WF_MODE             = "rolling"

WF_FULL_START       = "2019-01-01"  # Earliest data used in walk-forward
WF_FULL_END         = "2022-12-14"  # Last date of the full dataset
WF_TRAIN_DAYS       = 126           # ~2 trading years per train window
WF_OOS_DAYS         = 21            # ~3 months out-of-sample per step
WF_MIN_TRAIN_DAYS   = 0   # Minimum bars required before first OOS step
VERBOSE             = False  # Toggle print statements during training
# ──────────────────────────────────────────────────────────────────────────

# HMM & Kalman parameters
R_BASE          = 0.1
Q_NOISE         = 0.001
GAMMA           = 2.0
VOL_WINDOW      = 5
KVO_FAST_SPAN   = 34
KVO_SLOW_SPAN   = 55
HMM_ITER        = 150

# ── Correlation / PCA Regime Detection ────────────────────────────────────────
CORR_METRIC        = "garch_returns"   # Options: "garch_returns" | "volume" | "raw_returns"
CORR_WINDOW        = 21               # Rolling window in trading days
GARCH_P            = 1                # GARCH(p,q) order
GARCH_Q            = 1
PCA_N_COMPONENTS   = None             # None = keep all; int = keep top-N
CORR_DELTA_WINDOW  = 5                # Days over which to compute Eigenvalue_1_Delta
AR_SCORE_WEIGHT    = 0.15             # Weight to penalize high systemic risk in Rank_Score
# ──────────────────────────────────────────────────────────────────────────────

# Walk-forward backtest parameters
REBAL_FREQ      = 5        # Rebalance every N trading days
TOP_N           = 1        # Sectors held at a time
BULL_THRESH     = 0.55     # Minimum P(Bull) to qualify
BEAR_EXIT_PROB  = 0.25     # Immediate exit if P(Bear) exceeds this

# Soft-exit
DIVERGENCE_MULT    = 0.5
DIVERGENCE_LOOKBACK = 20

FEATURES = [
    'KVO', 'Innovation_Z', 'MACD'
]

CORR_COLS = ['Eigenvalue_1', 'Eigenvalue_2', 'Absorption_Ratio', 'Absorption_Ratio_Garch', 'Corr_Mean', 'Corr_Dispersion', 'Eigenvalue_1_Delta']
SIGNAL_COLS = ['VF', 'Filtered_VF', 'Innovation_Z', 'KVO_Fast', 'KVO_Slow', 'KVO', 'MACD']

REGIME_COLORS = {'Bull': '#4CAF50', 'Stagnant': '#FFC107', 'Bear': '#F44336'}
SECTOR_COLORS = [
    '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
    '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#aec7e8'
]
