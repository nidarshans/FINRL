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
    "VXX"
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

WF_FULL_START       = "2024-01-01"  # Earliest data used in walk-forward
WF_FULL_END         = "2026-05-14"  # Last date of the full dataset
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
CORR_METRIC        = "volume"   # Options: "garch_returns" | "volume" | "raw_returns"
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
    'KVO', 'Innovation_Z', 'MACD', 'Absorption_Ratio'
]

CORR_COLS = ['Eigenvalue_1', 'Eigenvalue_2', 'Absorption_Ratio', 'Absorption_Ratio_Garch', 'Corr_Mean', 'Corr_Dispersion', 'Eigenvalue_1_Delta']
SIGNAL_COLS = ['VF', 'Filtered_VF', 'Innovation_Z', 'KVO_Fast', 'KVO_Slow', 'KVO', 'MACD']

REGIME_COLORS = {'Bull': '#4CAF50', 'Stagnant': '#FFC107', 'Bear': '#F44336'}
SECTOR_COLORS = [
    '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
    '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#aec7e8'
]


gics_tickers_only = {
    "Energy": {
        "Oil & Gas Drilling": ["HP", "PTEN", "NBR", "NE", "DO"],
        "Oil & Gas Equipment & Services": ["SLB", "HAL", "BKR", "NOV", "FTI"],
        "Integrated Oil & Gas": ["XOM", "CVX", "SHEL", "TTE", "BP"],
        "Oil & Gas Exploration & Production": ["COP", "EOG", "OXY", "HES", "DVN"],
        "Oil & Gas Refining & Marketing": ["MPC", "VLO", "PSX", "DK", "HFMC"],
        "Oil & Gas Storage & Transportation": ["WMB", "OKE", "KMI", "TRGP", "LNG"],
        "Coal & Consumable Fuels": ["CEIX", "ARCH", "BTU", "AMR", "HCC"]
    },
    "Materials": {
        "Commodity Chemicals": ["DOW", "LYB", "EMN", "OLN", "WLK"],
        "Diversified Chemicals": ["DD", "CE", "FMC", "CC", "NEU"],
        "Fertilizers & Agricultural Chemicals": ["CTVA", "CF", "MOS", "NTR", "ICL"],
        "Industrial Gases": ["LIN", "APD", "ARG", "MATH"],
        "Specialty Chemicals": ["SHW", "ECL", "PPG", "IFF", "CE"],
        "Construction Materials": ["MLM", "VMC", "EXP", "CX", "CRH"],
        "Metal, Glass & Plastic Containers": ["BALL", "CCK", "OI", "BERY", "ATR"],
        "Paper & Plastic Packaging Products & Materials": ["WRK", "IP", "PKG", "SEE", "SON"],
        "Aluminum": ["AA", "CENX", "KALU", "ACH", "ALV"],
        "Diversified Metals & Mining": ["FCX", "NEM", "RIO", "BHP", "VALE"],
        "Copper": ["FCX", "SCCO", "TECK", "ERO", "CS"],
        "Gold": ["NEM", "GOLD", "FNV", "AEM", "GFI"],
        "Precious Metals & Minerals": ["MP", "LAC", "ALB", "SQM", "LTHM"],
        "Silver": ["PAAS", "HL", "AG", "FSM", "EXK"],
        "Steel": ["NUE", "STLD", "CLF", "X", "RS"],
        "Forest Products": ["WY", "PCH", "RYN", "UFPI", "MERC"],
        "Paper Products": ["IP", "PKG", "WRK", "SUZ", "MERC"]
    },
    "Industrials": {
        "Aerospace & Defense": ["GE", "RTX", "LMT", "BA", "NOC"],
        "Building Products": ["CARR", "TT", "JCI", "MAS", "AOI"],
        "Construction & Engineering": ["PWR", "ACM", "EME", "FLR", "KBR"],
        "Electrical Components & Equipment": ["EMR", "ETN", "AME", "GNRC", "HUBB"],
        "Heavy Electrical Equipment": ["GE", "VWS", "SIV", "BBRY", "BBN"],
        "Industrial Conglomerates": ["HON", "MMM", "GE", "DHR", "ROK"],
        "Construction Machinery & Heavy Transportation Equipment": ["CAT", "DE", "PCAR", "TEX", "OSK"],
        "Agricultural & Farm Machinery": ["DE", "AGCO", "CNHI", "TTC", "ALSN"],
        "Industrial Machinery & Supplies & Components": ["ITW", "PH", "IR", "SNA", "SWK"],
        "Trading Companies & Distributors": ["FAST", "GWW", "URI", "WCC", "FTV"],
        "Commercial Printing": ["RRD", "CMP", "EBF", "QUAD", "ARC"],
        "Environmental & Facilities Services": ["WM", "RSG", "SRCL", "WCN", "CLH"],
        "Office Services & Supplies": ["CINT", "UNF", "SPB", "ACCO", "HNI"],
        "Diversified Support Services": ["ARMK", "CINT", "PAYX", "ADP", "GPN"],
        "Security & Alarm Services": ["ADT", "ALLE", "BCO", "G4S", "GEO"],
        "Human Resource & Employment Services": ["MAN", "RHI", "KFY", "ASGN", "NSP"],
        "Research & Consulting Services": ["ACN", "EXPD", "REX", "G", "BR"],
        "Data Processing & Outsourced Services": ["FI", "FIS", "GPN", "BR", "PAYX"],
        "Air Freight & Logistics": ["UPS", "FDX", "EXPD", "CHRW", "DSV"],
        "Passenger Airlines": ["DAL", "UAL", "AAL", "LUV", "ALK"],
        "Marine Transportation": ["MATX", "GSL", "SBLK", "EGLE", "GNK"],
        "Rail Transportation": ["UNP", "CSX", "NSC", "CP", "CNI"],
        "Cargo Ground Transportation": ["JBHT", "ODFL", "KNX", "XPO", "ARCB"],
        "Passenger Ground Transportation": ["UBER", "LYFT", "HTZ", "CAR", "ALGT"],
        "Airport Services": ["PAC", "ASR", "OMAB", "AENA", "FHZN"],
        "Highways & Railtracks": ["ALSR", "GVP", "EAS", "VINCI", "ATL"],
        "Marine Ports & Services": ["DPW", "WWD", "HPH", "ICT", "PORT"],
        "Transportation Infrastructure": ["QRTEB", "MIC", "AMTR", "OSG", "GLNG"]
    },
    "Consumer Discretionary": {
        "Automotive Parts & Equipment": ["APTIV", "BWA", "LKQ", "ALV", "MGA"],
        "Tires & Rubber": ["GT", "BRDC", "MLNK", "CTTA", "HANK"],
        "Automobile Manufacturers": ["TSLA", "TM", "GM", "F", "RACE"],
        "Motorcycle Manufacturers": ["HOG", "PII", "HOND", "YAMH", "BMW"],
        "Consumer Electronics": ["AAPL", "SONY", "GME", "UEIC", "VZIO"],
        "Home Furnishings": ["TPX", "LEG", "HOFT", "LZB", "ETD"],
        "Homebuilding": ["DHI", "LEN", "PHM", "NVR", "TOL"],
        "Household Appliances": ["WHR", "IRBT", "NRE", "KAER", "SEB"],
        "Housewares & Specialties": ["NWL", "TUP", "WDFC", "HELE", "OXM"],
        "Leisure Products": ["HAS", "MAT", "BC", "PLOS", "YETI"],
        "Apparel, Accessories & Luxury Goods": ["NKE", "RL", "PVH", "TPR", "VFC"],
        "Footwear": ["NKE", "DECK", "SKX", "CROX", "WWW"],
        "Textiles": ["MHK", "VFC", "UFI", "CRI", "GIL"],
        "Casinos & Gaming": ["LVS", "MGM", "WYNN", "CZR", "PENN"],
        "Hotels, Resorts & Cruise Lines": ["MAR", "HLT", "RCL", "CCL", "H"],
        "Leisure Facilities": ["FUN", "SIX", "SEAS", "PLYA", "RCI"],
        "Restaurants": ["MCD", "SBUX", "CMG", "YUM", "DRI"],
        "Education Services": ["LOPE", "STRA", "PRDO", "TAL", "GOTU"],
        "Specialized Consumer Services": ["HRB", "BFAM", "WW", "MED", "SCI"],
        "Distributors": ["LKQ", "GPC", "POOL", "DORM", "WCC"],
        "Broadline Retail": ["AMZN", "BABA", "PDD", "EBAY", "MELI"],
        "Apparel Retail": ["TJX", "ROST", "LULU", "GPS", "ANF"],
        "Computer & Electronics Retail": ["BBY", "GME", "CONN", "CRSR", "ORIT"],
        "Home Improvement Retail": ["HD", "LOW", "LL", "TSCO", "HVT"],
        "Other Specialty Retail": ["ORLY", "AZO", "TSCO", "ULTA", "BRE"],
        "Automotive Retail": ["KMX", "AN", "PAG", "LAD", "SAH"],
        "Homefurnishing Retail": ["WSM", "RH", "BBBY", "ETH", "HAV"]
    },
    "Consumer Staples": {
        "Drug Retail": ["WBA", "CVS", "RAD", "GNC", "MED"],
        "Food Distributors": ["SYY", "USFD", "PFGC", "UNFI", "CHEF"],
        "Food Retail": ["KR", "SFM", "WFM", "ASDA", "TESCO"],
        "Consumer Staples Merchandise Retail": ["WMT", "COST", "TGT", "DG", "DLTR"],
        "Brewers": ["BUD", "TAP", "SAM", "HEINY", "CARLB"],
        "Distillers & Vintners": ["DEO", "STZ", "BF.B", "RI", "LVMH"],
        "Soft Drinks & Non-alcoholic Beverages": ["KO", "PEP", "MNST", "KDP", "CELH"],
        "Agricultural Products & Services": ["ADM", "BG", "DAR", "INGR", "ALCO"],
        "Packaged Foods & Meats": ["MDLZ", "GIS", "KHC", "K", "HRL"],
        "Tobacco": ["PM", "MO", "BTI", "UVV", "VGR"],
        "Household Products": ["PG", "CL", "CHD", "CLX", "RECK"],
        "Personal Care Products": ["EL", "COTY", "ELF", "KVUE", "UTHR"]
    },
    "Health Care": {
        "Health Care Equipment": ["MDT", "ABT", "SYK", "EW", "BSX"],
        "Health Care Supplies": ["BAX", "BDX", "COO", "HSIC", "XRAY"],
        "Health Care Distributors": ["MCK", "CAH", "HSIC", "ABC", "COR"],
        "Health Care Services": ["A", "LH", "DGX", "CRL", "MEDP"],
        "Health Care Facilities": ["HCA", "THC", "UHS", "ACHC", "ENSG"],
        "Managed Health Care": ["UNH", "ELV", "CI", "CVS", "CNC"],
        "Health Care Technology": ["VEEV", "CERT", "NXGN", "DH", "DOX"],
        "Biotechnology": ["ABBV", "AMGN", "VRTX", "REGN", "GILD"],
        "Life Sciences Tools & Services": ["TMO", "DHR", "A", "ILMN", "WAT"],
        "Pharmaceuticals": ["LLY", "JNJ", "MRK", "PFE", "NVO"]
    },
    "Financials": {
        "Diversified Banks": ["JPM", "BAC", "C", "WFC", "HSBC"],
        "Regional Banks": ["PNC", "TFC", "USB", "FITB", "KEY"],
        "Diversified Financial Services": ["BRK.B", "AXP", "CIT", "VMC", "NFP"],
        "Multi-Sector Holdings": ["BRK.A", "JARD", "FRFHF", "IEP", "L"],
        "Specialized Finance": ["AER", "OMF", "TROW", "BEN", "IVZ"],
        "Commercial & Residential Mortgage Finance": ["RKT", "UWMC", "LDI", "PFSI", "COOP"],
        "Transaction & Payment Processing Services": ["V", "MA", "FI", "PYPL", "GPN"],
        "Consumer Finance": ["COF", "DFS", "SYF", "ALLY", "SOFI"],
        "Asset Management & Custody Banks": ["BLK", "BK", "STT", "NTRS", "TROW"],
        "Investment Banking & Brokerage": ["GS", "MS", "SCHW", "RJF", "IBKR"],
        "Diversified Capital Markets": ["MS", "GS", "SCHW", "LPLA", "AMP"],
        "Financial Exchanges & Data": ["SPGI", "CME", "ICE", "MSCI", "NDAQ"],
        "Mortgage REITs": ["NLY", "AGNC", "STWD", "BXMT", "RWT"],
        "Insurance Brokers": ["MMC", "AON", "AJG", "WTW", "BRO"],
        "Life & Health Insurance": ["MET", "PRU", "AFL", "LNC", "GL"],
        "Multi-line Insurance": ["AIG", "HIG", "ALL", "CINF", "AIZ"],
        "Property & Casualty Insurance": ["PGR", "CB", "TRV", "WRB", "CINF"],
        "Reinsurance": ["RE", "RNR", "RGA", "MUV2", "SREN"]
    },
    "Information Technology": {
        "IT Consulting & Other Services": ["ACN", "INFY", "WIT", "CTSH", "IBM"],
        "Internet Services & Infrastructure": ["NET", "OKTA", "AKAM", "VRSN", "FSLY"],
        "Application Software": ["CRM", "INTU", "NOW", "CDNS", "SNPS"],
        "Systems Software": ["MSFT", "ORCL", "PANW", "FTNT", "CRWD"],
        "Communications Equipment": ["CSCO", "ANET", "MSI", "JNPR", "CIEN"],
        "Technology Hardware, Storage & Peripherals": ["AAPL", "HPQ", "DELL", "WDC", "STX"],
        "Electronic Equipment & Instruments": ["KEYS", "TRMB", "TDY", "ZBRA", "FLIR"],
        "Electronic Components": ["APH", "TEL", "GLW", "JBL", "SANM"],
        "Electronic Manufacturing Services": ["FLEX", "JBL", "SANM", "CLS", "TTMI"],
        "Technology Distributors": ["SNX", "WCC", "AVT", "IM", "ARROW"],
        "Semiconductor Materials & Equipment": ["AMAT", "LRCX", "KLAC", "ASML", "TER"],
        "Semiconductors": ["NVDA", "TSM", "AVGO", "AMD", "TXN"]
    },
    "Communication Services": {
        "Alternative Carriers": ["LUMN", "GOGO", "IRDM", "ORAN", "VOD"],
        "Integrated Telecommunication Services": ["T", "VZ", "CMCSA", "CHTR", "FYBR"],
        "Wireless Telecommunication Services": ["TMUS", "USM", "S", "AMX", "VOD"],
        "Advertising": ["OMC", "IPG", "WPP", "PUB", "STER"],
        "Broadcasting": ["NXST", "GTN", "SBGI", "TGNA", "AMCX"],
        "Publishing": ["NWSA", "NYT", "GCI", "SCHL", "WLY"],
        "Movies & Entertainment": ["NFLX", "DIS", "WBD", "PARA", "CNK"],
        "Interactive Home Entertainment": ["EA", "TTWO", "NTES", "SONY", "RBLX"],
        "Interactive Media & Services": ["GOOGL", "META", "SNAP", "PINS", "MTCH"]
    },
    "Utilities": {
        "Electric Utilities": ["NEE", "DUK", "SO", "AEP", "EIX"],
        "Gas Utilities": ["ATO", "NI", "SWX", "SR", "NJR"],
        "Multi-Utilities": ["SRE", "PEG", "WEC", "AWK", "ES"],
        "Water Utilities": ["AWK", "WTRG", "SJW", "AWR", "CWT"],
        "Independent Power Producers & Energy Traders": ["CEG", "VST", "NRG", "AES", "BIP"],
        "Renewable Electricity": ["NEE", "BEP", "HASI", "RUN", "ENPH"]
    },
    "Real Estate": {
        "Diversified Real Estate Activities": ["JOE", "HOV", "FRPH", "TPL", "AWI"],
        "Real Estate Operating Companies": ["CBRE", "JLL", "CWK", "NMRK", "HHC"],
        "Real Estate Development": ["JOE", "MDV", "MMI", "TPL", "GPK"],
        "Real Estate Services": ["CBRE", "JLL", "Z", "RDFN", "EXPI"],
        "Industrial REITs": ["PLD", "FR", "EGP", "STAG", "REXR"],
        "Hotel & Resort REITs": ["HST", "PK", "RHP", "SHO", "CHSP"],
        "Office REITs": ["BXP", "VNO", "KRC", "HIW", "CUZ"],
        "Health Care REITs": ["WELL", "PEAK", "VTR", "MPW", "SBRA"],
        "Residential REITs": ["EQR", "AVB", "MAA", "ESS", "UDR"],
        "Retail REITs": ["SPG", "KIM", "REG", "FRT", "BRX"],
        "Specialized REITs": ["AMT", "CCI", "EQIX", "DLR", "WY"]
    }
}