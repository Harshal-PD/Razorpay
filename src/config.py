"""Project constants. No silent-deny policy lives here — see src/policy.py."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"

RANDOM_SEED = 42

# UCI Online Retail II
UCI_ID = 502
UCI_XLSX_URLS = (
    "https://cdn.uci-ics-mlr-prod.aws.uci.edu/502/online+retail+ii.zip",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx",
    "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx",
)

# Invoice-level y_real: full-package analog
Y_REAL_FRAC_THRESHOLD = 0.80
MATCH_WINDOW_DAYS = 180

# Temporal split (UCI span Dec 2009–Dec 2011)
BURNIN_END = "2010-03-01"  # [min, burnin_end) history only
TRAIN_END = "2011-09-01"  # [burnin_end, train_end) train
VAL_END = "2011-11-01"  # [train_end, val_end) val
# test: [val_end, inf)  Nov–Dec 2011 festive holdout

NON_PRODUCT_STOCK = frozenset(
    {
        "POST",
        "M",
        "D",
        "DOT",
        "CRUK",
        "PADS",
        "C2",
        "S",
        "B",
        "BANK CHARGES",
        "AMAZONFEE",
        "DCGS",
    }
)

# Eval B cost priors (INR). Documented estimates, not measured on UCI.
C_FN_INR = 210.0  # midpoint of ~₹180–240 forward+reverse+handling
MARGIN_RATE = 0.25
CONVERSION_LOSS_IF_PREPAID = 0.15
VERIFY_COST_INR = 10.0
VERIFY_RTO_REDUCTION = 0.25  # assumption, not a measured lift

# Overlay
OVERLAY_SEED = 42
COD_BASE_RATE = 0.60
AMOUNT_INR_MEDIAN = 1100.0

# Scenario generator pin prior (generator only — not a feature)
PIN_PRIOR = {"metro": 0.0, "tier2": 0.40, "tier3": 0.80}

PIN_PREFIX_TABLE = {
    "metro": ("110001", "400001", "560001", "600001", "700001"),  # DEL, BOM, BLR, MAA, CCU
    "tier2": ("302001", "380001", "411001", "500001", "226001"),  # Jaipur, AMD, Pune, HYD, LKO
    "tier3": ("125001", "248001", "313001", "452001", "842001"),  # Hisar, DDN, Udaipur, Indore, MFP
}

CATEGORY_PROBS = {
    "fashion": 0.40,
    "beauty": 0.20,
    "home": 0.25,
    "electronics": 0.15,
}

FASHION_KEYWORDS = (
    "bag",
    "dress",
    "necklace",
    "jewellery",
    "jewelry",
    "handbag",
    "purse",
    "scarf",
    "sock",
    "hat",
    "glove",
    "belt",
    "ring",
    "earring",
    "bracelet",
    "feltcraft",
    "hairband",
    "t-shirt",
    "tshirt",
    "jacket",
    "skirt",
)

# Eval A proxy cost (no INR COD economics): missed full-return vs extra review
EVAL_A_C_FN = 40.0
EVAL_A_C_FP = 1.0

# Target scenario rates (calibration targets, not India gospel)
TARGET_PREPAID = 0.06
TARGET_COD = 0.30
