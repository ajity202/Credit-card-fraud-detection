from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# DATA PATHS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"


# ============================================================
# REPORT PATHS
# ============================================================

REPORTS_DIR = PROJECT_ROOT / "reports"

FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"


# ============================================================
# PROJECT SETTINGS
# ============================================================

RANDOM_STATE = 42

TARGET_COLUMN = "Class"