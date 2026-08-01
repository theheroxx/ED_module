"""
config.py
=========
Single source of truth for paths and shared parameters.
No hard-coded absolute/OS-specific paths anywhere else in the project.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (all relative to the project root -> portable across machines)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

# Raw input dataset. Drop GlobalWeatherRepository.csv into ./data/
RAW_DATA_PATH = DATA_DIR / "GlobalWeatherRepository.csv"

# Cached, fully-processed dataset (raw features + ED baseline columns).
# Produced ONCE by common.preprocessing.get_processed_data and reused by every
# step. This is the ONLY cross-step handoff artifact -- no step depends on
# another step's *analytical* output (keeps steps decoupled, per the plan).
COMMON_OUT = OUTPUTS_DIR / "common"
PROCESSED_DATA_PATH = COMMON_OUT / "processed_data.parquet"

STEP1_OUT = OUTPUTS_DIR / "step1_association"
STEP2_OUT = OUTPUTS_DIR / "step2_decision_tree"
STEP3_OUT = OUTPUTS_DIR / "step3_clustering"
STEP4_OUT = OUTPUTS_DIR / "step4_correlation"
STEP5_OUT = OUTPUTS_DIR / "step5_anomaly"
MODEL_OUT = OUTPUTS_DIR / "math_model"
GNN_OUT = OUTPUTS_DIR / "gnn_adjuster"

# ----------------------------------------------------------------------------
# ED baseline category bins (continuous score -> label). Used everywhere.
# ----------------------------------------------------------------------------
ED_CATEGORY_BINS = [
    (75, "ED_VERY_DANGEROUS"),
    (55, "ED_DANGEROUS"),
    (35, "ED_CAUTION"),
    (15, "ED_MODERATE_SAFE"),
    (0,  "ED_VERY_SAFE"),
]
ED_ORDINAL = {
    "ED_VERY_SAFE": 0,
    "ED_MODERATE_SAFE": 1,
    "ED_CAUTION": 2,
    "ED_DANGEROUS": 3,
    "ED_VERY_DANGEROUS": 4,
}

# ----------------------------------------------------------------------------
# Algorithm parameters
# ----------------------------------------------------------------------------
ASSOC_MIN_SUPPORT = 0.02
ASSOC_MIN_CONFIDENCE = 0.65
TREE_MAX_DEPTH = 4
TREE_MIN_SAMPLES_LEAF = 30
CLUSTER_MAX_K = 12
RANDOM_STATE = 42

# Location key used to aggregate rows -> per-location nodes for clustering/GNN
LOCATION_KEYS = ["location_name", "country", "latitude", "longitude"]
