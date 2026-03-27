# ============================================
# EXPERIMENTS GENERAL CONFIGURATIONS
# ============================================

# Seed to reproducibility (important for train/test split, K-Fold, model randomness, etc)
RANDOM_STATE = 42

# Number of folds for cross-validation (K-Fold)
N_SPLITS = 5

# ============================================
# PERFORMANCE CONFIGURATIONS
# ============================================

# Base path to save the results (metrics, plots, etc)
# Each experiment can have a different path to avoid overwriting results
PATH_BASE="./results_with_delta_features"

# ============================================
# XGBOOST MODEL CONFIGURATIONS
# ============================================

# Type of model to train.
#   - 'xgboost' (aliases: 'xgb')
MODEL_TYPE = "xgboost"

# Map of aliases to facilitate writing model
MODEL_NAME_ALIASES = {
    "xgboost": "xgboost",
    "xgb": "xgboost",
}

# XGBoost Params (these are the default params - defined in the paper) 
XGBOOST_PARAMS = {
    "objective": 'multi:softprob',  # multi:softprob to multiclass problem
    "eval_metric": 'mlogloss',
    "random_state": RANDOM_STATE,
}

# Map the parameters to the xgboost
MODEL_PARAMS = {
    "xgboost": XGBOOST_PARAMS,
}

# Description for the model to the report
MODEL_DESCRIPTIONS = {
    "xgboost": "XGBoost (Gradient Boosting)",
}

# ============================================
# DATASET CONFIGURATIONS
# ============================================

# Columns to discard from the dataset (non-informative or potentially leaking features)
# The class column is also discarded from features and used as target (y)
DISCARTED_COLUMNS = [
    'ethDst', 'ethSrc', 'gocbRef', 'datSet', 'goID', 'test',
    'ndsCom', 'protocol', 'ethType', 'TPID', 'gooseAppid', 'class'
]

# Remove delta features
# This features can leak attack informations
WITHOUT_DELTA_FEATURES = False

# If True, the delta features will be discarded from the dataset and not used for training or SHAP analysis
if WITHOUT_DELTA_FEATURES:
    DISCARD_FEATURES = [
    'stDiff', 'sqDiff', 'gooseLengthDiff', 'cbStatusDiff', 'apduSizeDiff',
    'frameLengthDiff', 'timestampDiff', 'tDiff', 'timeFromLastChange',
]
    PATH_BASE="./results_without_delta_features"
    DISCARTED_COLUMNS.extend(DISCARD_FEATURES)

# Classes names (these should correspond to the encoded classes in the target variable y)
# The order of the class names should match the order of the encoded classes (0, 1, 2, etc.)
CLASS_NAMES = ["SAG.DB", "FRG", "SAG.PB", "SAG.PBM", "Normal"]

# Dataset path
DATASET_PATH = "./data/CSV files/gray-GOOSE.csv"

# ============================================
# VIEW CONFIGURATION (SHAP)
# ============================================

# SHAP graphs types to generate for explainability analysis.
GRAPHICS = [
    "Bar Plot",                  # Features mean importance (|SHAP|)
    "Beeswarm Summary Plot",     # Dense visualization showing SHAP value vs feature value for each sample
]