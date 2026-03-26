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
PATH_BASE="./results_with_consistency_features"

# ============================================
# XGBOOST MODEL CONFIGURATIONS
# ============================================

# Type of model to train. You can choose between:
#   - 'xgboost' (aliases: 'xgb')
#   - 'random_forest' (aliases: 'rf', 'randomforest')
#   - 'svm' (aliases: 'support_vector_machine', 'supportvectormachine')
#   - 'mlp' (aliases: 'neural_network', 'multilayer_perceptron')
#   - 'decision_tree' (aliases: 'dt', 'decisiontree')
#   - 'logistic_regression' (aliases: 'lr', 'logisticregression')
MODEL_TYPE = "xgboost"

# Map of aliases to facilitate writing model names
MODEL_NAME_ALIASES = {
    # XGBoost
    "xgboost": "xgboost",
    "xgb": "xgboost",
    # Random Forest
    "random_forest": "random_forest",
    "rf": "random_forest",
    "randomforest": "random_forest",
    # SVM
    "svm": "svm",
    "support_vector_machine": "svm",
    "supportvectormachine": "svm",
    # MLP
    "mlp": "mlp",
    "neural_network": "mlp",
    "multilayer_perceptron": "mlp",
    # Decision Tree
    "decision_tree": "decision_tree",
    "dt": "decision_tree",
    "decisiontree": "decision_tree",
    # Logistic Regression
    "logistic_regression": "logistic_regression",
    "lr": "logistic_regression",
    "logisticregression": "logistic_regression",
}

# XGBoost Params (these are the default params - defined in the paper) 
XGBOOST_PARAMS = {
    "objective": 'multi:softprob',  # multi:softprob to multiclass problem
    "eval_metric": 'mlogloss',
    "random_state": RANDOM_STATE,
}

# Random Forest Params
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# SVM Params
SVM_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "probability": True,  
    "random_state": RANDOM_STATE,
}

# MLP params
MLP_PARAMS = {
    "hidden_layer_sizes": (100,),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "learning_rate": "constant",
    "max_iter": 200,
    "random_state": RANDOM_STATE,
}

# DT Params
DT_PARAMS = {
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
}

# Logistic Regression Params
LR_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 500,
    "multi_class": "multinomial",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Map the parameters to their respective model types for easy access during training
MODEL_PARAMS = {
    "xgboost": XGBOOST_PARAMS,
    "random_forest": RF_PARAMS,
    "svm": SVM_PARAMS,
    "mlp": MLP_PARAMS,
    "decision_tree": DT_PARAMS,
    "logistic_regression": LR_PARAMS,
}

# Description for each model type, used in the final report
MODEL_DESCRIPTIONS = {
    "xgboost": "XGBoost (Gradient Boosting)",
    "random_forest": "Random Forest (Ensemble - Baseline Clássico)",
    "svm": "SVM com kernel RBF (Tradicional em IDS)",
    "mlp": "MLP - Neural Network (Baseline Deep Learning)",
    "decision_tree": "Decision Tree (Baseline Interpretável)",
    "logistic_regression": "Logistic Regression (Baseline Linear)",
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
WITHOUT_DELTA_FEATURES = True

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
DATASET_PATH = "./data/CSV files/dataset_downsampled.csv"

# ============================================
# VIEW CONFIGURATION (SHAP)
# ============================================

# SHAP graphs types to generate for explainability analysis.
GRAPHICS = [
    "Bar Plot",                  # Features mean importance (|SHAP|)
    "Beeswarm Summary Plot",     # Dense visualization showing SHAP value vs feature value for each sample
]