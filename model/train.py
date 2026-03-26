import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import gc


def _get_classifier(model_type, params):
    """
    Return the configured classifier based on the provided model type and parameters.
    Args:
        model_type (str): The type of classifier to create (e.g., 'xgboost', 'random_forest', 'svm', 'mlp', 'decision_tree', 'logistic_regression').
        params (dict): A dictionary of parameters to initialize the classifier.
    """
    from config import MODEL_NAME_ALIASES
    
    # Normaliza the name: remove leading/trailing whitespace and convert to lowercase
    model_type_normalized = str(model_type).strip().lower()
    
    # Resolve the alias to get the canonical model name
    if model_type_normalized in MODEL_NAME_ALIASES:
        model_type_canonical = MODEL_NAME_ALIASES[model_type_normalized]
    else:
        raise ValueError(
            f"Model not supported: '{model_type}'. "
            f"Supported models: {', '.join(MODEL_NAME_ALIASES.values())}"
        )
    
    match model_type_canonical:
        case "xgboost":
            return xgb.XGBClassifier(**params)
        case "random_forest":
            return RandomForestClassifier(**params)
        case "svm":
            # SVM is sensable at scale, so we use a pipeline with StandardScaler
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(**params))
            ])
        case "mlp":
            # MLP is sensable at scale, so we use a pipeline with StandardScaler too
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(**params))
            ])
        case "decision_tree":
            return DecisionTreeClassifier(**params)
        case "logistic_regression":
            # LR is sensable at scale, so we use a pipeline with StandardScaler too
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**params))
            ])
        case _:
            raise ValueError(f"Model not supported: {model_type_canonical}")

def train_model(X, y, model_type, params, n_splits, seed):
    """
    Train a classifier (XGBoost or chosen baseline) using K-Fold Cross-Validation.

    Args:
        X (pd.DataFrame): DataFrame with the features.
        y (np.array): Array with the target variable (already encoded).
        model_type (str): Type of classifier (xgboost, random_forest, svm, mlp, decision_tree, logistic_regression).
        params (dict): Dictionary with the parameters of the chosen classifier.
        n_splits (int): NNumber of folds for cross-validation.
        seed (int): Seed for reproducibility.

    Returns:
        tuple: (cv_models, final_model)
            - cv_models: List of tuples (model, X_val, y_val) of the validation folds
            - final_model: Final model trained on all the data
    """
    print("=" * 60)
    print(f"STEP: K-Fold Cross-Validation ({n_splits} folds)")
    print("=" * 60)

    # Make K-fold split with stratification to maintain class distribution in each fold
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_models = []

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        # Split the data for this fold
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        print(f"\nTraining fold {fold_num}/{n_splits}...")
        print(f"  - Training: {len(X_train_fold)} samples")
        print(f"  - Validation: {len(X_val_fold)} samples")

        # Train the model for this fold
        model = _get_classifier(model_type, params)
        
        print("  - Training model...")
        model.fit(X_train_fold, y_train_fold)

        # Put the model and validation data in the list for later evaluation
        cv_models.append((model, X_val_fold, y_val_fold))

        # Clean up memory after each fold
        del X_train_fold, y_train_fold
        gc.collect()

    print("\n✓ Cross-Validation completed!")
    print()

    print("=" * 60)
    print("STEP 3: Final model training with all data (to shap values and production)")
    print("=" * 60)
    
    print(f"Training final model with all {len(X)} samples...")
    final_model = _get_classifier(model_type, params)
    final_model.fit(X, y)
    print("✓ Final model trained!")
    print()

    return cv_models, final_model