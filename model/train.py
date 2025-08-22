import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

def train_model(X, y, params, n_splits, seed):
    """Train an XGBoost model using K-Fold cross-validation.
    
    Args:
        X: DataFrame with features.
        y: Series with target variable.
        params: Dictionary with model parameters.
        n_splits: Number of folds for cross-validation.
        seed: Seed for reproducibility in Kfold object.
    Returns:
        List of trained models and their corresponding test sets.
    """
        
    print("Training model with K-Fold cross-validation...")
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = []
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]   

        model = xgb.XGBClassifier(**params)
        print(f"Training fold with {len(train_idx)} samples for training and {len(test_idx)} samples for testing. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        model.fit(X_train, y_train)
        models.append((model, X_test, y_test))
    return models
