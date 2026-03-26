"""
Complete training and evaluation pipeline for ML model

This script implements the best academic practices:
1. Stratified K-Fold cross-validation
2. Metric calculation with confidence intervals
3. Comprehensive reporting (Markdown + Log)
4. SHAP explainability analysis
5. Optimizations for handling large datasets
"""

import os
from config import *
from data.loader import load_dataset
from data.preprocess import preprocess
from model.train import train_model
from model.evaluate import evaluate_models, save_metrics_report
from explainability.shap_analysis import run_shap

def main():
    """
    Main function that orchestrates the entire pipeline:
    """

    print("\n" + "=" * 60)
    print(f"TRAINING AND EVALUATION PIPELINE - {MODEL_TYPE.upper()}")
    print("=" * 60)
    print()

    # ========================================
    # STEP 1: Dataset Loading
    # ========================================

    print("=" * 60)
    print("STEP 1: Dataset Loading")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    print(f"Dataset loading: {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)
    print(f"✓ Dataset load: {df.shape[0]} lines × {df.shape[1]} columns")
    print()

    # ========================================
    # STEP 2: Pré-processing
    # ========================================

    print("=" * 60)
    print("STEP 2: Pré-processing")
    print("=" * 60)

    # Separate features (X) and target (y)
    X, y, encoder = preprocess(df, target_column="class", discarted_columns=DISCARTED_COLUMNS)

    print(f"✓ Features: {X.shape[1]} columns")
    print(f"✓ Samples: {len(y)}")
    print(f"✓ Classes: {CLASS_NAMES}")
    print(f"✓ Class Distribution:")

    # Show the class distribution
    for i, cls in enumerate(CLASS_NAMES):
        count = (y == i).sum()
        percentage = (count / len(y)) * 100
        print(f"    - {cls}: {count} samples ({percentage:.1f}%)")
    print()

    # Free memory by deleting the original dataframe
    del df

    # ========================================
    # STEP 3: Training
    # ========================================

    # Show the chosen model and its description
    model_desc = MODEL_DESCRIPTIONS.get(MODEL_PARAMS.__class__.__name__, "")
    canonical_model = MODEL_NAME_ALIASES.get(MODEL_TYPE.lower(), MODEL_TYPE)
    print("=" * 60)
    print(f"CHOOSEN MODEL: {canonical_model.upper()}")
    print(f"Description: {MODEL_DESCRIPTIONS.get(canonical_model, canonical_model)}")
    print("=" * 60)
    print()

    # Train models using Cross-Validation K-Fold 
    cv_models, final_model = train_model(
        X, y,
        model_type=MODEL_TYPE,
        params=MODEL_PARAMS.get(canonical_model, XGBOOST_PARAMS),
        n_splits=N_SPLITS,
        seed=RANDOM_STATE
    )

    # ========================================
    # STEP 4: Evaluation
    # ========================================

    # Evaluate models in CV
    cv_metrics, kappa_mean, kappa_ci, cv_total_cm, all_y_true, all_y_pred = evaluate_models(
        cv_models, CLASS_NAMES
    )

    # ========================================
    # STEP 5: Final Results Summary
    # ========================================

    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print()

    print("📊 CROSS-VALIDATION (Mean ± 95% CI):")
    print("-" * 60)
    for i, cls in enumerate(CLASS_NAMES):
        print(f"\n{cls}:")
        print(f"  F1-score:  {cv_metrics['F1-score Mean'][i]:.4f} ± {cv_metrics['F1-score CI'][i]:.4f}")
        print(f"  Precision: {cv_metrics['Precision Mean'][i]:.4f} ± {cv_metrics['Precision CI'][i]:.4f}")
        print(f"  Recall:    {cv_metrics['Recall Mean'][i]:.4f} ± {cv_metrics['Recall CI'][i]:.4f}")

    # Show the global accuracy  and Cohen's Kappa metrics (Mean ± IC)
    if 'Global Accuracy Mean' in cv_metrics and 'Global Accuracy CI' in cv_metrics:
        print(f"\nGlobal Accuracy (CV): {cv_metrics['Global Accuracy Mean']:.4f} ± {cv_metrics['Global Accuracy CI']:.4f}")

    print(f"\nCohen's Kappa (CV): {kappa_mean:.4f} ± {kappa_ci:.4f}")
    print()

    # ========================================
    # STEP 5.1: Save Metrics Report
    # ========================================

    print("=" * 60)
    print("SAVING REPORTS")
    print("=" * 60)
    print()

    # Extract the dataset name from the path
    dataset_name = os.path.basename(DATASET_PATH).replace(".csv", "").replace(".parquet", "")

    # Save reports in Markdown and Log
    md_path, log_path = save_metrics_report(
        cv_metrics, kappa_mean, kappa_ci, CLASS_NAMES, dataset_name, output_dir=PATH_BASE, cv_total_cm=cv_total_cm,
        all_y_true=all_y_true, all_y_pred=all_y_pred
    )

    print(f"✓ Markdown report saved: {md_path}")
    print(f"✓ Log report saved: {log_path}")
    print()

    # ========================================
    # STEP 6: Explainability (SHAP)
    # ========================================

    print("=" * 60)
    print("STEP 6: Explainability Analysis (SHAP)")
    print("=" * 60)
    print()
    

    # Preparação para SHAP: usar o ÚLTIMO conjunto de validação do K-Fold
    if cv_models and len(cv_models) > 0:
        # cv_models é lista de tuplas (model, X_val, y_val)
        _, shap_X, _ = cv_models[-1]
        print(f"Using last fold  to validate to SHAP: {len(shap_X)} samples")
    else:
        shap_X = X
        print("Warning: cv_models is empty, using entire dataset for SHAP (not recommended for large datasets)")
        
    # This function runs the SHAP analysis and saves the plots in the specified path.
    run_shap(
        model=final_model,
        X_test=shap_X,
        class_names=CLASS_NAMES,
        dataset_name=dataset_name,
        path_base=PATH_BASE,
        graphics=GRAPHICS
    )

    print("=" * 60)
    print("✓ SUCCESSFULLY PIPELINE COMPLETED!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()