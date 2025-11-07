import os
from config import *
from data.loader import load_dataset
from data.preprocess import preprocess
from model.train import train_model
from model.evaluate import evaluate_models, mean_confidence_interval
from explainability.shap_analysis import run_shap

if os.path.exists(DATASET_PATH):
    df = load_dataset(DATASET_PATH)
else:
    df = load_dataset(DATASET_PATH)

X, y = preprocess(df, "class", DISCARTED_COLUMNS)
models = train_model(X, y, XGBOOST_PARAMS, N_SPLITS, RANDOM_STATE)

metrics_summary, accuracy_intervals, kappa_scores = evaluate_models(models, CLASS_NAMES)
mean_kappa, kappa_ci = mean_confidence_interval(kappa_scores)

print("\nMétricas por classe:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"{cls}: F1-score = {metrics_summary['F1-score Mean'][i]:.4f} ± {metrics_summary['F1-score CI'][i]:.4f}, ",
          f" Precision = {metrics_summary['Precision Mean'][i]:.4f} ± {metrics_summary['Precision CI'][i]:.4f}, ",
          f" Recall = {metrics_summary['Recall Mean'][i]:.4f} ± {metrics_summary['Recall CI'][i]:.4f}")
    
print(f"\nCohen's Kappa: {mean_kappa:.4f} ± {kappa_ci:.4f}")

# model_final, X_test, _ = models[-1]
# run_shap(model_final, X_test, CLASS_NAMES, dataset_name="grayhole60", path_base="./results", graphics=GRAPHICS)