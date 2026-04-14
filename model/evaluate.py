import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score
from scipy.stats import sem, t
from datetime import datetime
import os
from config import MODEL_TYPE, WITHOUT_DELTA_FEATURES
from model.matrix_confusion import plot_confusion_matrix

def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate the mean and confidence interval of a dataset.
    """
    n = len(data)
    m = np.mean(data)
    se = sem(data)

    if np.isnan(se) or se == 0:
        return m, 0.0

    h = se * t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def evaluate_models(cv_models, class_names):
    """
    Evaluate multiple models from cross-validation and calculate metrics with confidence intervals.

    Returns:
        tuple: (cv_metrics_summary, kappa_mean, kappa_ci, cv_total_cm, all_y_true, all_y_pred)
            - all_y_true: vetor com os labels reais de todos os folds (para McNemar)
            - all_y_pred: vetor com as predições de todos os folds (para McNemar)
    """

    print("=" * 60)
    print("EVALUATION: Metrics from Cross-Validation")
    print("=" * 60)

    accuracy_scores = []
    kappa_scores = []
    precision_scores = {cls: [] for cls in class_names}
    recall_scores = {cls: [] for cls in class_names}
    f1_scores = {cls: [] for cls in class_names}

    n_classes = len(class_names)
    cv_total_cm = np.zeros((n_classes, n_classes), dtype=int)

    # Acumulate all of the predictions of the all fold to McNemar test acumula
    all_y_true = []
    all_y_pred = []

    for fold_num, (model, X_val, y_val) in enumerate(cv_models, 1):
        y_pred = model.predict(X_val)

        # Acumulate vectors from this fold
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
        kappa = cohen_kappa_score(y_val, y_pred)
        kappa_scores.append(kappa)

        # Confusion matrix for this fold
        cm = confusion_matrix(y_val, y_pred)
        try:
            cv_total_cm += cm
        except Exception:
            cv_total_cm = cv_total_cm + np.asarray(cm, dtype=int)

        acc = accuracy_score(y_val, y_pred)
        accuracy_scores.append(acc)

        for cls in class_names:
            precision_scores[cls].append(report[cls]['precision'])
            recall_scores[cls].append(report[cls]['recall'])
            f1_scores[cls].append(report[cls]['f1-score'])

        print(f"Fold {fold_num}: Kappa = {kappa:.4f}")

    # Convert accumulated lists to numpy arrays for easier handling
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    accuracy_scores = np.array(accuracy_scores)
    acc_mean, acc_ci = mean_confidence_interval(accuracy_scores)
    kappa_mean, kappa_ci = mean_confidence_interval(kappa_scores)

    print(f"\n✓ Cohen's Kappa (CV): {kappa_mean:.4f} ± {kappa_ci:.4f}")
    print(f"✓ Predictions collected for McNemar: {len(all_y_true)} samples")
    print()

    print("Aggregated Confusion Matrix (CV - sum of all folds):")
    print(cv_total_cm)
    print()

    cv_metrics_summary = {
        "Classe": [],
        "Precision Mean": [], "Precision CI": [],
        "Recall Mean": [], "Recall CI": [],
        "F1-score Mean": [], "F1-score CI": []
    }

    # Calculate mean and confidence intervals for each class and metric
    for i, cls in enumerate(class_names):
        pm, pci = mean_confidence_interval(precision_scores[cls])
        rm, rci = mean_confidence_interval(recall_scores[cls])
        f1m, f1ci = mean_confidence_interval(f1_scores[cls])

        cv_metrics_summary["Classe"].append(cls)
        cv_metrics_summary["Precision Mean"].append(pm)
        cv_metrics_summary["Precision CI"].append(pci)
        cv_metrics_summary["Recall Mean"].append(rm)
        cv_metrics_summary["Recall CI"].append(rci)
        cv_metrics_summary["F1-score Mean"].append(f1m)
        cv_metrics_summary["F1-score CI"].append(f1ci)

    cv_metrics_summary["Global Accuracy Mean"] = acc_mean
    cv_metrics_summary["Global Accuracy CI"] = acc_ci

    return cv_metrics_summary, kappa_mean, kappa_ci, cv_total_cm, all_y_true, all_y_pred


def save_metrics_report(cv_metrics, kappa_mean, kappa_ci, class_names, dataset_name,
                        output_dir="./results", cv_total_cm=None,
                        all_y_true=None, all_y_pred=None): 
    """
    Save the evaluation metrics in both Markdown and Log formats, and also save the confusion matrix as an SVG.
    """

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_filename  = f"metrics_report_{dataset_name}_{timestamp}.md"
    log_filename = f"metrics_report_{dataset_name}_{timestamp}.log"
    md_path  = os.path.join(output_dir, md_filename)
    log_path = os.path.join(output_dir, log_filename)

    # Save the true and predicted labels for McNemar test 
    if all_y_true is not None and all_y_pred is not None:
        scenario = "without_delta" if WITHOUT_DELTA_FEATURES else "with_delta"
        npy_true_path = os.path.join(output_dir, f"y_true_{dataset_name}.npy")
        npy_pred_path = os.path.join(output_dir, f"y_pred_{scenario}_{dataset_name}.npy")

        # y_true only needs to be saved once
        if not os.path.exists(npy_true_path):
            np.save(npy_true_path, all_y_true)
            print(f"✓ y_true saved: {npy_true_path}")
        else:
            print(f"✓ y_true already exists, not overwritten: {npy_true_path}")

        np.save(npy_pred_path, all_y_pred)
        print(f"✓ y_pred ({scenario}) saved: {npy_pred_path}")

    # ==============================
    # MARKDOWN REPORT 
    # ==============================

    md_content = f"""# Metrics Report - {MODEL_TYPE.upper()}

**Dataset:** {dataset_name}
**Date/Time:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

---

## 📊 Cross-Validation (K-Fold)

Results of cross-validation with **95% confidence intervals** (CI 95%).

### Metrics by Class

"""
    md_content += "| Class | F1-Score | Precision | Recall |\n"
    md_content += "|--------|----------|-----------|--------|\n"

    for i, cls in enumerate(class_names):
        f1_mean  = cv_metrics['F1-score Mean'][i]
        f1_ci    = cv_metrics['F1-score CI'][i]
        prec_mean = cv_metrics['Precision Mean'][i]
        prec_ci   = cv_metrics['Precision CI'][i]
        rec_mean  = cv_metrics['Recall Mean'][i]
        rec_ci    = cv_metrics['Recall CI'][i]
        md_content += f"| **{cls}** | {f1_mean:.4f} ± {f1_ci:.4f} | {prec_mean:.4f} ± {prec_ci:.4f} | {rec_mean:.4f} ± {rec_ci:.4f} |\n"

    md_content += f"\n### Global Metrics (CV)\n\n"
    md_content += f"- **Accuracy (CV - Mean ± CI):** {cv_metrics['Global Accuracy Mean']:.4f} ± {cv_metrics['Global Accuracy CI']:.4f}\n"
    md_content += f"- **Cohen's Kappa:** {kappa_mean:.4f} ± {kappa_ci:.4f}\n"
    md_content += f"\n---\n\n"

    if cv_total_cm is not None:
        md_content += "## 🧾 Confusion Matrix (CV - Aggregated)\n\n"
        md_content += "```\nPredicted →    "
        for cls in class_names:
            md_content += f"{cls:>12} "
        md_content += "\nReal ↓\n"
        for i, cls in enumerate(class_names):
            md_content += f"{cls:12} "
            for j in range(len(class_names)):
                md_content += f"{cv_total_cm[i][j]:>12} "
            md_content += "\n"
        md_content += "```\n\n"

    md_content += f"---\n\n## 📈 Interpretation\n\n"
    best_class_idx = np.argmax([cv_metrics['F1-score Mean'][i] for i in range(len(class_names))])
    best_class = class_names[best_class_idx]
    best_f1 = cv_metrics['F1-score Mean'][best_class_idx]
    md_content += f"- **Best Performance (CV):** Class `{best_class}` with F1-Score of **{best_f1:.4f}**\n"

    if kappa_mean > 0.8:
        kappa_interp = "Agreement **almost perfect**"
    elif kappa_mean > 0.6:
        kappa_interp = "Agreement **substantial**"
    elif kappa_mean > 0.4:
        kappa_interp = "Agreement **moderate**"
    elif kappa_mean > 0.2:
        kappa_interp = "Agreement **weak**"
    else:
        kappa_interp = "Agreement **poor**"

    md_content += f"- **Cohen's Kappa (CV):** {kappa_interp} ({kappa_mean:.4f} ± {kappa_ci:.4f})\n"
    md_content += f"\n---\n\n"
    md_content += f"*Report generated automatically by the {MODEL_TYPE.upper()} training pipeline*\n"

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    # ==============================
    # LOG REPORT 
    # ==============================

    log_content = f"""{'='*80}
    REPORT OF METRICS - {MODEL_TYPE.upper()}
    {'='*80}

    Dataset: {dataset_name}
    Date/Time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

    {'='*80}
    CROSS-VALIDATION (K-Fold) - Mean ± 95% CI
    {'='*80}

    """
    for i, cls in enumerate(class_names):
        log_content += f"\nClass: {cls}\n"
        log_content += f"  F1-Score:  {cv_metrics['F1-score Mean'][i]:.4f} ± {cv_metrics['F1-score CI'][i]:.4f}\n"
        log_content += f"  Precision: {cv_metrics['Precision Mean'][i]:.4f} ± {cv_metrics['Precision CI'][i]:.4f}\n"
        log_content += f"  Recall:    {cv_metrics['Recall Mean'][i]:.4f} ± {cv_metrics['Recall CI'][i]:.4f}\n"

    log_content += f"\nCohen's Kappa (CV): {kappa_mean:.4f} ± {kappa_ci:.4f}\n"
    log_content += f"Accuracy (CV - Mean ± CI): {cv_metrics['Global Accuracy Mean']:.4f} ± {cv_metrics['Global Accuracy CI']:.4f}\n"

    if cv_total_cm is not None:
        log_content += f"\n{'-'*80}\n"
        log_content += "CONFUSION MATRIX (CV - Aggregated)\n"
        log_content += f"{'-'*80}\n\n"
        log_content += "Predicted →    "
        for cls in class_names:
            log_content += f"{cls:>12} "
        log_content += "\nReal ↓\n"
        for i, cls in enumerate(class_names):
            log_content += f"{cls:12} "
            for j in range(len(class_names)):
                log_content += f"{cv_total_cm[i][j]:>12} "
            log_content += "\n"
        log_content += f"\n{'='*80}\n"

    log_content += f"\n{'='*80}\n"
    log_content += f"Report saved at: {md_path}\n"
    log_content += f"{'='*80}\n"

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(log_content)

    if cv_total_cm is not None:
        try:
            svg_filename = f"confusion_matrix_{dataset_name}_{timestamp}.svg"
            svg_path = os.path.join(output_dir, svg_filename)
            if WITHOUT_DELTA_FEATURES == True:
                delta_flag = True
            else:
                delta_flag = False
            plot_confusion_matrix(cv_total_cm, class_names, svg_path, delta_features=delta_flag)
            print(f"Saved confusion matrix SVG: {svg_path}")
        except Exception as e:
            print(f"Warning: failed to generate confusion matrix SVG: {e}")

    return md_path, log_path