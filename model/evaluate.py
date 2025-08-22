import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from scipy.stats import sem, t

def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, h

def evaluate_models(models, class_names):
    """
    Function to evaluate multiple models and return a summary of metrics.
    Args:
        models (list): List of tuples containing model, X_test, and y_test.
        class_names (list): List of class names for the classification task.
    Returns:
        metrics_summary (dict): Summary of metrics including precision, recall, and F1-score.
        accuracy_intervals (list): Confidence intervals for accuracy scores.
        kappa_scores (list): List of Cohen's Kappa scores for each model.
    """
    print("Calculating metrics for each class...")
    
    accuracy_scores = []
    kappa_scores = []
    precision_scores = {cls: [] for cls in class_names}
    recall_scores = {cls: [] for cls in class_names}
    f1_scores = {cls: [] for cls in class_names}

    for (model, X_test, y_test) in models:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        kappa_scores.append(cohen_kappa_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        class_acc = cm.diagonal() / cm.sum(axis=1)
        accuracy_scores.append(class_acc)

        for cls in class_names:
            precision_scores[cls].append(report[cls]['precision'])
            recall_scores[cls].append(report[cls]['recall'])
            f1_scores[cls].append(report[cls]['f1-score'])

    print(f"kohen_kappa_scores: {kappa_scores}")
    print(report)
    accuracy_scores = np.array(accuracy_scores)
    accuracy_conf_intervals = [mean_confidence_interval(accuracy_scores[:, i]) for i in range(len(class_names))]

    metrics_summary = {
        "Classe": [], "Precision Mean": [], "Precision CI": [],
        "Recall Mean": [], "Recall CI": [],
        "F1-score Mean": [], "F1-score CI": []
    }

    for i, cls in enumerate(class_names):
        pm, pci = mean_confidence_interval(precision_scores[cls])
        rm, rci = mean_confidence_interval(recall_scores[cls])
        f1m, f1ci = mean_confidence_interval(f1_scores[cls])
        metrics_summary["Classe"].append(cls)
        metrics_summary["Precision Mean"].append(pm)
        metrics_summary["Precision CI"].append(pci)
        metrics_summary["Recall Mean"].append(rm)
        metrics_summary["Recall CI"].append(rci)
        metrics_summary["F1-score Mean"].append(f1m)
        metrics_summary["F1-score CI"].append(f1ci)

    return metrics_summary, accuracy_conf_intervals, kappa_scores