import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score
from scipy.stats import sem, t
from datetime import datetime
import os
from config import MODEL_TYPE

def mean_confidence_interval(data, confidence=0.95):
    """
    Calcula a média e o intervalo de confiança de um conjunto de dados.

    Args:
        data (list/array): Dados pra calcular a média e IC.
        confidence (float): Nível de confiança (padrão: 0.95 = 95%).

    Returns:
        tuple: (mean, margin_of_error)
            - mean: Média dos dados
            - margin_of_error: Margem de erro (metade do IC)
    """
    n = len(data)
    m = np.mean(data)
    se = sem(data)  # Erro padrão da média

    # Tratamento especial: se todos os valores são idênticos (sem = 0 ou nan)
    # isso acontece quando o modelo tem desempenho perfeito em todos os folds
    if np.isnan(se) or se == 0:
        return m, 0.0

    h = se * t.ppf((1 + confidence) / 2., n-1)  # Margem de erro usando t de Student
    return m, h


def evaluate_models(cv_models, final_model, class_names):
    """
    Avalia múltiplos modelos da validação cruzada.

    Args:
        cv_models (list): Lista de tuplas (model, X_val, y_val) dos folds de CV.
        final_model: Modelo final treinado em todos os dados.
        class_names (list): Lista com os nomes das classes.

    Returns:
        tuple: (cv_metrics_summary, kappa_mean, kappa_ci, cv_total_cm)
            - cv_metrics_summary: Métricas da validação cruzada (mean ± CI por classe)
            - kappa_mean: Média do Cohen's Kappa nos folds de CV
            - kappa_ci: Intervalo de confiança do Cohen's Kappa
            - cv_total_cm: Matriz de confusão agregada (soma dos folds)
    """

    print("=" * 60)
    print("AVALIAÇÃO: Métricas da Validação Cruzada")
    print("=" * 60)

    # ========================================
    # PARTE 1: Métricas da Validação Cruzada
    # ========================================

    # Coleta as métricas de cada fold
    accuracy_scores = []
    kappa_scores = []
    precision_scores = {cls: [] for cls in class_names}
    recall_scores = {cls: [] for cls in class_names}
    f1_scores = {cls: [] for cls in class_names}
    # Matriz de confusão agregada (soma das matrizes dos folds)
    n_classes = len(class_names)
    cv_total_cm = np.zeros((n_classes, n_classes), dtype=int)

    for fold_num, (model, X_val, y_val) in enumerate(cv_models, 1):
        # Prediz no conjunto de validação desse fold
        y_pred = model.predict(X_val)

        # Calcula report com precision, recall e f1 por classe
        report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)

        # Calcula Cohen's Kappa (mede concordância além do acaso)
        kappa = cohen_kappa_score(y_val, y_pred)
        kappa_scores.append(kappa)

        # Matriz de confusão (para agregação)
        cm = confusion_matrix(y_val, y_pred)
        # Agrega a matriz de confusão deste fold na matriz total
        try:
            cv_total_cm += cm
        except Exception:
            # Caso haja problemas de shape, assegura conversão para array
            cv_total_cm = cv_total_cm + np.asarray(cm, dtype=int)
        # Calcula acurácia global (um valor por fold)
        acc = accuracy_score(y_val, y_pred)
        accuracy_scores.append(acc)

        # Coleta precision, recall e f1 de cada classe
        for cls in class_names:
            precision_scores[cls].append(report[cls]['precision'])
            recall_scores[cls].append(report[cls]['recall'])
            f1_scores[cls].append(report[cls]['f1-score'])

        print(f"Fold {fold_num}: Kappa = {kappa:.4f}")

    # Agrega os resultados (mean ± CI) da acurácia global por fold
    accuracy_scores = np.array(accuracy_scores)
    acc_mean, acc_ci = mean_confidence_interval(accuracy_scores)

    # CORRIGIDO: Agrega o Kappa corretamente (era uma lista, agora é mean ± CI)
    kappa_mean, kappa_ci = mean_confidence_interval(kappa_scores)
    print(f"\n✓ Cohen's Kappa (CV): {kappa_mean:.4f} ± {kappa_ci:.4f}")
    print()

    # Mostra a matriz de confusão agregada dos folds
    print("Matriz de Confusão Agregada (CV - soma de todos os folds):")
    print(cv_total_cm)
    print()

    # Monta o resumo das métricas por classe (sem acurácia por classe)
    cv_metrics_summary = {
        "Classe": [],
        "Precision Mean": [], "Precision CI": [],
        "Recall Mean": [], "Recall CI": [],
        "F1-score Mean": [], "F1-score CI": []
    }

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

    # Inclui a acurácia global (CV) no resumo para uso nos relatórios
    cv_metrics_summary["Global Accuracy Mean"] = acc_mean
    cv_metrics_summary["Global Accuracy CI"] = acc_ci

    # Retorna métricas de CV e o Kappa agregado, além da matriz de confusão agregada
    return cv_metrics_summary, kappa_mean, kappa_ci, cv_total_cm


def save_metrics_report(cv_metrics, kappa_mean, kappa_ci, class_names, dataset_name, output_dir="./results", cv_total_cm=None):
    """
    Salva um relatório completo das métricas em formato Markdown e Log.

    Args:
        cv_metrics (dict): Métricas da validação cruzada.
        kappa_mean (float): Média do Kappa na CV.
        kappa_ci (float): IC do Kappa na CV.
        cv_total_cm (np.array, optional): Matriz de confusão agregada da CV (soma dos folds).
        class_names (list): Nomes das classes.
        dataset_name (str): Nome do dataset.
        output_dir (str): Diretório onde salvar os relatórios.

    Returns:
        tuple: (caminho_markdown, caminho_log)
    """

    # Cria o diretório se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Timestamp pra nome do arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Nomes dos arquivos
    md_filename = f"metrics_report_{dataset_name}_{timestamp}.md"
    log_filename = f"metrics_report_{dataset_name}_{timestamp}.log"

    md_path = os.path.join(output_dir, md_filename)
    log_path = os.path.join(output_dir, log_filename)

    # ========================================
    # GERA RELATÓRIO EM MARKDOWN
    # ========================================

    md_content = f"""# Relatório de Métricas - {MODEL_TYPE.upper()}

**Dataset:** {dataset_name}
**Data/Hora:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

---

## 📊 Validação Cruzada (K-Fold)

Resultados da validação cruzada com **intervalo de confiança de 95%** (IC 95%).

### Métricas por Classe

"""

    # Tabela de métricas da CV (sem acurácia por classe)
    md_content += "| Classe | F1-Score | Precision | Recall |\n"
    md_content += "|--------|----------|-----------|--------|\n"

    for i, cls in enumerate(class_names):
        f1_mean = cv_metrics['F1-score Mean'][i]
        f1_ci = cv_metrics['F1-score CI'][i]
        prec_mean = cv_metrics['Precision Mean'][i]
        prec_ci = cv_metrics['Precision CI'][i]
        rec_mean = cv_metrics['Recall Mean'][i]
        rec_ci = cv_metrics['Recall CI'][i]

        md_content += f"| **{cls}** | {f1_mean:.4f} ± {f1_ci:.4f} | {prec_mean:.4f} ± {prec_ci:.4f} | {rec_mean:.4f} ± {rec_ci:.4f} |\n"

    md_content += f"\n### Métricas Globais (CV)\n\n"
    md_content += f"- **Acurácia (CV - Média ± IC):** {cv_metrics['Global Accuracy Mean']:.4f} ± {cv_metrics['Global Accuracy CI']:.4f}\n"
    md_content += f"- **Cohen's Kappa:** {kappa_mean:.4f} ± {kappa_ci:.4f}\n"
    md_content += f"\n---\n\n"

    # Se disponível, inclui a matriz de confusão agregada da CV
    if cv_total_cm is not None:
        md_content += "## 🧾 Matriz de Confusão (CV - Agregada)\n\n"
        md_content += "```\nPredito →    "
        for cls in class_names:
            md_content += f"{cls:>12} "
        md_content += "\nReal ↓\n"

        for i, cls in enumerate(class_names):
            md_content += f"{cls:12} "
            for j in range(len(class_names)):
                md_content += f"{cv_total_cm[i][j]:>12} "
            md_content += "\n"

        md_content += "```\n\n"

    # ========================================
    # Interpretação
    # ========================================

    md_content += f"---\n\n## 📈 Interpretação\n\n"

    # Melhor classe (maior F1 na Validação Cruzada)
    best_class_idx = np.argmax([cv_metrics['F1-score Mean'][i] for i in range(len(class_names))])
    best_class = class_names[best_class_idx]
    best_f1 = cv_metrics['F1-score Mean'][best_class_idx]

    md_content += f"- **Melhor desempenho (CV):** Classe `{best_class}` com F1-Score de **{best_f1:.4f}**\n"

    # Kappa interpretation (usando Kappa médio da CV)
    if kappa_mean > 0.8:
        kappa_interp = "Concordância **quase perfeita**"
    elif kappa_mean > 0.6:
        kappa_interp = "Concordância **substancial**"
    elif kappa_mean > 0.4:
        kappa_interp = "Concordância **moderada**"
    elif kappa_mean > 0.2:
        kappa_interp = "Concordância **fraca**"
    else:
        kappa_interp = "Concordância **pobre**"

    md_content += f"- **Cohen's Kappa (CV):** {kappa_interp} ({kappa_mean:.4f} ± {kappa_ci:.4f})\n"

    md_content += f"\n---\n\n"
    md_content += f"*Relatório gerado automaticamente pelo pipeline de treinamento {MODEL_TYPE.upper()}*\n"

    # Salva o arquivo Markdown
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    # ========================================
    # GERA RELATÓRIO EM LOG (texto simples)
    # ========================================

    log_content = f"""{'='*80}
RELATÓRIO DE MÉTRICAS - {MODEL_TYPE.upper()}
{'='*80}

Dataset: {dataset_name}
Data/Hora: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

{'='*80}
VALIDAÇÃO CRUZADA (K-Fold) - Média ± IC 95%
{'='*80}

"""

    # Métricas CV em texto
    for i, cls in enumerate(class_names):
        log_content += f"\nClasse: {cls}\n"
        log_content += f"  F1-Score:  {cv_metrics['F1-score Mean'][i]:.4f} ± {cv_metrics['F1-score CI'][i]:.4f}\n"
        log_content += f"  Precision: {cv_metrics['Precision Mean'][i]:.4f} ± {cv_metrics['Precision CI'][i]:.4f}\n"
        log_content += f"  Recall:    {cv_metrics['Recall Mean'][i]:.4f} ± {cv_metrics['Recall CI'][i]:.4f}\n"

    log_content += f"\nCohen's Kappa (CV): {kappa_mean:.4f} ± {kappa_ci:.4f}\n"
    log_content += f"Acurácia (CV - Média ± IC): {cv_metrics['Global Accuracy Mean']:.4f} ± {cv_metrics['Global Accuracy CI']:.4f}\n"

    # Inclui matriz de confusão agregada da CV, se disponível
    if cv_total_cm is not None:
        log_content += f"\n{'-'*80}\n"
        log_content += "MATRIZ DE CONFUSÃO (CV - Agregada)\n"
        log_content += f"{'-'*80}\n\n"

        # Header
        log_content += "Predito →    "
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
    log_content += f"Relatório salvo em: {md_path}\n"
    log_content += f"{'='*80}\n"

    # Salva o arquivo Log
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(log_content)

    return md_path, log_path