import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from scipy.stats import sem, t
from datetime import datetime
import os
from config import MODEL_TYPE
import time
import pickle
import tracemalloc
import sys
import tracemalloc
import sys

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


def evaluate_models(cv_models, final_model, X_test, y_test, class_names):
    """
    Avalia múltiplos modelos da validação cruzada E o modelo final no teste hold-out.

    IMPORTANTE: Segue boas práticas acadêmicas!
    1. Calcula métricas de VALIDAÇÃO CRUZADA (mean ± CI) pros folds
    2. Calcula métricas do TESTE FINAL no conjunto hold-out
    3. Agrega Cohen's Kappa corretamente (mean ± CI, não lista!)

    Args:
        cv_models (list): Lista de tuplas (model, X_val, y_val) dos folds de CV.
        final_model: Modelo final treinado em todos os dados de treino.
        X_test (pd.DataFrame): Features do conjunto de teste final (hold-out).
        y_test (np.array): Classes do conjunto de teste final (hold-out).
        class_names (list): Lista com os nomes das classes.

    Returns:
        tuple: (cv_metrics_summary, test_metrics, kappa_mean, kappa_ci)
            - cv_metrics_summary: Métricas da validação cruzada (mean ± CI por classe)
            - test_metrics: Métricas do teste final (hold-out)
            - kappa_mean: Média do Cohen's Kappa nos folds de CV
            - kappa_ci: Intervalo de confiança do Cohen's Kappa
            - test_kappa: Kappa no teste final
            - test_cm: Matriz de confusão do teste final
            - cv_total_cm: Matriz de confusão agregada da CV
            - deployment_metrics: Métricas de latência/memória para inferência
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

        # Calcula acurácia por classe (diagonal da matriz de confusão)
        cm = confusion_matrix(y_val, y_pred)
        # Agrega a matriz de confusão deste fold na matriz total
        try:
            cv_total_cm += cm
        except Exception:
            # Caso haja problemas de shape, assegura conversão para array
            cv_total_cm = cv_total_cm + np.asarray(cm, dtype=int)
        class_acc = cm.diagonal() / cm.sum(axis=1)
        accuracy_scores.append(class_acc)

        # Coleta precision, recall e f1 de cada classe
        for cls in class_names:
            precision_scores[cls].append(report[cls]['precision'])
            recall_scores[cls].append(report[cls]['recall'])
            f1_scores[cls].append(report[cls]['f1-score'])

        print(f"Fold {fold_num}: Kappa = {kappa:.4f}")

    # Agrega os resultados (mean ± CI)
    accuracy_scores = np.array(accuracy_scores)
    accuracy_conf_intervals = [mean_confidence_interval(accuracy_scores[:, i]) for i in range(len(class_names))]

    # CORRIGIDO: Agrega o Kappa corretamente (era uma lista, agora é mean ± CI)
    kappa_mean, kappa_ci = mean_confidence_interval(kappa_scores)
    print(f"\n✓ Cohen's Kappa (CV): {kappa_mean:.4f} ± {kappa_ci:.4f}")
    print()

    # Mostra a matriz de confusão agregada dos folds
    print("Matriz de Confusão Agregada (CV - soma de todos os folds):")
    print(cv_total_cm)
    print()

    # Monta o resumo das métricas por classe
    cv_metrics_summary = {
        "Classe": [],
        "Precision Mean": [], "Precision CI": [],
        "Recall Mean": [], "Recall CI": [],
        "F1-score Mean": [], "F1-score CI": [],
        "Accuracy Mean": [], "Accuracy CI": []
    }

    for i, cls in enumerate(class_names):
        pm, pci = mean_confidence_interval(precision_scores[cls])
        rm, rci = mean_confidence_interval(recall_scores[cls])
        f1m, f1ci = mean_confidence_interval(f1_scores[cls])
        acc_m, acc_ci = accuracy_conf_intervals[i]

        cv_metrics_summary["Classe"].append(cls)
        cv_metrics_summary["Precision Mean"].append(pm)
        cv_metrics_summary["Precision CI"].append(pci)
        cv_metrics_summary["Recall Mean"].append(rm)
        cv_metrics_summary["Recall CI"].append(rci)
        cv_metrics_summary["F1-score Mean"].append(f1m)
        cv_metrics_summary["F1-score CI"].append(f1ci)
        cv_metrics_summary["Accuracy Mean"].append(acc_m)
        cv_metrics_summary["Accuracy CI"].append(acc_ci)

    # ========================================
    # PARTE 2: Métricas do Teste Final (Hold-out)
    # ========================================

    print("=" * 60)
    print("AVALIAÇÃO: Métricas do Teste Final (Hold-out)")
    print("=" * 60)

    # Avalia o modelo final no conjunto de teste que NUNCA foi visto
    y_test_pred = final_model.predict(X_test)

    # Report completo
    test_report = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True)

    # Cohen's Kappa do teste final
    test_kappa = cohen_kappa_score(y_test, y_test_pred)

    # Matriz de confusão
    test_cm = confusion_matrix(y_test, y_test_pred)
    test_class_acc = test_cm.diagonal() / test_cm.sum(axis=1)

    # Monta dicionário com métricas do teste
    test_metrics = {
        "Classe": [],
        "Precision": [],
        "Recall": [],
        "F1-score": [],
        "Accuracy": []
    }

    for i, cls in enumerate(class_names):
        test_metrics["Classe"].append(cls)
        test_metrics["Precision"].append(test_report[cls]['precision'])
        test_metrics["Recall"].append(test_report[cls]['recall'])
        test_metrics["F1-score"].append(test_report[cls]['f1-score'])
        test_metrics["Accuracy"].append(test_class_acc[i])

    print(f"✓ Cohen's Kappa (Teste): {test_kappa:.4f}")
    print(f"✓ Acurácia Global (Teste): {test_report['accuracy']:.4f}")
    print()

    # Mostra matriz de confusão
    print("Matriz de Confusão (Teste Final):")
    print(test_cm)
    print()

    # ========================================
    # PARTE 3: Métricas de Desempenho de Inferência
    # ========================================

    print("=" * 60)
    print("AVALIAÇÃO: Métricas de Desempenho de Inferência")
    print("=" * 60)

    # Garantir numpy array para fatias de lote
    X_test_np = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)

    batch_sizes = [1, 8, 32, 128]
    repetitions = 10  # Aumentado para melhor estatística

    # Medição de latência por batch size
    latency_results = {}
    all_times_for_analysis = []  # Para estatísticas globais
    
    for bs in batch_sizes:
        # Limitar ao tamanho disponível
        n = min(len(X_test_np), bs)
        if n == 0:
            continue
        x_batch = X_test_np[:n]
        times = []
        for _ in range(repetitions):
            start = time.perf_counter()
            _ = final_model.predict(x_batch)
            end = time.perf_counter()
            elapsed = end - start
            times.append(elapsed)
            all_times_for_analysis.append(elapsed)
        
        times_ms = np.array(times) * 1000  # Converter para ms
        mean_time = np.mean(times)
        
        latency_results[bs] = {
            "mean_ms": float(np.mean(times_ms)),
            "std_ms": float(np.std(times_ms, ddof=1)) if len(times_ms) > 1 else 0.0,
            "p95_ms": float(np.percentile(times_ms, 95)),
            "p99_ms": float(np.percentile(times_ms, 99)),
            "max_ms": float(np.max(times_ms)),
            "per_sample_us": float((mean_time / n) * 1e6),
            "throughput_samples_per_sec": float(n / mean_time) if mean_time > 0 else 0.0
        }

    # Processing Time per Sample (média global de batch size 1)
    processing_time_per_sample_us = latency_results.get(1, {}).get("per_sample_us", 0.0)

    # Tamanho do modelo em memória (serializado)
    try:
        model_bytes = pickle.dumps(final_model)
        model_size_mb = len(model_bytes) / (1024 * 1024)
    except Exception:
        model_size_mb = None

    # Runtime memory (memória usada durante inferência)
    tracemalloc.start()
    baseline_memory = tracemalloc.get_traced_memory()[0]
    
    # Fazer algumas predições para capturar uso de memória
    test_batch = X_test_np[:min(100, len(X_test_np))]
    for _ in range(5):
        _ = final_model.predict(test_batch)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    runtime_memory_mb = (peak - baseline_memory) / (1024 * 1024)
    memory_overhead_mb = runtime_memory_mb

    # GOOSE deadline compatibility (IEC 61850: típico 3-4ms)
    goose_deadline_ms = 3.0  # Threshold padrão do protocolo GOOSE
    single_sample_latency_ms = latency_results.get(1, {}).get("mean_ms", 0.0)
    goose_compatible = single_sample_latency_ms <= goose_deadline_ms

    # Real-time capability (mensagens por segundo)
    real_time_msg_per_sec = latency_results.get(1, {}).get("throughput_samples_per_sec", 0.0)

    # Memory feasibility (comparação com limites típicos de edge devices)
    memory_limits = {
        "minimal": 256,  # 256 MB
        "standard": 512,  # 512 MB
        "comfortable": 1024  # 1 GB
    }
    
    total_memory_estimate_mb = (model_size_mb or 0) + runtime_memory_mb
    memory_feasibility = "Unknown"
    if total_memory_estimate_mb <= memory_limits["minimal"]:
        memory_feasibility = "Minimal (≤256MB) ✓"
    elif total_memory_estimate_mb <= memory_limits["standard"]:
        memory_feasibility = "Standard (≤512MB) ✓"
    elif total_memory_estimate_mb <= memory_limits["comfortable"]:
        memory_feasibility = "Comfortable (≤1GB) ✓"
    else:
        memory_feasibility = "High Memory (>1GB) ⚠"

    # Latency scaling (comparação entre batch sizes)
    latency_scaling = {}
    if 1 in latency_results and 128 in latency_results:
        time_1 = latency_results[1]["mean_ms"]
        time_128 = latency_results[128]["mean_ms"]
        scaling_factor = time_128 / (time_1 * 128) if time_1 > 0 else 0
        latency_scaling = {
            "batch_1_to_128_efficiency": float(scaling_factor),
            "interpretation": "Eficiente" if scaling_factor < 1.2 else "Ineficiente" if scaling_factor > 2.0 else "Moderado"
        }

    deployment_metrics = {
        "latency": latency_results,
        "processing_time_per_sample_us": processing_time_per_sample_us,
        "model_size_mb": model_size_mb,
        "runtime_memory_mb": runtime_memory_mb,
        "memory_overhead_mb": memory_overhead_mb,
        "total_memory_estimate_mb": total_memory_estimate_mb,
        "memory_feasibility": memory_feasibility,
        "goose_deadline_ms": goose_deadline_ms,
        "goose_compatible": goose_compatible,
        "real_time_capability_msg_per_sec": real_time_msg_per_sec,
        "latency_scaling": latency_scaling
    }

    # Print resumo expandido
    print("\n📊 LATÊNCIA:")
    for bs in sorted(latency_results.keys()):
        lr = latency_results[bs]
        print(f"  Batch {bs:3d}: Mean={lr['mean_ms']:7.3f}ms | Std={lr['std_ms']:6.3f}ms | "
              f"P95={lr['p95_ms']:7.3f}ms | P99={lr['p99_ms']:7.3f}ms | Max={lr['max_ms']:7.3f}ms")
        print(f"             Per-sample={lr['per_sample_us']:8.2f}µs | Throughput={lr['throughput_samples_per_sec']:8.1f} samples/s")
    
    print(f"\n✓ Processing Time per Sample: {processing_time_per_sample_us:.2f} µs")
    
    print("\n💾 MEMÓRIA:")
    if model_size_mb is not None:
        print(f"  Model size (serialized):  {model_size_mb:8.2f} MB")
    print(f"  Runtime memory (peak):    {runtime_memory_mb:8.2f} MB")
    print(f"  Memory overhead:          {memory_overhead_mb:8.2f} MB")
    print(f"  Total estimate:           {total_memory_estimate_mb:8.2f} MB")
    print(f"  Memory feasibility:       {memory_feasibility}")
    
    print("\n⚡ DEPLOYMENT / REAL-TIME:")
    print(f"  GOOSE deadline (3ms):     {'✓ COMPATIBLE' if goose_compatible else '✗ NOT COMPATIBLE'} ({single_sample_latency_ms:.3f}ms)")
    print(f"  Real-time capability:     {real_time_msg_per_sec:.1f} msg/s")
    
    if latency_scaling:
        print(f"  Latency scaling (1→128):  {latency_scaling['interpretation']} (factor: {latency_scaling['batch_1_to_128_efficiency']:.2f})")
    
    print("\n📌 NOTAS:")
    print("  - Overhead in passive monitoring: Latência adicional depende de integração com sistema")
    print("  - Embedded device performance: Valores são estimativas; teste em hardware real recomendado")
    print("  - Efficiency trade-offs: Considere quantização para reduzir model size e latência")
    print()

    return cv_metrics_summary, test_metrics, kappa_mean, kappa_ci, test_kappa, test_cm, cv_total_cm, deployment_metrics


def save_metrics_report(cv_metrics, test_metrics, kappa_mean, kappa_ci, test_kappa,
                       test_cm, class_names, dataset_name, output_dir="./results", cv_total_cm=None, deployment_metrics=None):
    """
    Salva um relatório completo das métricas em formato Markdown e Log.

    Args:
        cv_metrics (dict): Métricas da validação cruzada.
        test_metrics (dict): Métricas do teste final.
        kappa_mean (float): Média do Kappa na CV.
        kappa_ci (float): IC do Kappa na CV.
        test_kappa (float): Kappa do teste final.
        test_cm (np.array): Matriz de confusão do teste final.
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

    # Tabela de métricas da CV
    md_content += "| Classe | F1-Score | Precision | Recall | Accuracy |\n"
    md_content += "|--------|----------|-----------|--------|----------|\n"

    for i, cls in enumerate(class_names):
        f1_mean = cv_metrics['F1-score Mean'][i]
        f1_ci = cv_metrics['F1-score CI'][i]
        prec_mean = cv_metrics['Precision Mean'][i]
        prec_ci = cv_metrics['Precision CI'][i]
        rec_mean = cv_metrics['Recall Mean'][i]
        rec_ci = cv_metrics['Recall CI'][i]
        acc_mean = cv_metrics['Accuracy Mean'][i]
        acc_ci = cv_metrics['Accuracy CI'][i]

        md_content += f"| **{cls}** | {f1_mean:.4f} ± {f1_ci:.4f} | {prec_mean:.4f} ± {prec_ci:.4f} | {rec_mean:.4f} ± {rec_ci:.4f} | {acc_mean:.4f} ± {acc_ci:.4f} |\n"

    md_content += f"\n### Métricas Globais (CV)\n\n"
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
    # Métricas do Teste Final
    # ========================================

    md_content += f"## 🎯 Teste Final (Hold-out)\n\n"
    md_content += f"Resultados no conjunto de teste final (nunca visto durante o treinamento).\n\n"
    md_content += f"### Métricas por Classe\n\n"

    # Tabela de métricas do teste
    md_content += "| Classe | F1-Score | Precision | Recall | Accuracy |\n"
    md_content += "|--------|----------|-----------|--------|----------|\n"

    for i, cls in enumerate(class_names):
        f1 = test_metrics['F1-score'][i]
        prec = test_metrics['Precision'][i]
        rec = test_metrics['Recall'][i]
        acc = test_metrics['Accuracy'][i]

        md_content += f"| **{cls}** | {f1:.4f} | {prec:.4f} | {rec:.4f} | {acc:.4f} |\n"

    md_content += f"\n### Métricas Globais (Teste)\n\n"
    md_content += f"- **Cohen's Kappa:** {test_kappa:.4f}\n\n"

    # Matriz de confusão
    md_content += f"### Matriz de Confusão\n\n"
    md_content += "```\n"

    # Header
    md_content += "Predito →    "
    for cls in class_names:
        md_content += f"{cls:>12} "
    md_content += "\n"
    md_content += "Real ↓\n"

    # Linhas da matriz
    for i, cls in enumerate(class_names):
        md_content += f"{cls:12} "
        for j in range(len(class_names)):
            md_content += f"{test_cm[i][j]:>12} "
        md_content += "\n"

    md_content += "```\n\n"

    # ========================================
    # Desempenho de Inferência
    # ========================================
    if deployment_metrics is not None:
        md_content += f"## ⚙️ Desempenho de Inferência\n\n"
        
        # Seção de Latência
        md_content += f"### 📊 Latência\n\n"
        md_content += "| Batch | Mean (ms) | Std (ms) | P95 (ms) | P99 (ms) | Max (ms) | Per-Sample (µs) | Throughput (samples/s) |\n"
        md_content += "|-------|-----------|----------|----------|----------|----------|-----------------|------------------------|\n"
        for bs in sorted(deployment_metrics["latency"].keys()):
            lr = deployment_metrics["latency"][bs]
            md_content += f"| {bs:5d} | {lr['mean_ms']:9.3f} | {lr['std_ms']:8.3f} | {lr['p95_ms']:8.3f} | {lr['p99_ms']:8.3f} | {lr['max_ms']:8.3f} | {lr['per_sample_us']:15.2f} | {lr['throughput_samples_per_sec']:22.1f} |\n"
        
        md_content += f"\n**✓ Processing Time per Sample:** {deployment_metrics['processing_time_per_sample_us']:.2f} µs\n\n"
        
        # Latency Scaling
        if deployment_metrics.get("latency_scaling"):
            ls = deployment_metrics["latency_scaling"]
            md_content += f"**Latency Scaling (Batch 1→128):** {ls['interpretation']} (efficiency factor: {ls['batch_1_to_128_efficiency']:.2f})\n\n"
        
        # Seção de Memória
        md_content += f"### 💾 Memória\n\n"
        md_content += "| Métrica | Valor |\n"
        md_content += "|---------|-------|\n"
        if deployment_metrics.get("model_size_mb") is not None:
            md_content += f"| Model size (serialized) | {deployment_metrics['model_size_mb']:.2f} MB |\n"
        md_content += f"| Runtime memory (peak) | {deployment_metrics['runtime_memory_mb']:.2f} MB |\n"
        md_content += f"| Memory overhead | {deployment_metrics['memory_overhead_mb']:.2f} MB |\n"
        md_content += f"| **Total estimate** | **{deployment_metrics['total_memory_estimate_mb']:.2f} MB** |\n"
        md_content += f"| Memory feasibility | {deployment_metrics['memory_feasibility']} |\n\n"
        
        # Seção de Deployment/Real-time
        md_content += f"### ⚡ Deployment / Real-time\n\n"
        md_content += "| Métrica | Valor |\n"
        md_content += "|---------|-------|\n"
        goose_status = "✓ COMPATIBLE" if deployment_metrics['goose_compatible'] else "✗ NOT COMPATIBLE"
        goose_latency = deployment_metrics['latency'].get(1, {}).get('mean_ms', 0.0)
        md_content += f"| GOOSE deadline compatibility (3ms) | {goose_status} ({goose_latency:.3f}ms) |\n"
        md_content += f"| Real-time capability | {deployment_metrics['real_time_capability_msg_per_sec']:.1f} msg/s |\n\n"
        
        md_content += f"**📌 Notas sobre Deployment:**\n\n"
        md_content += f"- **Overhead in passive monitoring:** A latência adicional depende da integração com o sistema de monitoramento\n"
        md_content += f"- **Embedded device performance:** Os valores apresentados são estimativas; testes em hardware real são recomendados\n"
        md_content += f"- **Efficiency trade-offs:** Considere técnicas de quantização para reduzir model size e latência em ~2-4x\n\n"

    # ========================================
    # Interpretação
    # ========================================

    md_content += f"---\n\n## 📈 Interpretação\n\n"

    # Melhor classe (maior F1 no teste)
    best_class_idx = np.argmax([test_metrics['F1-score'][i] for i in range(len(class_names))])
    best_class = class_names[best_class_idx]
    best_f1 = test_metrics['F1-score'][best_class_idx]

    md_content += f"- **Melhor desempenho:** Classe `{best_class}` com F1-Score de **{best_f1:.4f}**\n"

    # Kappa interpretation
    if test_kappa > 0.8:
        kappa_interp = "Concordância **quase perfeita**"
    elif test_kappa > 0.6:
        kappa_interp = "Concordância **substancial**"
    elif test_kappa > 0.4:
        kappa_interp = "Concordância **moderada**"
    elif test_kappa > 0.2:
        kappa_interp = "Concordância **fraca**"
    else:
        kappa_interp = "Concordância **pobre**"

    md_content += f"- **Cohen's Kappa ({test_kappa:.4f}):** {kappa_interp}\n"

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
        log_content += f"  Accuracy:  {cv_metrics['Accuracy Mean'][i]:.4f} ± {cv_metrics['Accuracy CI'][i]:.4f}\n"

    log_content += f"\nCohen's Kappa (CV): {kappa_mean:.4f} ± {kappa_ci:.4f}\n"

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
    log_content += f"TESTE FINAL (Hold-out)\n"
    log_content += f"{'='*80}\n"

    # Métricas teste em texto
    for i, cls in enumerate(class_names):
        log_content += f"\nClasse: {cls}\n"
        log_content += f"  F1-Score:  {test_metrics['F1-score'][i]:.4f}\n"
        log_content += f"  Precision: {test_metrics['Precision'][i]:.4f}\n"
        log_content += f"  Recall:    {test_metrics['Recall'][i]:.4f}\n"
        log_content += f"  Accuracy:  {test_metrics['Accuracy'][i]:.4f}\n"

    log_content += f"\nCohen's Kappa (Teste): {test_kappa:.4f}\n"

    # Matriz de confusão
    log_content += f"\n{'-'*80}\n"
    log_content += "MATRIZ DE CONFUSÃO (Teste Final)\n"
    log_content += f"{'-'*80}\n\n"

    # Header
    log_content += "Predito →    "
    for cls in class_names:
        log_content += f"{cls:>12} "
    log_content += "\nReal ↓\n"

    # Linhas
    for i, cls in enumerate(class_names):
        log_content += f"{cls:12} "
        for j in range(len(class_names)):
            log_content += f"{test_cm[i][j]:>12} "
        log_content += "\n"

    # ========================================
    # DESEMPENHO DE INFERÊNCIA (LOG)
    # ========================================
    if deployment_metrics is not None:
        log_content += f"\n{'='*80}\n"
        log_content += f"DESEMPENHO DE INFERÊNCIA\n"
        log_content += f"{'='*80}\n\n"
        
        log_content += "📊 LATÊNCIA:\n"
        for bs in sorted(deployment_metrics["latency"].keys()):
            lr = deployment_metrics["latency"][bs]
            log_content += f"  Batch {bs:3d}: Mean={lr['mean_ms']:7.3f}ms | Std={lr['std_ms']:6.3f}ms | "
            log_content += f"P95={lr['p95_ms']:7.3f}ms | P99={lr['p99_ms']:7.3f}ms | Max={lr['max_ms']:7.3f}ms\n"
            log_content += f"             Per-sample={lr['per_sample_us']:8.2f}µs | Throughput={lr['throughput_samples_per_sec']:8.1f} samples/s\n"
        
        log_content += f"\n✓ Processing Time per Sample: {deployment_metrics['processing_time_per_sample_us']:.2f} µs\n"
        
        if deployment_metrics.get("latency_scaling"):
            ls = deployment_metrics["latency_scaling"]
            log_content += f"  Latency scaling (1→128): {ls['interpretation']} (factor: {ls['batch_1_to_128_efficiency']:.2f})\n"
        
        log_content += "\n💾 MEMÓRIA:\n"
        if deployment_metrics.get("model_size_mb") is not None:
            log_content += f"  Model size (serialized):  {deployment_metrics['model_size_mb']:8.2f} MB\n"
        log_content += f"  Runtime memory (peak):    {deployment_metrics['runtime_memory_mb']:8.2f} MB\n"
        log_content += f"  Memory overhead:          {deployment_metrics['memory_overhead_mb']:8.2f} MB\n"
        log_content += f"  Total estimate:           {deployment_metrics['total_memory_estimate_mb']:8.2f} MB\n"
        log_content += f"  Memory feasibility:       {deployment_metrics['memory_feasibility']}\n"
        
        log_content += "\n⚡ DEPLOYMENT / REAL-TIME:\n"
        goose_status = "✓ COMPATIBLE" if deployment_metrics['goose_compatible'] else "✗ NOT COMPATIBLE"
        goose_latency = deployment_metrics['latency'].get(1, {}).get('mean_ms', 0.0)
        log_content += f"  GOOSE deadline (3ms):     {goose_status} ({goose_latency:.3f}ms)\n"
        log_content += f"  Real-time capability:     {deployment_metrics['real_time_capability_msg_per_sec']:.1f} msg/s\n"
        
        log_content += "\n📌 NOTAS:\n"
        log_content += "  - Overhead in passive monitoring: Latência adicional depende de integração com sistema\n"
        log_content += "  - Embedded device performance: Valores são estimativas; teste em hardware real recomendado\n"
        log_content += "  - Efficiency trade-offs: Considere quantização para reduzir model size e latência\n"

    log_content += f"\n{'='*80}\n"
    log_content += f"Relatório salvo em: {md_path}\n"
    log_content += f"{'='*80}\n"

    # Salva o arquivo Log
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(log_content)

    return md_path, log_path