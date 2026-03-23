"""
Pipeline completo de treinamento e avaliação de modelo XGBoost.

Este script implementa as melhores práticas acadêmicas:
1. Separação correta de treino/teste (hold-out)
2. Validação cruzada K-Fold estratificada
3. Cálculo de métricas com intervalos de confiança
4. Análise de explicabilidade com SHAP
5. Otimizações pra lidar com datasets grandes
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
    Função principal que executa todo o pipeline.
    """

    print("\n" + "=" * 60)
    print(f"PIPELINE DE TREINAMENTO E AVALIAÇÃO - {MODEL_TYPE.upper()}")
    print("=" * 60)
    print()

    # ========================================
    # ETAPA 1: Carregamento dos Dados
    # ========================================

    print("=" * 60)
    print("ETAPA 1: Carregamento dos Dados")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET_PATH}")

    print(f"Carregando dataset: {DATASET_PATH}")
    df = load_dataset(DATASET_PATH, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)
    print(f"✓ Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
    print()

    # ========================================
    # ETAPA 2: Pré-processamento
    # ========================================

    print("=" * 60)
    print("ETAPA 2: Pré-processamento")
    print("=" * 60)

    # Separa features (X) e target (y)
    X, y, encoder = preprocess(df, target_column="class", discarted_columns=DISCARTED_COLUMNS)

    print(f"✓ Features: {X.shape[1]} colunas")
    print(f"✓ Amostras: {len(y)}")
    print(f"✓ Classes: {CLASS_NAMES}")
    print(f"✓ Distribuição de classes:")

    # Mostra a distribuição de classes (importante pra detectar desbalanceamento)
    for i, cls in enumerate(CLASS_NAMES):
        count = (y == i).sum()
        percentage = (count / len(y)) * 100
        print(f"    - {cls}: {count} amostras ({percentage:.1f}%)")
    print()

    # Libera memória do DataFrame original
    del df

    # ========================================
    # ETAPA 3: Treinamento
    # ========================================

    # Exibe qual modelo foi escolhido
    model_desc = MODEL_DESCRIPTIONS.get(MODEL_PARAMS.__class__.__name__, "")
    canonical_model = MODEL_NAME_ALIASES.get(MODEL_TYPE.lower(), MODEL_TYPE)
    print("=" * 60)
    print(f"MODELO ESCOLHIDO: {canonical_model.upper()}")
    print(f"Description: {MODEL_DESCRIPTIONS.get(canonical_model, canonical_model)}")
    print("=" * 60)
    print()

    # Treina modelos usando validação cruzada + hold-out test set
    cv_models, final_model, X_test, y_test = train_model(
        X, y,
        model_type=MODEL_TYPE,
        params=MODEL_PARAMS.get(canonical_model, XGBOOST_PARAMS),
        n_splits=N_SPLITS,
        seed=RANDOM_STATE,
        test_size=TEST_SIZE
    )

    # ========================================
    # ETAPA 4: Avaliação
    # ========================================

    # Avalia modelos da CV e o modelo final
    cv_metrics, test_metrics, kappa_mean, kappa_ci, test_kappa, test_cm, cv_total_cm = evaluate_models(
        cv_models, final_model, X_test, y_test, CLASS_NAMES
    )

    # ========================================
    # ETAPA 5: Resultados Finais
    # ========================================

    print("=" * 60)
    print("RESULTADOS FINAIS")
    print("=" * 60)
    print()

    print("📊 VALIDAÇÃO CRUZADA (Média ± IC 95%):")
    print("-" * 60)
    for i, cls in enumerate(CLASS_NAMES):
        print(f"\n{cls}:")
        print(f"  F1-score:  {cv_metrics['F1-score Mean'][i]:.4f} ± {cv_metrics['F1-score CI'][i]:.4f}")
        print(f"  Precision: {cv_metrics['Precision Mean'][i]:.4f} ± {cv_metrics['Precision CI'][i]:.4f}")
        print(f"  Recall:    {cv_metrics['Recall Mean'][i]:.4f} ± {cv_metrics['Recall CI'][i]:.4f}")

    # Mostra acurácia global da CV (média ± IC)
    if 'Global Accuracy Mean' in cv_metrics and 'Global Accuracy CI' in cv_metrics:
        print(f"\nAcurácia (CV): {cv_metrics['Global Accuracy Mean']:.4f} ± {cv_metrics['Global Accuracy CI']:.4f}")

    print(f"\nCohen's Kappa (CV): {kappa_mean:.4f} ± {kappa_ci:.4f}")
    print()

    print("🎯 TESTE FINAL (Hold-out - Conjunto nunca visto):")
    print("-" * 60)
    for i, cls in enumerate(CLASS_NAMES):
        print(f"\n{cls}:")
        print(f"  F1-score:  {test_metrics['F1-score'][i]:.4f}")
        print(f"  Precision: {test_metrics['Precision'][i]:.4f}")
        print(f"  Recall:    {test_metrics['Recall'][i]:.4f}")

    # Calcula e mostra acurácia global do teste a partir da matriz de confusão retornada
    try:
        test_acc = test_cm.diagonal().sum() / test_cm.sum()
    except Exception:
        test_acc = 0.0
    print(f"\nAcurácia (Teste): {test_acc:.4f}")

    print(f"\nCohen's Kappa (Teste): {test_kappa:.4f}")
    print()

    # ========================================
    # ETAPA 5.1: Salvar Relatório de Métricas
    # ========================================

    print("=" * 60)
    print("SALVANDO RELATÓRIOS")
    print("=" * 60)
    print()

    # Extrai o nome do dataset do caminho
    dataset_name = os.path.basename(DATASET_PATH).replace(".csv", "").replace(".parquet", "")

    # Salva relatórios em Markdown e Log
    md_path, log_path = save_metrics_report(
        cv_metrics, test_metrics, kappa_mean, kappa_ci, test_kappa,
        test_cm, CLASS_NAMES, dataset_name, output_dir=PATH_BASE, cv_total_cm=cv_total_cm
    )

    print(f"✓ Relatório Markdown salvo: {md_path}")
    print(f"✓ Relatório Log salvo: {log_path}")
    print()

    # ========================================
    # ETAPA 6: Explicabilidade (SHAP)
    # ========================================

    # Descomenta abaixo pra gerar os gráficos SHAP
    # ATENÇÃO: pode demorar bastante dependendo do tamanho do dataset!

    print("=" * 60)
    print("ETAPA 6: Análise de Explicabilidade (SHAP)")
    print("=" * 60)
    print()
    print("⚠️  A análise SHAP está DESABILITADA por padrão.")
    print("    Pra habilitar, descomente as linhas no final do main.py")
    print("    (pode demorar vários minutos dependendo do dataset!)")
    print()
    
    print("conteúdo presente no xtest:", X_test.columns.tolist())

    # Descomenta as linhas abaixo pra rodar o SHAP:
    # run_shap(
    #     final_model,
    #     X_test,
    #     CLASS_NAMES,
    #     dataset_name=dataset_name,
    #     path_base=PATH_BASE,
    #     graphics=GRAPHICS,
    #     sample_percentage=SHAP_SAMPLE_PERCENTAGE,
    #     random_state=RANDOM_STATE
    # )

    print("=" * 60)
    print("✓ PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()