import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
import gc

def train_model(X, y, params, n_splits, seed, test_size=0.2):
    """
    Treina um modelo XGBoost usando validação cruzada K-Fold.

    IMPORTANTE: Segue boas práticas acadêmicas!
    1. Separa um conjunto de TESTE FINAL (hold-out) ANTES do K-Fold
    2. Usa K-Fold apenas no conjunto de TREINO
    3. O teste final fica intocado até a avaliação final

    Args:
        X (pd.DataFrame): DataFrame com as features.
        y (np.array): Array com a variável target (já codificada).
        params (dict): Dicionário com os parâmetros do XGBoost.
        n_splits (int): Número de folds pra validação cruzada.
        seed (int): Seed pra reprodutibilidade.
        test_size (float): Proporção do dataset reservada pro teste final (padrão: 0.2 = 20%).

    Returns:
        tuple: (cv_models, final_model, X_test, y_test)
            - cv_models: Lista de tuplas (model, X_val, y_val) dos folds de validação
            - final_model: Modelo final treinado em TODOS os dados de treino
            - X_test: Features do conjunto de teste final (hold-out)
            - y_test: Classes do conjunto de teste final (hold-out)
    """

    print("=" * 60)
    print("ETAPA 1: Separando conjunto de teste final (hold-out)")
    print("=" * 60)

    # PASSO 1: Separa teste final ANTES de qualquer coisa
    # Isso é ESSENCIAL pra ter uma avaliação honesta do modelo
    # Esse conjunto NÃO PODE ser usado no treinamento!
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,  # Mantém a proporção de classes
        random_state=seed
    )

    print(f"✓ Dados divididos:")
    print(f"  - Treino (pra K-Fold): {len(X_train_full)} amostras ({(1-test_size)*100:.0f}%)")
    print(f"  - Teste final (hold-out): {len(X_test)} amostras ({test_size*100:.0f}%)")
    print()

    print("=" * 60)
    print(f"ETAPA 2: Validação cruzada K-Fold ({n_splits} folds)")
    print("=" * 60)

    # PASSO 2: Faz K-Fold APENAS nos dados de treino
    # Cada fold vai treinar e validar pra estimar a performance
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_models = []

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_full), 1):
        # Divide o conjunto de treino em treino e validação desse fold
        X_train_fold = X_train_full.iloc[train_idx]
        X_val_fold = X_train_full.iloc[val_idx]
        y_train_fold = y_train_full[train_idx]
        y_val_fold = y_train_full[val_idx]

        print(f"\nTreinando fold {fold_num}/{n_splits}...")
        print(f"  - Treino: {len(X_train_fold)} amostras")
        print(f"  - Validação: {len(X_val_fold)} amostras")

        # Treina o modelo nesse fold
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_fold, y_train_fold)

        # Guarda o modelo e os dados de validação pra calcular métricas depois
        cv_models.append((model, X_val_fold, y_val_fold))

        # Limpa memória pra não estourar em datasets grandes
        del X_train_fold, y_train_fold
        gc.collect()

    print("\n✓ Validação cruzada concluída!")
    print()

    print("=" * 60)
    print("ETAPA 3: Treinando modelo final")
    print("=" * 60)

    # PASSO 3: Treina modelo final usando TODOS os dados de treino
    # Esse modelo vai ser usado pro SHAP e pra produção
    print(f"Treinando modelo final com TODOS os {len(X_train_full)} dados de treino...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X_train_full, y_train_full)
    print("✓ Modelo final treinado!")
    print()

    # Retorna:
    # - cv_models: pros cálculos de métrica (mean ± CI)
    # - final_model: pro SHAP e avaliação final
    # - X_test, y_test: pro teste final (hold-out)
    return cv_models, final_model, X_test, y_test
