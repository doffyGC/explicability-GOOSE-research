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
    Retorna o classificador configurado com base no tipo informado.
    
    Suporta aliases de nomes (ex: 'xgb' -> 'xgboost', 'rf' -> 'random_forest', etc).
    """
    from config import MODEL_NAME_ALIASES
    
    # Normaliza o nome: lowercase e remove espaços
    model_type_normalized = str(model_type).strip().lower()
    
    # Resolve o alias pro nome canônico
    if model_type_normalized in MODEL_NAME_ALIASES:
        model_type_canonical = MODEL_NAME_ALIASES[model_type_normalized]
    else:
        raise ValueError(
            f"Modelo não suportado: '{model_type}'. "
            f"Modelos disponíveis: {', '.join(MODEL_NAME_ALIASES.values())}"
        )
    
    match model_type_canonical:
        case "xgboost":
            return xgb.XGBClassifier(**params)
        case "random_forest":
            return RandomForestClassifier(**params)
        case "svm":
            # SVM é sensível à escala: usa StandardScaler no pipeline
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(**params))
            ])
        case "mlp":
            # MLP também se beneficia de normalização
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(**params))
            ])
        case "decision_tree":
            return DecisionTreeClassifier(**params)
        case "logistic_regression":
            # LR (multinomial) também é sensível à escala
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**params))
            ])
        case _:
            raise ValueError(f"Modelo não suportado: {model_type_canonical}")

def train_model(X, y, model_type, params, n_splits, seed):
    """
    Treina um classificador (XGBoost ou baseline escolhido) usando validação cruzada K-Fold.

    IMPORTANTE: Segue boas práticas acadêmicas!
    1. Usa Validação Cruzada K-Fold em todo o conjunto de dados

    Args:
        X (pd.DataFrame): DataFrame com as features.
        y (np.array): Array com a variável target (já codificada).
        model_type (str): Tipo de classificador (xgboost, random_forest, svm, mlp, decision_tree, logistic_regression).
        params (dict): Dicionário com os parâmetros do classificador escolhido.
        n_splits (int): Número de folds pra validação cruzada.
        seed (int): Seed pra reprodutibilidade.
        test_size (float): Proporção do dataset reservada pro teste final (padrão: 0.2 = 20%).

    Returns:
        tuple: (cv_models, final_model)
            - cv_models: Lista de tuplas (model, X_val, y_val) dos folds de validação
            - final_model: Modelo final treinado em TODOS os dados
    """
    print("=" * 60)
    print(f"ETAPA: Validação cruzada K-Fold ({n_splits} folds)")
    print("=" * 60)

    # Faz K-Fold em TODO o conjunto de dados
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_models = []

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        # Divide o conjunto em treino e validação desse fold
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        print(f"\nTreinando fold {fold_num}/{n_splits}...")
        print(f"  - Treino: {len(X_train_fold)} amostras")
        print(f"  - Validação: {len(X_val_fold)} amostras")

        # Treina o modelo nesse fold
        model = _get_classifier(model_type, params)
        
        print("  - Treinando modelo...")
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
    print(f"Treinando modelo final com TODOS os {len(X)} dados...")
    final_model = _get_classifier(model_type, params)
    final_model.fit(X, y)
    print("✓ Modelo final treinado!")
    print()

    # Retorna cv_models (para métricas) e final_model (para SHAP)
    return cv_models, final_model
