# ============================================
# CONFIGURAÇÕES GERAIS DO EXPERIMENTO
# ============================================

# Seed pra reprodutibilidade - sempre usa o mesmo valor em tudo!
RANDOM_STATE = 42

# Número de folds pra validação cruzada (K-Fold)
# 5 é o padrão acadêmico, mas pode aumentar pra 10 se quiser mais robustez
N_SPLITS = 5

# Proporção do dataset reservada pro teste final
# 20% é padrão, o resto (80%) vai ser usado no K-Fold
TEST_SIZE = 0.2

# ============================================
# CONFIGURAÇÕES DE PERFORMANCE
# ============================================

# Limite de linhas pro dataset (None = usa tudo)
# Reduz isso se tiver problema de memória com datasets grandes
# Ex: SAMPLE_SIZE = 100000 pra pegar só 100k linhas
SAMPLE_SIZE = None

# Porcentagem do dataset de teste usada pro cálculo do SHAP
# SHAP é MUITO pesado! Recomendado: 10-20% do dataset de teste
# Mesmo com amostragem, os resultados são bem representativos
# Usa None pra processar todo o dataset de teste (não recomendado pra datasets grandes)
SHAP_SAMPLE_PERCENTAGE = 1  # 20% do dataset de teste

# Caminho base pra salvar os resultados dos experimentos
# Cada experimento vai criar uma subpasta aqui
PATH_BASE="./results_with_consistency_features"

# ============================================
# PARÂMETROS DO MODELO XGBOOST
# ============================================

# Tipo de modelo/classificador a ser usado.
# Opções completas:
#   - 'xgboost' (aliases: 'xgb')
#   - 'random_forest' (aliases: 'rf', 'randomforest')
#   - 'svm' (aliases: 'support_vector_machine', 'supportvectormachine')
#   - 'mlp' (aliases: 'neural_network', 'multilayer_perceptron')
#   - 'decision_tree' (aliases: 'dt', 'decisiontree')
#   - 'logistic_regression' (aliases: 'lr', 'logisticregression')
MODEL_TYPE = "mlp"

# Mapa de aliases para facilitar a escrita de nomes dos modelos
MODEL_NAME_ALIASES = {
    # XGBoost
    "xgboost": "xgboost",
    "xgb": "xgboost",
    # Random Forest
    "random_forest": "random_forest",
    "rf": "random_forest",
    "randomforest": "random_forest",
    # SVM
    "svm": "svm",
    "support_vector_machine": "svm",
    "supportvectormachine": "svm",
    # MLP
    "mlp": "mlp",
    "neural_network": "mlp",
    "multilayer_perceptron": "mlp",
    # Decision Tree
    "decision_tree": "decision_tree",
    "dt": "decision_tree",
    "decisiontree": "decision_tree",
    # Logistic Regression
    "logistic_regression": "logistic_regression",
    "lr": "logistic_regression",
    "logisticregression": "logistic_regression",
}

XGBOOST_PARAMS = {
    # Função objetivo: binary:logistic pra classificação binária
    "objective": 'multi:softprob',  # multi:softprob pra multi-class

    # Métrica de avaliação: logloss (correto pra binary:logistic)
    # Obs: mlogloss é pra multi-class, logloss é pra binário
    "eval_metric": 'mlogloss',
    # Seed pra reprodutibilidade
    "random_state": RANDOM_STATE,

    # Parâmetros de regularização pra evitar overfitting
    # Descomenta e ajusta esses se quiser fazer tuning:
    # "max_depth": 6,              # Profundidade máxima das árvores
    # "learning_rate": 0.1,        # Taxa de aprendizado (eta)
    # "n_estimators": 100,         # Número de árvores
    # "min_child_weight": 1,       # Peso mínimo necessário em uma folha
    # "subsample": 0.8,            # Proporção de amostras usadas por árvore
    # "colsample_bytree": 0.8,     # Proporção de features usadas por árvore
    # "scale_pos_weight": 1,       # Peso pra classe positiva (útil em desbalanceamento)
}

# Parâmetros para Random Forest (baseline ensemble)
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# Parâmetros para SVM com kernel RBF (tradicional em IDS)
SVM_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "probability": True,  # habilita predict_proba
    "random_state": RANDOM_STATE,
}

# Parâmetros para MLP (baseline deep learning)
MLP_PARAMS = {
    "hidden_layer_sizes": (100,),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "learning_rate": "constant",
    "max_iter": 200,
    "random_state": RANDOM_STATE,
}

# Parâmetros para Decision Tree (interpretável)
DT_PARAMS = {
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
}

# Parâmetros para Logistic Regression (linear)
LR_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 500,
    "multi_class": "multinomial",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Mapa para facilitar a escolha do modelo (usando nomes canônicos)
MODEL_PARAMS = {
    "xgboost": XGBOOST_PARAMS,
    "random_forest": RF_PARAMS,
    "svm": SVM_PARAMS,
    "mlp": MLP_PARAMS,
    "decision_tree": DT_PARAMS,
    "logistic_regression": LR_PARAMS,
}

# Descrição dos modelos pra logging
MODEL_DESCRIPTIONS = {
    "xgboost": "XGBoost (Gradient Boosting)",
    "random_forest": "Random Forest (Ensemble - Baseline Clássico)",
    "svm": "SVM com kernel RBF (Tradicional em IDS)",
    "mlp": "MLP - Neural Network (Baseline Deep Learning)",
    "decision_tree": "Decision Tree (Baseline Interpretável)",
    "logistic_regression": "Logistic Regression (Baseline Linear)",
}

# ============================================
# CONFIGURAÇÕES DO DATASET
# ============================================

# Colunas que não devem ser usadas como features
# A coluna 'class' é o target, então tem que ser removida
DISCARTED_COLUMNS = [
    'ethDst', 'ethSrc', 'gocbRef', 'datSet', 'goID', 'test',
    'ndsCom', 'protocol', 'ethType', 'TPID', 'gooseAppid', 'class'
]

# Remove features de consistência (baseadas em temporização)
# Essas features podem vazar informação do alvo
# Pro caso de uso real, é melhor não usar essas features
WITHOUT_CONSISTENCY_FEATURES = False

# Se for pra remover, adiciona elas na lista de descartadas
if WITHOUT_CONSISTENCY_FEATURES:
    CONSISTENCY_FEATURES = [
        'stDiff', 'sqDiff', 'gooseLengthDiff', 'cbStatusDiff', 'apduSizeDiff', 
        'frameLengthDiff', 'timestampDiff', 'tDiff', 'timeFromLastChange'
    ]
    PATH_BASE="./results_without_consistency_features"
    DISCARTED_COLUMNS.extend(CONSISTENCY_FEATURES)

# Nomes das classes pro problema de classificação
# A ordem importa! Deve corresponder à ordem do LabelEncoder
CLASS_NAMES = ["SOG.DB", "FRG", "SOG.PB", "SOG.PBM", "Normal"]

# Caminho pro dataset
DATASET_PATH = "./data/CSV files/dataset_downsampled.csv"

# ============================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO (SHAP)
# ============================================

# Tipos de gráficos SHAP pra gerar
# Cada gráfico dá uma perspectiva diferente da explicabilidade
GRAPHICS = [
    # "Violin Summary Plot",      # Mostra distribuição dos valores SHAP
    # "Bar Plot",                  # Importância média das features
    # "Beeswarm Summary Plot",     # Visualização densa dos valores SHAP
    # "Waterfall Summary Plot",     # Contribuição individual de cada feature
    # "Force Plot"                # Contribuição detalhada para uma predição específica
]