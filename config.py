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

# ============================================
# PARÂMETROS DO MODELO XGBOOST
# ============================================

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
WITHOUT_CONSISTENCY_FEATURES = True

# Se for pra remover, adiciona elas na lista de descartadas
if WITHOUT_CONSISTENCY_FEATURES:
    CONSISTENCY_FEATURES = [
        'stDiff', 'sqDiff', 'gooseLengthDiff', 'cbStatusDiff', 'apduSizeDiff', 
        'frameLengthDiff', 'timestampDiff', 'tDiff', 'timeFromLastChange'
    ]
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
    "Violin Summary Plot",      # Mostra distribuição dos valores SHAP
    "Bar Plot",                  # Importância média das features
    "Beeswarm Summary Plot",     # Visualização densa dos valores SHAP
    "Waterfall Summary Plot",     # Contribuição individual de cada feature
    "Force Plot"                # Contribuição detalhada para uma predição específica
]