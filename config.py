N_SPLITS = 5
RANDOM_STATE = 42
XGBOOST_PARAMS = {
    # "objective":'multi:softprob',
    "objective":'binary:logistic',
    "eval_metric":'mlogloss',
    "random_state": RANDOM_STATE
}
DISCARTED_COLUMNS = ['ethDst', 'ethSrc', 'gocbRef', 'datSet', 'goID', 'test', 'ndsCom', 'protocol', 'ethType', 'TPID', 'gooseAppid', 'class']
CLASS_NAMES = ["SOG.DB", "Normal"]
# "SOG.DB", "SOG.PB", "SOG.PBM", 
DATASET_PATH = "./data/CSV files/DB.csv"
GRAPHICS = ["Violin Summary Plot", "Bar Plot", "Beeswarm Summary Plot", "Waterfall Summary Plot"]