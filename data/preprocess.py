from json import encoder
from sklearn.preprocessing import LabelEncoder

def preprocess(df, target_column, discarted_columns):
    """
    Faz o pré-processamento do dataset, separando features (X) e target (y).

    Args:
        df (pd.DataFrame): Dataset completo.
        target_column (str): Nome da coluna que é o alvo (classe).
        discarted_columns (list): Lista de colunas pra descartar (não usar como features).

    Returns:
        tuple: (X, y_encoded, encoder)
            - X: DataFrame com as features
            - y_encoded: Array numpy com as classes codificadas numericamente
            - encoder: LabelEncoder ajustado (útil pra decodificar depois)
    """
    # Separa features (X) do target (y)
    # Remove as colunas que não devem ser usadas como features
    X = df.drop(columns=discarted_columns)
    y = df[target_column]

    # Codifica as classes de string pra números (0, 1, 2, ...)
    # Ex: ["FRG", "Normal"] vira [0, 1]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Retorna também o encoder pra poder decodificar as predições depois
    return X, y_encoded, encoder