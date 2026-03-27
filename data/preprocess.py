from json import encoder
from sklearn.preprocessing import LabelEncoder

def preprocess(df, target_column, discarted_columns):
    """
    Make the preprocessing of the dataset, separating features (X) and target (y).

    Args:
        df (pd.DataFrame): Complete dataset.
        target_column (str): Name of the column that is the target (class).
        discarted_columns (list): List of columns to discard (do not use as features).

    Returns:
        tuple: (X, y_encoded, encoder)
            - X: DataFrame with the features
            - y_encoded: NumPy array with the encoded classes
            - encoder: Fitted LabelEncoder (useful for decoding predictions later)
    """
    # Separate features (X) and target (y)
    # Remove the columns that should not be used as features
    X = df.drop(columns=discarted_columns)
    y = df[target_column]

    # Encode the class labels from strings to numbers (0, 1, 2, ...)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    return X, y_encoded, encoder