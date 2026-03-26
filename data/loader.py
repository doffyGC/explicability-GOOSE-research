import pandas as pd

def load_dataset(file_path):
    """
    Loads a dataset from a file.
    Carrega um dataset de um arquivo, com opção de amostragem pra datasets grandes.

    Args:
        file_path (str): File path of the dataset.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    # Load the dataset based on the file extension
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, encoding='utf-8')
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("File format not supported. Use .csv or .parquet.")

    return df