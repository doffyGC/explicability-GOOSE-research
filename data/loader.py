import pandas as pd

def load_dataset(file_path):
    """
    Function to load a dataset from a file.
    
    Args:
        file_path (str): The file path to the dataset.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path, encoding='utf-7')
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .parquet files.")
