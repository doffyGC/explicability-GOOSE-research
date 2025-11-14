import pandas as pd

def load_dataset(file_path, sample_size=None, random_state=42):
    """
    Carrega um dataset de um arquivo, com opção de amostragem pra datasets grandes.

    Args:
        file_path (str): Caminho pro arquivo do dataset.
        sample_size (int, optional): Se especificado, faz amostragem aleatória quando o dataset
                                     for maior que esse valor. Útil pra datasets gigantes.
        random_state (int): Seed pra reprodutibilidade da amostragem.

    Returns:
        pd.DataFrame: Dataset carregado como DataFrame do pandas (possivelmente amostrado).
    """
    # Carrega o dataset baseado na extensão do arquivo
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, encoding='utf-7')
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("Formato de arquivo não suportado. Use .csv ou .parquet.")

    # Se o dataset for muito grande, faz amostragem estratificada
    # Isso ajuda a economizar memória e tempo de processamento
    if sample_size and len(df) > sample_size:
        print(f"⚠️  Dataset tem {len(df)} linhas. Fazendo amostragem de {sample_size} linhas...")
        df = df.sample(n=sample_size, random_state=random_state)
        print(f"✓ Amostragem concluída. Usando {len(df)} linhas.")

    return df
