import shap
import matplotlib.pyplot as plt
import os
import gc

def run_shap(model, X_test, class_names, dataset_name, path_base, graphics, sample_percentage=0.20, random_state=42):
    """
    Executa análise SHAP pra explicar as predições do modelo.

    IMPORTANTE: SHAP é MUITO pesado computacionalmente!
    - Pra datasets grandes, usa amostragem baseada em porcentagem
    - Mesmo com 10-20% do dataset, os resultados são representativos
    - Gera gráficos pra cada classe e salva em arquivos PNG

    Args:
        model: Modelo XGBoost treinado.
        X_test (pd.DataFrame): Conjunto de teste (ou validação) pra explicar.
        class_names (list): Lista com os nomes das classes.
        dataset_name (str): Nome do dataset (usado no nome dos arquivos).
        path_base (str): Diretório base pra salvar os gráficos.
        graphics (list): Lista com os tipos de gráficos SHAP pra gerar.
        sample_percentage (float): Porcentagem do dataset pra usar no SHAP (padrão: 0.20 = 20%).
                                   Usa None pra processar todo o dataset.
        random_state (int): Seed pra reprodutibilidade da amostragem.

    Returns:
        None (salva os gráficos em arquivos)
    """

    print("=" * 60)
    print("EXPLICABILIDADE: Análise SHAP")
    print("=" * 60)

    # Calcula o número de amostras baseado na porcentagem
    # SHAP é O(n²) em complexidade, então fica impraticável com muitos dados
    if sample_percentage is not None and sample_percentage < 1.0:
        n_samples = int(len(X_test) * sample_percentage)
        print(f"⚠️  Conjunto de teste tem {len(X_test)} amostras.")
        print(f"    Fazendo amostragem de {n_samples} amostras ({sample_percentage*100:.0f}% do dataset) pro SHAP...")
        print(f"    (isso é normal e recomendado pra economizar tempo!)")
        X_test_sample = X_test.sample(n=n_samples, random_state=random_state)
    else:
        print(f"✓ Usando todo o conjunto de teste ({len(X_test)} amostras) pro SHAP.")
        X_test_sample = X_test

    # Cria o explainer SHAP
    # TreeExplainer é otimizado pra modelos baseados em árvores (XGBoost, RF, etc)
    print("\nCriando explainer SHAP...")
    explainer = shap.Explainer(model)

    # Calcula os valores SHAP
    # Isso pode demorar alguns minutos dependendo do tamanho do dataset!
    print(f"Calculando valores SHAP pra {len(X_test_sample)} amostras...")
    print("(pode demorar um pouco...)")
    shap_values = explainer(X_test_sample)
    print("✓ Valores SHAP calculados!")
    print()

    # Detecta se é classificação binária ou multiclasse
    # Binária: shap_values tem shape (n_samples, n_features)
    # Multiclasse: shap_values tem shape (n_samples, n_features, n_classes)
    is_binary = len(shap_values.shape) == 2 or (hasattr(shap_values, 'values') and len(shap_values.values.shape) == 2)

    if is_binary:
        print("✓ Classificação binária detectada (SHAP em 2D)")
        print("  Gerando gráficos apenas pra classe positiva (mais comum)")
        print()
    else:
        print("✓ Classificação multiclasse detectada (SHAP em 3D)")
        print()

    # Gera os gráficos
    print("Gerando gráficos SHAP...")

    if is_binary:
        # ========================================
        # CLASSIFICAÇÃO BINÁRIA - GRÁFICOS PRA AMBAS AS CLASSES
        # ========================================
        # Em binário, SHAP retorna valores pra classe positiva (índice 1)
        # Mas podemos gerar gráficos pras DUAS classes:
        #   1. Classe Positiva: usa os valores SHAP como estão
        #   2. Classe Negativa: inverte os valores SHAP (multiplica por -1)
        #
        # EXCEÇÃO: Bar Plot mostra |SHAP| (valor absoluto), então é igual pras duas classes
        #          Vamos gerar apenas UMA VEZ

        # Conta quantos gráficos vão ser gerados
        bar_plot_in_graphics = "Bar Plot" in graphics
        total_graphics = len(graphics) * 2  # Gráficos direcionais pras duas classes
        if bar_plot_in_graphics:
            total_graphics -= 1  # Bar Plot só uma vez
        current = 0

        # PRIMEIRO: Gera o Bar Plot UMA VEZ (importância geral, não depende de classe)
        if bar_plot_in_graphics:
            current += 1
            print(f"  [{current}/{total_graphics}] Gerando Bar Plot (importância geral das features)...")

            save_path = os.path.join(path_base, "Bar Plot")
            os.makedirs(save_path, exist_ok=True)

            plt.figure(figsize=(12, 8))
            plt.title(f"Bar Plot - Features General Importance\n(mean(|SHAP|))")

            # Bar plot usa valor absoluto, então não importa inverter
            shap.plots.bar(shap_values, max_display=20, show=False)

            filename = f"Bar Plot dataset {dataset_name} (geral).png"
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close()
            gc.collect()

        # SEGUNDO: Gera os gráficos DIRECIONAIS pra cada classe
        # (Violin, Beeswarm, Waterfall mostram direção dos valores SHAP)
        for class_idx, cls in enumerate(class_names):
            for graphic in graphics:
                # Pula o Bar Plot (já foi gerado acima)
                if graphic == "Bar Plot":
                    continue

                current += 1
                print(f"  [{current}/{total_graphics}] Gerando {graphic} pra classe '{cls}'...")

                # Cria o diretório pro tipo de gráfico
                save_path = os.path.join(path_base, graphic)
                os.makedirs(save_path, exist_ok=True)

                # Configura a figura
                plt.figure(figsize=(12, 8))

                # IMPORTANTE: Inverte os valores SHAP pra classe negativa (índice 0)
                # Isso mostra "o que contribui pra prever essa classe"
                if class_idx == 0:
                    # Classe negativa (FRG/Maligno): inverte os valores
                    # Agora valores POSITIVOS = contribuem pra FRG
                    shap_values_class = shap_values * -1
                    plt.title(f"{graphic} - {cls} (Negative class)\nPositive values = contribute to {cls}")
                else:
                    # Classe positiva (Normal): usa valores originais
                    # Valores POSITIVOS = contribuem pra Normal
                    shap_values_class = shap_values
                    plt.title(f"{graphic} - {cls} (Positive class)\nPositive values = contribute to {cls}")

                # Gera o tipo de gráfico apropriado
                match graphic:
                    case "Violin Summary Plot":
                        # Mostra a distribuição dos valores SHAP pra cada feature
                        shap.plots.violin(shap_values_class, max_display=20, show=False)

                    case "Beeswarm Summary Plot":
                        # Visualização densa mostrando valor SHAP vs valor da feature
                        shap.plots.beeswarm(shap_values_class, max_display=20, show=False)

                    case "Waterfall Summary Plot":
                        # Mostra a contribuição de cada feature numa ÚNICA predição
                        # Usa a primeira amostra como exemplo
                        shap.plots.waterfall(shap_values_class[0], max_display=20, show=False)

                # Salva o gráfico em alta resolução
                filename = f"{graphic} dataset {dataset_name} class {cls}.png"
                full_path = os.path.join(save_path, filename)
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                plt.close()

                # Limpa memória pra não estourar
                gc.collect()

        print(f"\n✓ {total_graphics} gráficos SHAP salvos em: {path_base}")

    else:
        # ========================================
        # CLASSIFICAÇÃO MULTICLASSE
        # ========================================
        total_graphics = len(graphics) * len(class_names)
        current = 0

        for graphic in graphics:
            for i, cls in enumerate(class_names):
                current += 1
                print(f"  [{current}/{total_graphics}] Gerando {graphic} pra classe '{cls}'...")

                # Cria o diretório pro tipo de gráfico
                save_path = os.path.join(path_base, graphic)
                os.makedirs(save_path, exist_ok=True)

                # Configura a figura
                plt.figure(figsize=(12, 8))
                plt.title(f"{cls} Class {graphic}")

                # Gera o tipo de gráfico apropriado
                match graphic:
                    case "Violin Summary Plot":
                        # Mostra a distribuição dos valores SHAP pra cada feature
                        shap.plots.violin(shap_values[:,:, i], max_display=20,
                                         feature_names=X_test_sample.columns, show=False)

                    case "Bar Plot":
                        # Mostra a importância média (|SHAP|) de cada feature
                        shap.plots.bar(shap_values[:,:, i], max_display=20, show=False)

                    case "Beeswarm Summary Plot":
                        # Visualização densa mostrando valor SHAP vs valor da feature
                        shap.plots.beeswarm(shap_values[:,:, i], max_display=20, show=False)

                    case "Waterfall Summary Plot":
                        # Mostra a contribuição de cada feature numa ÚNICA predição
                        # Usa a primeira amostra como exemplo
                        shap.plots.waterfall(shap_values[0,:, i], max_display=20, show=False)

                # Salva o gráfico em alta resolução
                filename = f"{graphic} dataset {dataset_name} class {cls}.png"
                full_path = os.path.join(save_path, filename)
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                plt.close()

                # Limpa memória pra não estourar
                gc.collect()

        print(f"\n✓ {total_graphics} gráficos SHAP salvos em: {path_base}")

    print()