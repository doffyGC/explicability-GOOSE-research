import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc

def get_ten_most_important_features(shap_values, X):
    """
    Returns the top 10 most important features based on SHAP values.
    
    args:
    - shap_values: SHAP values calculated for the test set 
    - X: DataFrame with the test features (used to get feature names)
    """

    # Extract the SHAP values as a numpy array
    if hasattr(shap_values, "values"):
        shap_array = shap_values.values
    else:
        shap_array = shap_values  # If it's already a numpy array

    # If its multiclass problem → mean across
    if shap_array.ndim == 3:
        shap_array = np.mean(np.abs(shap_array), axis=2)
    else:
        shap_array = np.abs(shap_array)

    # Média por feature
    mean_abs_shap = shap_array.mean(axis=0)

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": mean_abs_shap
    })

    top_10 = feature_importance.nlargest(10, "importance")
    top_10["percentage"] = (top_10["importance"] / top_10["importance"].sum()) * 100

    print(top_10)


def run_shap(model, X_test, class_names, dataset_name, path_base, graphics):
    """
    Execute SHAP analysis to explain the model's predictions.

    IMPORTANT: SHAP is TOO SLOW for large datasets. For real use cases, consider:
    - Using a smaller sample of the test set (e.g., 10-20%) for SHAP analysis
    - Using SHAP's TreeExplainer which is optimized for tree-based models like XGBoost

    Args:
        model: XGBoost model trained.
        X_test (pd.DataFrame): Test set to explain.
        class_names (list): List with the names of the classes.
        dataset_name (str): Name of the dataset (used in the file names).
        path_base (str): Base directory to save the plots.
        graphics (list): List with the types of SHAP plots to generate.

    Returns:
        None (saves the plots to files)
    """

    print("=" * 60)
    print("EXPLAINABILITY: SHAP Analysis")
    print("=" * 60)

    print(f"✓ Using test set ({len(X_test)} samples) for SHAP.")
    X_test_sample = X_test

    # Create the SHAP explainer
    # TreeExplainer is otimized to tree-based models (XGBoost, RF, etc)
    print("\nCreating SHAP explainer...")
    explainer = shap.Explainer(model)

    # Calculate the SHAP values
    print(f"Calculating SHAP values for {len(X_test_sample)} samples...")
    print("(this might take a while...)")
    shap_values = explainer(X_test_sample)
    print("✓ SHAP values calculated!")
    print()
    
    # Verify the 10 most important features
    print("Calculating the 10 most important features based on mean absolute SHAP values...")
    get_ten_most_important_features(shap_values, X_test_sample)

    # Detect if it's binary or multiclass classification based on the shape of SHAP values
    # Binary: shap_values has shape (n_samples, n_features)
    # Multiclass: shap_values has shape (n_samples, n_features, n_classes)
    is_binary = len(shap_values.shape) == 2 or (hasattr(shap_values, 'values') and len(shap_values.values.shape) == 2)

    if is_binary:
        print("✓ Binary classification detected (SHAP in 2D)")
        print("  Generating plots only for the positive class (most common)")
        print()
    else:
        print("✓ Multiclass classification detected (SHAP in 3D)")
        print()

    # Generate the plots
    print("Generating SHAP plots...")

    if is_binary:
        # ========================================
        # BINARY CLASSIFICATION - GRAPHS TO BOTH CLASSES
        # ========================================
        # In binary classification, SHAP returns values for the positive class (index 1)
        # But we can generate plots for BOTH classes:
        #   1. Positive Class: use the SHAP values as they are
        #   2. Negative Class: invert the SHAP values (multiply by -1)
        #
        # EXCEPTION: Bar Plot shows |SHAP| (absolute value), so it doesn't matter if we invert or not.

        # Count how many graphics we will generate to show progress
        bar_plot_in_graphics = "Bar Plot" in graphics
        total_graphics = len(graphics) * 2  # Graphics for both classes
        if bar_plot_in_graphics:
            total_graphics -= 1  # Bar Plot is generated only once
        current = 0

        # FIRST: Generate the Bar Plot at ONCE
        if bar_plot_in_graphics:
            current += 1
            print(f"  [{current}/{total_graphics}] Generating Bar Plot (general feature importance)...")

            save_path = os.path.join(path_base, "Bar Plot")
            os.makedirs(save_path, exist_ok=True)

            plt.figure(figsize=(12, 8))
            plt.title(f"Bar Plot - Features General Importance\n(mean(|SHAP|))")

            shap.plots.bar(shap_values, max_display=20, show=False)

            filename = f"Bar Plot dataset {dataset_name} (geral).png"
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close()
            gc.collect()

        # SECOND: Generate the DIRECTIONAL plots for each class
        for class_idx, cls in enumerate(class_names):
            for graphic in graphics:
                current += 1
                print(f"  [{current}/{total_graphics}] Generating {graphic} for class '{cls}'...")

                # Create the directory for this type of graphic
                save_path = os.path.join(path_base, graphic)
                os.makedirs(save_path, exist_ok=True)

                # Configure the figure
                plt.figure(figsize=(12, 8))

                # IMPORTANT: Invert the SHAP values for the negative class to show the correct direction of contribution
                # This way, for the negative class (e.g., Attack), positive SHAP values will indicate features that contribute to that class, and negative SHAP values will indicate features that contribute to the opposite class (Normal).
                if class_idx == 0:
                    # Negative Class (Attack): invert the values
                    # Now the positive values = help to predict Attack
                    shap_values_class = shap_values * -1
                    plt.title(f"{graphic} - {cls} (Negative class)\nPositive values = contribute to {cls}")
                else:
                    # Positive Class (Normal): use original values
                    # Positive values = contribute to Normal
                    shap_values_class = shap_values
                    plt.title(f"{graphic} - {cls} (Positive class)\nPositive values = contribute to {cls}")

                # Generate the appropriate type of graphic
                match graphic:
                    case "Beeswarm Summary Plot":
                        # View showing SHAP value vs feature value for each feature and sample
                        shap.plots.beeswarm(shap_values_class, max_display=20, show=False)
                        
                # Save the graph in high resolution
                filename = f"{graphic} - {cls}.png"
                full_path = os.path.join(save_path, filename)
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                plt.close()

                # Clean memory to avoid crashes with large datasets
                gc.collect()

        print(f"\n✓ {total_graphics} SHAP plots generated in: {path_base}")

    else:
    # ========================================
    # MULTICLASS CLASSIFICATION
    # ========================================

        bar_plot_in_graphics = "Bar Plot" in graphics

        # Total correto de gráficos
        total_graphics = len(graphics) * len(class_names)
        if bar_plot_in_graphics:
            total_graphics -= (len(class_names) - 1)  # Bar plot só uma vez

        current = 0

        # ========================================
        # 1. BAR PLOT (APENAS UMA VEZ)
        # ========================================
        if bar_plot_in_graphics:
            current += 1
            print(f"  [{current}/{total_graphics}] Generating Bar Plot (global feature importance)...")

            save_path = os.path.join(path_base, "Bar Plot")
            os.makedirs(save_path, exist_ok=True)

            shap.summary_plot(
                shap_values,
                X_test,
                plot_type="bar",
                class_names=class_names,
                show=False
            )

            filename = f"Bar_Plot - {dataset_name}.png"
            full_path = os.path.join(save_path, filename)

            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            gc.collect()

        # ========================================
        # 2. OUTROS GRÁFICOS (POR CLASSE)
        # ========================================
        for graphic in graphics:
            if graphic == "Bar Plot":
                continue  # já foi gerado

            for i, cls in enumerate(class_names):
                current += 1
                print(f"  [{current}/{total_graphics}] Generating {graphic} for class '{cls}'...")

                save_path = os.path.join(path_base, graphic)
                os.makedirs(save_path, exist_ok=True)

                if graphic == "Beeswarm Summary Plot":
                    shap.plots.beeswarm(shap_values[..., i], max_display=20, show=False)

                filename = f"{graphic} - {cls}.png".replace(" ", "_")
                full_path = os.path.join(save_path, filename)

                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                plt.close('all')
                gc.collect()

        print(f"\n✓ {total_graphics} SHAP plots generated in: {path_base}")

    print()