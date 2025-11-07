import shap
import matplotlib.pyplot as plt
import os

def run_shap(model, X_test, class_names, dataset_name, path_base, graphics):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    for graphic in graphics:
        for i, cls in enumerate(class_names):
            save_path = os.path.join(path_base, graphic)
            os.makedirs(save_path, exist_ok=True)
            plt.figure(figsize=(12, 8))
            plt.title(f"{cls} Class {graphic}")
            
            match graphic:
                case "Violin Summary Plot":
                    shap.plots.violin(shap_values[:,:, i], max_display=20, feature_names=X_test.columns, show=False)
                case "Bar Plot":
                    shap.plots.bar(shap_values[:,:, i], max_display=20, show=False)     
                case "Beeswarm Summary Plot":
                    shap.plots.beeswarm(shap_values[:,:, i], max_display=20, show=False)
                case "Waterfall Summary Plot":
                    shap.plots.waterfall(shap_values[0,:, i], max_display=20, show=False)

            plt.savefig(os.path.join(save_path, f"{graphic} dataset {dataset_name} class {cls}.png"), dpi=300)
            plt.close()