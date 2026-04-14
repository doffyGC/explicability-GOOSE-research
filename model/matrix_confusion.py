import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os


def plot_confusion_matrix(cm, labels, out_path, delta_features=False, figsize=(10, 7)):
    """
    Plot and save a confusion matrix with a smooth logarithmic scale, 
    maintaining the visual scheme used previously (cividis cmap, annotations with dynamic color).

    Args:
        cm (array-like): Confusion matrix (2D).
        labels (list): List of class labels (order matters).
        out_path (str): Path to save the SVG (directory will be created if necessary).
        delta_features (bool): If True, title indicates "With Delta Features", otherwise "Without".
        figsize (tuple): Figure size.
    """

    # Generate a DataFrame for better handling of labels and values
    df = pd.DataFrame(cm, index=labels, columns=labels)

    # Prepare output directory
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # Plot
    plt.figure(figsize=figsize)
    norm = LogNorm(vmin=max(df.values.min(), 1), vmax=df.values.max())
    cmap = plt.cm.cividis

    img = plt.imshow(df.replace(0, 1), norm=norm, cmap=cmap)
    plt.colorbar()

    tick_label_fontsize = 11
    cell_text_fontsize = 12

    plt.xticks(range(len(labels)), labels, rotation=0, ha='center', fontsize=tick_label_fontsize)
    plt.yticks(range(len(labels)), labels, fontsize=tick_label_fontsize)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    title = "Confusion Matrix With Delta Features" if not delta_features else "Confusion Matrix Without Delta Features"
    plt.title(title)

    # Annotate values with dynamic text color for visibility
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = int(df.iloc[i, j])
            rgba = cmap(norm(max(value, 1)))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if brightness < 0.5 else "black"
            plt.text(j, i, str(value), ha="center", va="center", color=text_color, fontsize=cell_text_fontsize)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', format='svg')
    plt.close()