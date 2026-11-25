from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics import confusion_matrix


# LabML7_RBF.py
def plot_confusion_matrix(y_true, y_pred, title, output_path, show=False, cmap="coolwarm"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Без show создаём изображение без открытия графического окна.
    fig = plt.figure(figsize=(7, 6)) if show else Figure(figsize=(7, 6))
    try:
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        fig.colorbar(im, ax=ax)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12, fontweight="bold",
                )

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_title(title)
        ax.set_xticks(np.arange(3), ["low", "medium", "high"])
        ax.set_yticks(np.arange(3), ["low", "medium", "high"])
        fig.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return output_path
    finally:
        plt.close(fig)
