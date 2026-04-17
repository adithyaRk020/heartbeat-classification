from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_curves(history: Dict[str, Sequence[float]], output_dir: str | Path, model_name: str) -> None:
    output_path = _ensure_output_dir(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.get("loss", []), label="Train Loss")
    axes[0].plot(history.get("val_loss", []), label="Validation Loss")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="Train Accuracy")
    axes[1].plot(history.get("val_accuracy", []), label="Validation Accuracy")
    axes[1].set_title(f"{model_name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path / f"{model_name.lower().replace(' ', '_')}_training_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | Path,
    model_name: str,
) -> None:
    output_path = _ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_classification_report(report_dict: Dict[str, Dict[str, float]], output_dir: str | Path, model_name: str) -> None:
    output_path = _ensure_output_dir(output_dir)
    rows = []
    index = []
    for key, value in report_dict.items():
        if isinstance(value, dict) and {"precision", "recall", "f1-score"}.issubset(value.keys()):
            index.append(key)
            rows.append([value["precision"], value["recall"], value["f1-score"]])

    if not rows:
        return

    data = np.array(rows)
    fig, ax = plt.subplots(figsize=(8, max(3, len(index) * 0.5)))
    sns.heatmap(data, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=["precision", "recall", "f1-score"], yticklabels=index, ax=ax)
    ax.set_title(f"{model_name} Classification Report")
    fig.tight_layout()
    fig.savefig(output_path / f"{model_name.lower().replace(' ', '_')}_classification_report.png", dpi=150)
    plt.close(fig)
