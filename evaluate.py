from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from visualize import plot_classification_report, plot_confusion_matrix


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | Path,
    model_name: str,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    label_ids = list(range(len(class_names)))
    cm = confusion_matrix(y_test, y_pred, labels=label_ids)
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=label_ids,
        target_names=list(class_names),
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=label_ids,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )

    np.save(output_dir / "confusion_matrix.npy", cm)
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    plot_confusion_matrix(cm, class_names, output_dir, model_name=model_name)
    plot_classification_report(report_dict, output_dir, model_name=model_name)

    return {
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }
