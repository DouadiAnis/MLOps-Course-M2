from __future__ import annotations
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, ConfusionMatrixDisplay)


# ------------------------- IO helpers -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_joblib(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    joblib.dump(obj, path)


def load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df


# ------------------------- Plotting helpers -------------------

def plot_and_save_roc(y_true, y_proba, out_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_and_save_pr(y_true, y_proba, out_path: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_and_save_confusion(y_true, y_pred, out_path: str, labels: Tuple[str, str] = ("No", "Yes")) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)