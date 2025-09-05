from __future__ import annotations
import argparse
import os
import yaml
import numpy as np
import pandas as pd

import mlflow
import joblib
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
)

from src.utils import load_csv, plot_and_save_roc, plot_and_save_pr, plot_and_save_confusion, ensure_dir


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]

    df = load_csv(data_cfg["csv_path"])

    y = df[data_cfg["target"]]
    if y.dtype == object:
        y = y.map({"No": 0, "Yes": 1}).fillna(y)

    X = df[feat_cfg["numeric"] + feat_cfg["categorical"]]

    model_path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Did you run train?")

    model = joblib.load(model_path)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
    }

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-exp")
    mlflow.set_experiment(exp_name)

    ensure_dir("artifacts")
    roc_path = "artifacts/roc_curve.png"
    pr_path = "artifacts/pr_curve.png"
    cm_path = "artifacts/confusion_matrix.png"

    plot_and_save_roc(y, y_proba, roc_path)
    plot_and_save_pr(y, y_proba, pr_path)
    plot_and_save_confusion(y, y_pred, cm_path, labels=("No", "Yes"))

    with mlflow.start_run(run_name="final_evaluation"):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(pr_path)
        mlflow.log_artifact(cm_path)

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)