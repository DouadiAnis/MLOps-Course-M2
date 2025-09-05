from __future__ import annotations
import argparse
import os
import yaml
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from pipeline import build_pipeline
from utils import load_csv, save_joblib, ensure_dir


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_param_grid(model_type: str, cfg_params: dict) -> dict:
    # select the sub-grid matching chosen model type
    grid = cfg_params.get(model_type, {})
    if model_type == "logreg":
        return {f"model__{k}": v for k, v in grid.items()}
    if model_type == "random_forest":
        return {f"model__{k}": v for k, v in grid.items()}
    raise ValueError(f"Unsupported model_type: {model_type}")


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]
    cv_cfg = cfg["cv"]

    csv_path = data_cfg["csv_path"]
    target = data_cfg["target"]

    df = load_csv(csv_path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


    # Convert target to binary 0/1 if it's Yes/No like Telco
    y = df[target]
    if y.dtype == object:
        y = y.map({"No": 0, "Yes": 1}).fillna(y)
    X = df[feat_cfg["numeric"] + feat_cfg["categorical"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_cfg["test_size"], random_state=data_cfg["random_state"], stratify=y
    )

    model_type = model_cfg["type"]
    pipe = build_pipeline(feat_cfg["numeric"], feat_cfg["categorical"], model_type=model_type)

    # CV strategy
    if cv_cfg.get("strategy", "StratifiedKFold") == "StratifiedKFold":
        kf = StratifiedKFold(n_splits=cv_cfg["n_splits"], shuffle=cv_cfg.get("shuffle", True), random_state=data_cfg["random_state"])
    else:
        raise ValueError("Only StratifiedKFold is supported in this template")

    param_grid = get_param_grid(model_type, model_cfg["params"])

    mlflow.sklearn.autolog(log_models=True)
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-exp")
    mlflow.set_experiment(exp_name)

    with mlflow.start_run():
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=kf,
            scoring=cv_cfg["scoring"],
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_proba = best_model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
        mlflow.log_metric("test_roc_auc", roc)

        ensure_dir("artifacts")
        model_path = "artifacts/model.joblib"
        save_joblib(best_model, model_path)
        mlflow.log_artifact(model_path)

        # Optional: register model if registry available
        if os.getenv("MLFLOW_REGISTER_MODEL", "false").lower() == "true":
            model_name = os.getenv("MLFLOW_MODEL_NAME", "ChurnClassifier")
            mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name=model_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml")
    args = ap.parse_args()
    main(args.config)