from __future__ import annotations
from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_pipeline(numeric: List[str], categorical: List[str], model_type: str = "logreg") -> Pipeline:
    """Build a preprocessing + model pipeline for tabular data.

    Parameters
    ----------
    numeric : list of numeric column names
    categorical : list of categorical column names
    model_type : "logreg" or "random_forest"
    """
    num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num, numeric),
            ("cat", cat, categorical)
        ]
    )

    if model_type == "logreg":
        model = LogisticRegression(max_iter=500)
    elif model_type == "random_forest":
        model = RandomForestClassifier()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline(steps=[("pre", pre), ("model", model)])