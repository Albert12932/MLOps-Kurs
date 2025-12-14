import argparse
from pathlib import Path
import yaml

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)

    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run(run_name=cfg["run_name"]):

        df = pd.read_csv(cfg["data"]["train_path"])
        mlflow.log_param("dataset_path", cfg["data"]["train_path"])
        mlflow.log_param(
            "dataset_name",
            Path(cfg["data"]["train_path"]).name
        )

        texts = df[cfg["data"]["text_column"]].astype(str).tolist()
        labels = df[cfg["data"]["label_columns"]].values

        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=cfg.get("seed", 42),
        )

        vec_cfg = cfg["vectorizer"]

        vectorizer = TfidfVectorizer(
            max_features=vec_cfg.get("max_features", 20000),
            ngram_range=tuple(vec_cfg.get("ngram_range", [1, 2])),
            lowercase=vec_cfg.get("lowercase", True),
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        base_model = LogisticRegression(
            max_iter=cfg["training"].get("max_iter", 300),
            solver="liblinear",
        )

        model = OneVsRestClassifier(base_model)

        model.fit(X_train_vec, y_train)

        val_probs = model.predict_proba(X_val_vec)
        val_preds = (val_probs > 0.5).astype(int)

        f1_micro = f1_score(y_val, val_preds, average="micro", zero_division=0)
        f1_macro = f1_score(y_val, val_preds, average="macro", zero_division=0)

        mlflow.log_params({
            "model_type": "logistic_regression",
            "max_features": vec_cfg.get("max_features"),
            "ngram_range": vec_cfg.get("ngram_range"),
            "max_iter": cfg["training"].get("max_iter"),
        })

        mlflow.log_artifact(
            cfg["data"]["train_path"],
            artifact_path="dataset"
        )

        mlflow.log_metrics({
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
        })

        output_dir = Path(cfg["output"]["model_path"])
        output_dir.mkdir(parents=True, exist_ok=True)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        mlflow.log_artifact(config_path, artifact_path="config")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to baseline experiment config",
    )
    args = parser.parse_args()

    main(args.config)