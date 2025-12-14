import argparse
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import mlflow
import mlflow.pytorch

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split



class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        if text is None or pd.isna(text):
            text = ""
        elif not isinstance(text, str):
            text = str(text)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }



def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)



def main(config_path: str):
    cfg = load_config(config_path)

    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run(run_name=cfg["run_name"]):

        mlflow.log_params({
            "model_name": cfg["model"]["name"],
            "num_epochs": cfg["training"]["num_train_epochs"],
            "batch_size": cfg["training"]["batch_size"],
            "learning_rate": cfg["training"]["learning_rate"],
            "max_length": cfg["tokenizer"]["max_length"],
            "freeze_encoder": cfg["model"].get("freeze_encoder", False),
        })

        df = pd.read_csv(cfg["data"]["train_path"])
        mlflow.log_param("dataset_path", cfg["data"]["train_path"])
        mlflow.log_param(
            "dataset_name",
            Path(cfg["data"]["train_path"]).name
        )

        texts = df[cfg["data"]["text_column"]].tolist()
        labels = df[cfg["data"]["label_columns"]].values

        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=cfg["training"].get("seed", 42),
        )

        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])

        train_ds = EmotionDataset(
            X_train, y_train, tokenizer, cfg["tokenizer"]["max_length"]
        )
        val_ds = EmotionDataset(
            X_val, y_val, tokenizer, cfg["tokenizer"]["max_length"]
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model"]["name"],
            num_labels=len(cfg["data"]["label_columns"]),
            problem_type=cfg["model"]["problem_type"],
        )

        if cfg["model"].get("freeze_encoder", False):
            for p in model.base_model.parameters():
                p.requires_grad = False

        training_args = TrainingArguments(
            output_dir=cfg["output"]["model_dir"],
            run_name=cfg["run_name"],
            num_train_epochs=cfg["training"]["num_train_epochs"],
            per_device_train_batch_size=cfg["training"]["batch_size"],
            per_device_eval_batch_size=cfg["training"]["batch_size"],
            learning_rate=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"].get("weight_decay", 0.0),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_micro",
            logging_steps=100,
            seed=cfg["training"].get("seed", 42),
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()

        mlflow.log_metrics({
            "f1_micro": metrics["eval_f1_micro"],
            "f1_macro": metrics["eval_f1_macro"],
        })
        mlflow.log_artifact(
            cfg["data"]["train_path"],
            artifact_path="dataset"
        )

        output_dir = Path(cfg["output"]["model_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        mlflow.log_artifacts(str(output_dir), artifact_path="model")
        mlflow.log_artifact(config_path, artifact_path="config")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)
