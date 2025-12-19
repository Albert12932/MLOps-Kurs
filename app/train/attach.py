from pathlib import Path
import mlflow
import pandas as pd

RUN_ID = "940d2894c27e4e1590d625014fc5e4a3"
DATASET_PATH = "data/preprocessed/30k_dataset.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")

with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_artifact(
        DATASET_PATH,
        artifact_path="dataset"
    )

    df = pd.read_csv(DATASET_PATH)
    preview_path = PROJECT_ROOT / "dataset_preview.csv"
    df.head(20).to_csv(preview_path, index=False)

    mlflow.log_artifact(
        preview_path,
        artifact_path="dataset"
    )