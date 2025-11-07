import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_PATH = "data/raw/emotions.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "third_test_model.pkl")

# параметры эксперимента
params = {
    "test_size": 0.2,
    "random_state": 42,
    "ngram_range": (1, 2),
    "max_features": 20000,
    "C": 1.0,
    "max_iter": 300,
}

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Файл {DATA_PATH} не найден.")

df = pd.read_csv(DATA_PATH, sep=";")
text_col = "text"
target_col = "emotion"

# Делим выборку на train/test
X_train, X_test, y_train, y_test = train_test_split(
    df[text_col],
    df[target_col],
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=df[target_col],
)

# Создаем пайплайн

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=params["ngram_range"], max_features=params["max_features"])),
    ("clf", LogisticRegression(C=params["C"], max_iter=params["max_iter"]))
])

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("emotion-classifier")

# Запускаем MLFlow
with mlflow.start_run():
    mlflow.log_params(params)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})

    report = classification_report(y_test, y_pred, digits=4)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    mlflow.sklearn.log_model(pipe, artifact_path="model")

    print("Accuracy:", acc)
    print("F1_macro:", f1)
    print("Модель сохранена в", MODEL_PATH)