import joblib
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import re
import yaml

# Загружаем конфигурацию
with open("configs/train_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_path = config["data"]["path"]
model_path = config["output"]["model_path"]

# Загружаем данные
df = pd.read_csv(data_path, sep=";", names=["text", "emotion"])
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)
X = df["text"]
y = df["emotion"]

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаём векторизатор и модель
vec_params = config.get("vectorizer", {})
if "ngram_range" in vec_params and isinstance(vec_params["ngram_range"], list):
    vec_params["ngram_range"] = tuple(vec_params["ngram_range"])

vectorizer = TfidfVectorizer(**vec_params)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=200, solver="liblinear")

# Запускаем трекинг через MLflow
mlflow.set_experiment(config["experiment_name"])
with mlflow.start_run(run_name=config["run_name"]):
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1)

    print(f"Accuracy: {acc}")
    print(f"F1_macro: {f1}")

    # Сохраняем артефакты
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Модель и векторизатор сохранены.")
    mlflow.sklearn.log_model(model, artifact_path="model")
