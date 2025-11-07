import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Загружаем конфигурацию
with open("configs/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Настройки MLflow
mlflow.set_experiment(config["experiment_name"])

# Загружаем данные
df = pd.read_csv(config["data"]["path"], names=["text", "emotion"], sep=';')
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["emotion"], test_size=0.2, random_state=42
)

# Преобразование текста
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Обучаем модель
with mlflow.start_run(run_name=config["run_name"]):
    model_params = config["model"]["params"]
    model = LogisticRegression(**model_params)
    model.fit(X_train_vec, y_train)

    # Предсказания
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    # Логирование в MLflow
    mlflow.log_params(model_params)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    # Сохраняем артефакты локально
    joblib.dump(model, config["output"]["model_path"])
    print(f"Model saved to {config['output']['model_path']}")
    print(f"Accuracy: {acc}")
