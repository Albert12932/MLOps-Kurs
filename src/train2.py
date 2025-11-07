import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Пути
DATA_PATH = "data/raw/emotions.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "second_test_model.pkl")

# Проверяем, что данные существуют
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Файл {DATA_PATH} не найден."
    )

# Загружаем данные
df = pd.read_csv(DATA_PATH, sep=';')

text_col = "text"
target_col = "emotion"

print(f"Загружено {len(df)} записей, классы: {df[target_col].unique()}")

# Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(
    df[text_col],
    df[target_col],
    test_size=0.2,
    random_state=42,
    stratify=df[target_col],
)

# Собираем пайплайн
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
    ("clf", LogisticRegression(max_iter=300, C=1.0))
])

# Обучаем модель
pipe.fit(X_train, y_train)
print("Модель обучена")

# Оцениваем
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
print(f"Accuracy: {acc:.4f}, F1_macro: {f1:.4f}")

# Отчёт по классам
print("\nКлассификационный отчёт:")
print(classification_report(y_test, y_pred))

# Сохраняем модель
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(pipe, MODEL_PATH)
print(f"Модель сохранена в {MODEL_PATH}")
