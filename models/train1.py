import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

texts = ["I am happy", "I feel sad", "This is bad", "Great experience"]
labels = ["joy", "sadness", "sadness", "joy"]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])
pipe.fit(texts, labels)

import os
os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/first_test_model.pkl")
print("Первая тестовая модель сохранена в models/first_test_model.pkl") 