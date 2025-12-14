import re
import json
from pathlib import Path
from typing import List

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


EMOTION_COLUMNS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral"
]

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text_v2(text: str) -> str:
    text = clean_text(text)
    tokens = []
    for word in text.split():
        if word not in STOPWORDS:
            tokens.append(lemmatizer.lemmatize(word))
    return " ".join(tokens)

def preprocess_and_save(
    raw_csv_path: str,
    output_dir: str = "data/preprocessed",
) -> None:
    """
    Загружает сырой датасет, чистит его и сохраняет результат на диск.
    """
    df = pd.read_csv(raw_csv_path)
    df = df.dropna(subset=["text"])

    if "example_very_unclear" in df.columns:
        df = df[df["example_very_unclear"] != 1]

    # v1 — clean
    df_v1 = df.copy()
    df_v1["text"] = df_v1["text"].apply(clean_text)
    df_v1 = df_v1[["text"] + EMOTION_COLUMNS]

    # v2 — lemma + stopwords
    df_v2 = df.copy()
    df_v2["text"] = df_v2["text"].apply(clean_text_v2)
    df_v2 = df_v2[["text"] + EMOTION_COLUMNS]

    # директория
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # сохраняем полный v1 (если нужен)
    df_v1.to_csv(output_path / "dataset.csv", index=False)

    # одинаковый сэмпл для честного сравнения
    df_v1_30k = df_v1.sample(n=30000, random_state=42)
    df_v2_30k = df_v2.loc[df_v1_30k.index]

    df_v1_30k.to_csv(output_path / "30k_dataset.csv", index=False)
    df_v2_30k.to_csv(output_path / "30k_dataset_v2_lemma.csv", index=False)


if __name__ == "__main__":
    preprocess_and_save(
        raw_csv_path="data/raw/go_emotions_dataset.csv",
        output_dir="data/preprocessed",
    )