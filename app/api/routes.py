from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List
import torch
from app.inference.model_loader import ModelLoader

router = APIRouter()

class TextInput(BaseModel):
    text: str = Field(
        ...,
        description="Входной текст для определения эмоций",
        example="I feel excited but also a little nervous"
    )

class EmotionScore(BaseModel):
    label: str = Field(..., description="Название эмоции")
    score: float = Field(..., description="Вероятность эмоции")

class PredictionResponse(BaseModel):
    text: str
    predicted_emotions: list[EmotionScore]
    emotions: List[EmotionScore]

model, tokenizer, label_names, thresholds = ModelLoader.get_model()
model.eval()

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Предсказание эмоций по тексту",
    description=(
        "Эндпоинт принимает текст и возвращает эмоции с вероятностями. "
        "Поддерживает фильтрацию по threshold и ограничение top_k."
    ),
)
def predict(input_data: TextInput, top_k: int = 3, threshold: float = 0.2):
    text = input_data.text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze(0)
    emotions_with_probs = []
    for idx, label in enumerate(label_names):
        prob = probs[idx].item()
        emotions_with_probs.append(EmotionScore(emotion=label, probability=prob))
    filtered = [e for e in emotions_with_probs if e.probability > threshold]
    filtered.sort(key=lambda x: x.probability, reverse=True)
    filtered = filtered[:top_k]
    if not filtered:
        max_emotion = max(emotions_with_probs, key=lambda x: x.probability)
        filtered = [max_emotion]
    predicted_emotions = [e.emotion for e in filtered]
    return PredictionResponse(
        text=text,
        emotions=emotions_with_probs,
        predicted_emotions=predicted_emotions
    )

@router.get(
    "/health",
    summary="Проверка состояния сервиса",
    description="Возвращает статус работы API"
)
def health():
    return {"status": "ok"}