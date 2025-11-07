from fastapi import APIRouter
from pydantic import BaseModel
from src.inference.model_loader import ModelLoader

router = APIRouter()

class TextInput(BaseModel):
    text: str

# Загружаем модель и векторизатор один раз при импорте
model, vectorizer = ModelLoader.get_model()

@router.post("/predict")
def predict(input_data: TextInput):
    text = input_data.text
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return {"text": text, "predicted_emotion": prediction}
