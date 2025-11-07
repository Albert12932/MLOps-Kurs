from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import uvicorn
import os

# создание приложения
app = FastAPI(title="Emotion API", version="0.1")


MODEL_PATH = os.getenv("MODEL_PATH", "models/second_test_model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Модель загружена из {MODEL_PATH}")
else:
    print(f"Не найден файл модели в {MODEL_PATH}")

class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    emotions: List[str]

# эндпоинт для теста, пока ничего полезного не дает
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="Не удалось получить текст из тела запроса")

    if model is None:
        # если модель не загружена — возвращаем neutral, как заглушку
        predictions = ["neutral" for _ in req.texts]
    else:
        # используем модель для предсказания
        predictions = model.predict(req.texts)

    return {"emotions": [str(p) for p in predictions]}

# запускаем сервер слушать 8000 порт
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
