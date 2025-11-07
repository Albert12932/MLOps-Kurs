from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

# создание приложения
app = FastAPI(title="Emotion API", version="0.1")

class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    emotions: List[str]

# эндпоинт для теста, пока ничего полезного не дает
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # пока просто возвращаем "neutral" для всех текстов
    predictions = ["neutral" for _ in req.texts]
    return {"emotions": predictions}

# запускаем сервер слушать 8000 порт
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
