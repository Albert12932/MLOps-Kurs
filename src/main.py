from fastapi import FastAPI
from src.api.routes import router

# Инициализация приложения
app = FastAPI(
    title="Emotion Detection API",
    description="API для предсказания эмоциональной окраски текста",
    version="1.1"
)

# Подключаем маршруты
app.include_router(router)

@app.get("/")
def root():
    return {"message": "Emotion Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
