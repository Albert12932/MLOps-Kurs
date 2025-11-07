from fastapi import FastAPI
import uvicorn

# создание приложения
app = FastAPI(title="Emotion API", version="0.1")

# эндпоинт для теста, пока ничего полезного не дает
@app.get("/health")
def health():
    return {"status": "ok"}

# запускаем сервер слушать 8000 порт
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
