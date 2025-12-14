# main.py
from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Emotion Detection API",
    description="API для предсказания эмоциональной окраски текста",
    version="1.1"
)

app.include_router(router)

@app.get("/health")
def root():
    return {"message": "Emotion Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

# uvicorn app.main:app --host 0.0.0.0 --port 8000