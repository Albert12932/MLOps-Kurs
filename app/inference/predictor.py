from .model_loader import ModelLoader

def predict_emotion(text: str) -> str:
    model, vectorizer = ModelLoader.load()
    X_vec = vectorizer.transform([text])
    prediction = model.predict(X_vec)[0]
    return prediction
