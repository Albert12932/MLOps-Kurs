import torch
import torch.nn.functional as F
from .model_loader import ModelLoader

def predict_emotions(text: str) -> dict:
    model, tokenizer, label_names, thresholds = ModelLoader.get_model()
    inputs = tokenizer(text, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze(0)
    emotions = {label: prob.item() for label, prob in zip(label_names, probs)}
    predicted_emotions = [label for label, prob, thresh in zip(label_names, probs, thresholds) if prob >= thresh]
    return {
        "emotions": emotions,
        "predicted_emotions": predicted_emotions
    }
