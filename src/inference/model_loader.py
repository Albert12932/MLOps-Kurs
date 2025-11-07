import yaml
import joblib
import os

class ModelLoader:
    _model = None
    _vectorizer = None

    @classmethod
    def load_config(cls, config_path: str = "configs/inference_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def get_model(cls):
        if cls._model is None or cls._vectorizer is None:
            config = cls.load_config()
            model_path = config["model"]["path"]
            vectorizer_path = config["vectorizer"]["path"]

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(vectorizer_path):
                raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

            cls._model = joblib.load(model_path)
            cls._vectorizer = joblib.load(vectorizer_path)
        return cls._model, cls._vectorizer
