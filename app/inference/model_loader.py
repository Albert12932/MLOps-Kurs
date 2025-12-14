import yaml
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ModelLoader:
    _model = None
    _tokenizer = None
    _label_names = None
    _thresholds = None

    @classmethod
    def load_config(cls):
        config_path = "configs/inference_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    @classmethod
    def get_model(cls):
        if cls._model is None or cls._tokenizer is None:
            config = cls.load_config()
            model_path = config["model"]["path"]

            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"Model directory not found: {model_path}")

            cls._model = AutoModelForSequenceClassification.from_pretrained(model_path)
            cls._model.eval()  # Set model to evaluation mode

            cls._tokenizer = AutoTokenizer.from_pretrained(model_path)

            cls._label_names = config.get("label_names", [])
            cls._thresholds = config.get("thresholds", None)

        return cls._model, cls._tokenizer, cls._label_names, cls._thresholds
