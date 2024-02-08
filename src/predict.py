from transformers import AutoTokenizer, pipeline
from src.model import TFAutoModelForSequenceClassification


def load_model(model_path):
    try:
        model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        return model
    except Exception as e:
        print(f"Error in loading model: {e}")
        return None


def make_prediction(model_path, text):
    predictor = pipeline('text-classification', model=model_path, tokenizer=model_path)
    prediction = predictor(text)
    return prediction
