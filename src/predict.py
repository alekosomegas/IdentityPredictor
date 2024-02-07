from transformers import AutoTokenizer, pipeline
from src.model import TFAutoModelForSequenceClassification


def load_model(model_path):
    try:
        model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        return model
    except Exception as e:
        print(f"Error in loading model: {e}")
        return None


def make_prediction(model, text):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    predictor = pipeline('text-classification', model=model, tokenizer=tokenizer)
    prediction = predictor(text)
    return prediction
