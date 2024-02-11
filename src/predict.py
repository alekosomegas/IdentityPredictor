from transformers import pipeline

def make_prediction(model_path, text):
    predictor = pipeline('text-classification', model=model_path, tokenizer=model_path)
    prediction = predictor(text)
    return prediction
