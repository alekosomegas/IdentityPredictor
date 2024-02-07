import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, Pipeline, pipeline
import numpy as np

def load_data(dataset_path):
    try:
        df = pd.read_excel(dataset_path, engine='openpyxl')
        # use only marker 1 column and change labels to Group Identity = 1, Individual identity = 0)
        df['Marker 1'] = df['Marker 1'].apply(lambda x: 1 if x == 'Group' else 0)
        df = df.rename(columns={'Marker 1': 'labels'})
        df.drop(['Marker 2'], inplace=True, axis=1)
        return df
    except Exception as e:
        print(f"Error when loading data: {e}")
        raise e


def split_and_tokenize(df):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # split the data
    train_text, test_text, train_labels, test_labels = (
        train_test_split(
            df['text'], 
            df['labels'], 
            test_size=0.2, 
            random_state=42
        ))    
    # tokenize the text
    train_data = tokenizer(train_text.tolist(), return_tensors="tf", padding=True)
    test_data = tokenizer(test_text.tolist(), return_tensors="tf", padding=True)
    return dict(train_data), dict(test_data), np.array(train_labels), np.array(test_labels)


def create_model():
    print('Creating model...')
    id2label = {0: "Individual Identity", 1: "Group Identity"}
    label2id = {"Individual Identity": 0, "Group Identity": 1}
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        id2label=id2label, 
        label2id=label2id, 
        num_labels=2
    )
    model.compile()
    return model


def train(model, train_data, train_labels, test_data, test_labels):
    print('Training the model...')
    model.fit(train_data, train_labels)
    results = model.evaluate(test_data, test_labels)
    print('Model is trained and evaluated with accuracy: ', results)


def build_model(dataset_path):
    df = load_data(dataset_path)
    train_text, test_text, train_labels, test_labels = split_and_tokenize(df)
    model = create_model()
    train(model, train_text, train_labels, test_text, test_labels)
    try:
        model.save_pretrained('models')
        print('Model saved successfully.')
    except Exception as e:
        print(f"Error in saving model: {e}")
    return model