import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertTokenizer,TFBertForSequenceClassification, TFAutoModelForSequenceClassification, Pipeline, pipeline
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, DatasetDict
from keras.callbacks import EarlyStopping

# check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = Dataset.from_pandas(df)
    # Split the Dataset
    dataset_dict = dataset.train_test_split(test_size=0.3, seed=42)
    print(dataset_dict)
    def tokenize(data):
        return tokenizer(data["text"], truncation=True, padding='longest')
    
    dataset = dataset_dict.map(tokenize, batched=True)
    tokenizer.save_pretrained('models')
    return dataset['train'], dataset['test'], np.array(dataset_dict['train']['labels']), np.array(dataset_dict['test']['labels'])


def create_model():
    print('Creating model...')
    id2label = {0: "Individual Identity", 1: "Group Identity"}
    label2id = {"Individual Identity": 0, "Group Identity": 1}
    model = TFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        id2label=id2label, 
        label2id=label2id, 
        num_labels=2
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss='sparse_categorical_crossentropy',
        metrics=tf.metrics.SparseCategoricalAccuracy())
    print(model.summary())
    return model

    
def train(model, train_data, train_labels, test_data, test_labels):
    print('Training the model...')

    # Initialize the early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    model.fit([np.array(train_data['input_ids']), np.array(train_data['attention_mask'])], train_labels, epochs=40, batch_size=64, 
              validation_data=([np.array(test_data['input_ids']), np.array(test_data['attention_mask'])], test_labels), callbacks=[early_stopping])
    
    return model


def build_model(dataset_path):
    df = load_data(dataset_path)
    train_data, test_data, train_labels, test_labels = split_and_tokenize(df)
    model = create_model()
    model = train(model, train_data, train_labels, test_data, test_labels)
    try:
        model.save_pretrained('models')
        print('Model saved successfully.')
    except Exception as e:
        print(f"Error in saving model: {e}")
    return model