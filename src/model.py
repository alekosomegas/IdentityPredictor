import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
import tensorflow as tf
from datasets import Dataset
from keras.callbacks import EarlyStopping
import os

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


def split_and_tokenize(df, model_path):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = Dataset.from_pandas(df)
    # Split the Dataset
    dataset_split = dataset.train_test_split(test_size=0.3, seed=42)
    print(dataset_split)

    def tokenize(data):
        return tokenizer(data["text"], truncation=True, padding='longest')
    
    tokenized_dataset = dataset_split.map(tokenize, batched=True)
    try:
        tokenizer.save_pretrained(model_path)
    except Exception as e: 
        print(f"Error in saving tokenizer: {e}")

    return tokenized_dataset['train'], tokenized_dataset['test'], np.array(dataset_split['train']['labels']), np.array(dataset_split['test']['labels'])


def create_model():
    learning_rate = 3e-5
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = 'sparse_categorical_crossentropy'
    metrics = tf.metrics.SparseCategoricalAccuracy()

    print('Creating model...')

    id2label = {0: "Individual Identity", 1: "Group Identity"}
    label2id = {"Individual Identity": 0, "Group Identity": 1}
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        id2label=id2label, 
        label2id=label2id, 
        num_labels=2
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)
    print(model.summary())
    return model

    
def train(model, train_data, train_labels, test_data, test_labels):
    epochs = 40
    batch_size = 64

    print('Training the model...')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        [np.array(train_data['input_ids']), np.array(train_data['attention_mask'])], train_labels, 
        epochs=epochs, batch_size=batch_size, 
        validation_data=([np.array(test_data['input_ids']), np.array(test_data['attention_mask'])], test_labels), 
        callbacks=[early_stopping])
    
    # save the history
    history_df = pd.DataFrame(history.history)
    return history_df
    

def build_model(dataset_path, model_path):
    df = load_data(dataset_path)
    train_data, test_data, train_labels, test_labels = split_and_tokenize(df, model_path)
    model = create_model()
    history = train(model, train_data, train_labels, test_data, test_labels)
    try:
        model.save_pretrained(model_path)
        history.to_csv(os.path.join(model_path, 'history.csv'), index=False)
        print(f'Model ({model_path}) saved successfully.')
    except Exception as e:
        print(f"Error in saving model: {e} exiting...")
        raise e
