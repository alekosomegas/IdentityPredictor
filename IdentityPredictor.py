import os
from src.model import build_model
from src.predict import make_prediction
from datetime import datetime

MODELS_PATH = 'models'
DEFAULT_DATASET_PATH = 'data/Emergency_Expressions.xlsx'

def create_model_name():
    new_model_path = os.path.join(MODELS_PATH, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(new_model_path)
    return new_model_path

def main():
    trained_model_path = None
    # check if folder is empty
    if os.listdir(MODELS_PATH).__len__() == 0:
        print("No models found. Build and train a new model.")
        dataset_path = input("Please enter the path to the dataset (leave empty to use default): ")
        if not dataset_path:
            dataset_path = DEFAULT_DATASET_PATH
        
        trained_model_path = create_model_name()
        build_model(dataset_path, trained_model_path)
    else:
        while True:
            choice = input("Model(s) found. Do you want to use an existing model (y) or train a new one? (n)").lower()
            if choice == 'n':
                trained_model_path = create_model_name()
                build_model(DEFAULT_DATASET_PATH, trained_model_path)
                break
            elif choice == 'y':
                while True:
                    model_path = input("Please enter the path to the model: (YYYYMMDD-HHMMSS) ")
                    # check if model exists
                    if os.path.exists(os.path.join(MODELS_PATH, model_path)):
                        print("Model found. Ready to make predictions.")
                        trained_model_path = os.path.join(MODELS_PATH, model_path)
                        break
                    else:
                        print(f"Model {'model_path'} not found. try again")

    try:
        while True:
            text = input("Please enter some text (or 'Ctrl+C' to exit): ")
            prediction = make_prediction(trained_model_path, text)
            print(f"The prediction for '{text}' is: {prediction}")
    except KeyboardInterrupt:
        print("Exiting...")
    

if __name__ == "__main__":
    main()