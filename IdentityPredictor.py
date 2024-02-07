import os
from src.model import build_model
from src.predict import load_model, make_prediction

def main():
    model_path = 'models'
    # check if folder is empty
    if os.listdir(model_path).__len__() == 0:
        print("No model found. Build and train a new model.")
        dataset_path = input("Please enter the path to the dataset (leave empty to use default): ")
        if not dataset_path:
            dataset_path = 'data/Emergency_Expressions.xlsx'
        model = build_model(dataset_path)
    else:
        model = load_model(model_path)

     
    while True:
            text = input("Please enter some text (or 'quit' to exit): ")
            if text.lower() == 'quit':
                break
            prediction = make_prediction(model, text)
            print(f"The prediction for '{text}' is: {prediction}")
            # correct = input("Was this prediction correct? (yes/no): ")
            # if correct.lower() == 'no':
            #     # Update model based on user feedback
            #     model = update_model(model, text)
            # add to cvs file and re-train model
            

if __name__ == "__main__":
    main()