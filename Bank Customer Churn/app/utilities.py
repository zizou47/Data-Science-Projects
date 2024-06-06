import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load("saved_model/RF.joblib")  # Ensure the path is correct
print("Model loaded successfully")

def preprocess_input(data):
    # Assuming the same preprocessing steps as during training
    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
    data['country'] = data['country'].map({'France': 1, 'Spain': 0, 'Germany': 2})
    
    numeric_columns = ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data

def prediction_pipeline(data):
    # Preprocess the input data
    data_preprocessed = preprocess_input(data)
    
    # Ensure the features match
    if data_preprocessed.shape[1] != model.n_features_in_:
        raise ValueError(f"Model expects {model.n_features_in_} features, but got {data_preprocessed.shape[1]}")
    
    # Make a prediction
    prediction = model.predict(data_preprocessed)
    
    return prediction
