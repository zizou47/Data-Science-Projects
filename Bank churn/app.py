from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

# Import the preprocessing functions from build_features.py
from src.features.build_features import preprocess_numeric_data, preprocess_categorical_data, combine_processed_data

# Initialize the FastAPI app
app = FastAPI()

# Load the preprocessing tools and the model
scaler = joblib.load("src/models/save_scaler.joblib")
encoder = joblib.load("src/models/save_encoder.joblib")
poly = joblib.load("src/models/polynomial_features.joblib")
model = joblib.load("src/models/model_logistic.joblib")

# Define the request body schema
class DataItem(BaseModel):
    customer_id: int = Field(None, example=1)  # Optional field
    credit_score: int
    country: str
    gender: str
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int
    active_member: int
    estimated_salary: float

class PredictionRequest(BaseModel):
    data: list[DataItem]

# Define the response schema
class PredictionResponse(BaseModel):
    predictions: list[int]

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame([item.dict() for item in request.data])

        # Drop customer_id if it's included
        if 'customer_id' in input_data.columns:
            input_data = input_data.drop(columns=['customer_id'])

        # Log the columns of input data
        print("Input data columns:", input_data.columns.tolist())

        # Separate numeric and categorical columns
        numerics = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
        categorical_cols = ['country', 'gender', 'credit_card', 'active_member']

        # Ensure all columns are present and in the correct order
        input_data = input_data[numerics + categorical_cols]

        # Log the columns after reordering
        print("Reordered input data columns:", input_data.columns.tolist())

        # Preprocess the numeric and categorical data
        processed_num_x = scaler.transform(input_data[numerics])
        processed_cat_x = encoder.transform(input_data[categorical_cols])

        # Combine the processed numeric and categorical data
        xtest_scl = np.hstack([processed_num_x, processed_cat_x.todense()])

        # Apply polynomial transformation
        poly_test = poly.transform(np.asarray(xtest_scl))

        # Log the shape of the polynomial features
        print("Shape of polynomial features:", poly_test.shape)

        # Make predictions using the loaded model
        predictions = model.predict(np.asarray(poly_test))

        return PredictionResponse(predictions=predictions.tolist())

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
