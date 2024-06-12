from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

# Define the request body schema
class DataItem(BaseModel):
    customer_id: int
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
    customer_id: int
    prediction: int

app = FastAPI()

def load_model():
    scaler = joblib.load("src/models/save_scaler.joblib")
    encoder = joblib.load("src/models/save_encoder.joblib")
    poly = joblib.load("src/models/polynomial_features.joblib")
    model = joblib.load("src/models/model_logistic.joblib")
    return scaler, encoder, poly, model

def preprocess_input_data(input_data, scaler, encoder, numerics, categorical_cols):
    processed_num_x = scaler.transform(input_data[numerics])
    processed_cat_x = encoder.transform(input_data[categorical_cols])
    xtest_scl = np.hstack([processed_num_x, processed_cat_x.todense()])
    return xtest_scl

@app.on_event("startup")
def load():
    global scaler, encoder, poly, model
    scaler, encoder, poly, model = load_model()

@app.post("/predict", response_model=list[PredictionResponse])
def predict(request: PredictionRequest):
    try:
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame([item.dict() for item in request.data])

        # Extract customer_id and preserve it for the response
        customer_ids = input_data['customer_id'].tolist()

        # Drop customer_id for preprocessing
        input_data = input_data.drop(columns=['customer_id'])

        # Define numeric and categorical columns
        numerics = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
        categorical_cols = ['country', 'gender', 'credit_card', 'active_member']

        # Ensure columns are in the correct order
        input_data = input_data[numerics + categorical_cols]

        # Preprocess the input data
        xtest_scl = preprocess_input_data(input_data, scaler, encoder, numerics, categorical_cols)

        # Apply polynomial transformation
        poly_test = poly.transform(np.asarray(xtest_scl))

        # Make predictions using the loaded model
        predictions = model.predict(np.asarray(poly_test))

        # Return the predictions along with customer_id
        response = [{"customer_id": cid, "prediction": pred} for cid, pred in zip(customer_ids, predictions)]
        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
