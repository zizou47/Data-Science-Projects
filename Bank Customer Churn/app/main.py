from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
import joblib
import pandas as pd
import sys
import os

# Add the path to the EDA_Model directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'EDA_Model'))

from model_functions import preprocess_data, get_numerics_col, get_categorical_col, select_features_and_target

# Initialize FastAPI app
app = FastAPI()

# Load the saved model
model_path = os.path.join(os.path.dirname(__file__), '..', 'saved_model', 'RF.joblib')
model = joblib.load(model_path)

# Define request body structure
class ChurnRequest(BaseModel):
    credit_score: int = Field(..., example=600)
    country: str = Field(..., example="France")
    gender: str = Field(..., example="Female")
    age: int = Field(..., example=40)
    tenure: int = Field(..., example=3)
    balance: float = Field(..., example=60000.0)
    products_number: int = Field(..., example=2)
    credit_card: int = Field(..., example=1)
    active_member: int = Field(..., example=1)
    estimated_salary: float = Field(..., example=50000.0)

# Define preprocessing function
def preprocess_input(data):
    # Convert data to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Map categorical variables
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['country'] = df['country'].map({'France': 1, 'Spain': 0, 'Germany': 2})
    
    numeric_columns = ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]
    categorical_columns = ["country", "gender", "credit_card", "active_member"]
    
    # Preprocess the input data
    X_preprocessed, _ = preprocess_data(df, df, numeric_columns, categorical_columns)
    
    return X_preprocessed

@app.post("/predict")
async def predict_churn(request: ChurnRequest):
    try:
        # Preprocess the input data
        data_preprocessed = preprocess_input(request)
        
        # Make prediction
        prediction = model.predict(data_preprocessed)
        prediction_proba = model.predict_proba(data_preprocessed)
        
        # Interpret prediction
        result = 'Churn' if prediction[0] == 1 else 'No Churn'
        proba = prediction_proba[0].tolist()
        
        return {"prediction": result, "probability": proba}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define a root endpoint for basic status check
@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API"}
