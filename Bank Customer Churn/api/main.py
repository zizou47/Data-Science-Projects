from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import joblib

app = FastAPI()

# Define Pydantic model for input data
class InputData(BaseModel):
    diagonal: float
    height_right: float
    margin_low: float
    margin_up: float
    length: float

# Load your machine learning model
def load_model():
    model = joblib.load("../model/model.joblib")
    return model

# Connect to PostgreSQL database
def connect_to_database():
    # Replace 'your_database_name', 'your_username', and 'your_password' with your actual database credentials
    conn = psycopg2.connect(
        dbname="mydsp",
        user="postgres",
        password="yazidov47",
        host="localhost",  # Or your database host address
        port="5432"  # Or your database port
    )
    return conn

# Function to save prediction to PostgreSQL database
def save_prediction_to_database(prediction, used_features):
    conn = connect_to_database()
    cur = conn.cursor()
    sql = """INSERT INTO predictions (prediction, used_features) VALUES (%s, %s);"""
    cur.execute(sql, (prediction, used_features))
    conn.commit()
    cur.close()

# Prediction endpoint
@app.post("/predict/")
def predict_endpoint(data: InputData):
    # Load the model
    model = load_model()
    
    # Make prediction using the loaded model
    prediction = predict(model, data)
    # Save prediction to database
    save_prediction_to_database(prediction, data.dict())
    return {"prediction": prediction}

# Retrieve past predictions endpoint
@app.get("/past_predictions/")
def past_predictions_endpoint():
    conn = connect_to_database()
    cur = conn.cursor()
    sql = """SELECT prediction, used_features FROM predictions;"""
    cur.execute(sql)
    past_predictions = cur.fetchall()
    conn.commit()
    cur.close()
    return {"past_predictions": past_predictions}
