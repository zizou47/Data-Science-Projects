import os
import joblib
import pandas as pd
import numpy as np

def load_model():
    scaler = joblib.load("src/models/save_scaler.joblib")
    encoder = joblib.load("src/models/save_encoder.joblib")
    poly = joblib.load("src/models/polynomial.joblib")
    model = joblib.load("src/models/model_logistic.joblib")
    return scaler, encoder, poly, model

def preprocess_input_data(input_data, scaler, encoder, numerics, categorical_cols):
    processed_num_x = scaler.transform(input_data[numerics])
    processed_cat_x = encoder.transform(input_data[categorical_cols])
    xtest_scl = np.hstack([processed_num_x, processed_cat_x.todense()])
    return xtest_scl

if __name__ == "__main__":
    # Load the model
    scaler, encoder, poly, model = load_model()

    # Load the test data
    test_filepath = r"../../outputs/test_csv.csv"
    test_df = pd.read_csv(test_filepath, sep=',')
    test_df = test_df.drop(columns=test_df.columns[0])

    numerics = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
    categorical_cols = ['country', 'gender', 'credit_card', 'active_member']

    xtest_scl = preprocess_input_data(test_df, scaler, encoder, numerics, categorical_cols)

    # Apply polynomial transformation
    poly_test = poly.transform(np.asarray(xtest_scl))

    # Make predictions using the loaded model
    y_pred = model.predict(np.asarray(poly_test))

    print(y_pred[:10], type(y_pred))
    df = pd.DataFrame({"predictions": y_pred})
    df.to_csv("../../outputs/predict_output.csv")
