import streamlit as st
import requests
import pandas as pd
import json

BASE_URL = "http://localhost:8000"
PREDICT_URL = f"{BASE_URL}/predict/"
BATCH_PREDICT_URL = f"{BASE_URL}/batch-predict/"
HEADERS = {'Content-type': 'application/json'}
COLUMNS = ['height_right', 'margin_low', 'margin_up', 'length']


def send_prediction_request(features: pd.DataFrame,
                            source_application: str) -> pd.DataFrame:
    features_dict = features.to_dict(orient='records')[0]
    features_dict["source_application"] = source_application

    response = requests.post(PREDICT_URL, json=features_dict, timeout=5000,
                             headers=HEADERS)
    if response.status_code == 200:
        return pd.DataFrame([json.loads(response.content)])
    else:
        st.error(f"Error: {response.status_code} - {response.reason}")
        return pd.DataFrame()


def send_batch_prediction_request(file_or_files, source_application: str):
    # Ensure files are in a list
    files = [file_or_files] if not isinstance(file_or_files,
                                              list) else file_or_files

    # Create a dictionary to hold the files
    upload_files = {'files': [(file.name, file, 'text/csv') for file in files]}

    # Include 'source_application' in the data
    data = {'source_application': source_application}

    # Make the POST request
    response = requests.post(BATCH_PREDICT_URL, files=upload_files, data=data)

    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error(f"Error: {response.status_code} - {response.reason}")
        return pd.DataFrame()


st.title("Bill Authenticity Checker")
st.subheader("Fill in the bill's info, and we will tell you if it's genuine!")


# Slider creation function
def create_slider(label, min_val, max_val, value):
    return st.slider(label, min_val, max_val, value, 0.01)


# Create sliders and get selected values
height_right = create_slider("Height Right", 101.0, 106.0, 103.0)
margin_low = create_slider("Margin Low", 2.0, 8.0, 5.0)
margin_up = create_slider("Margin Up", 1.0, 5.0, 3.0)
length = create_slider("Length", 108.0, 115.0, 111.0)

if st.button('Get a Prediction!'):
    prediction_df = pd.DataFrame(
        [[height_right, margin_low, margin_up, length]], columns=COLUMNS)
    prediction_result = send_prediction_request(prediction_df, "webapp")
    if not prediction_result.empty:
        pred_value = int(prediction_result.iloc[0]['pred_result'])
        pred_score = prediction_result.iloc[0]['pred_result']
        if pred_value == 1:
            st.success(
                f"The bill is likely real. ✅ (Prediction Score: {pred_score})")
        elif pred_value == 0:
            st.warning(
                f"The bill is likely fake. ❌ (Prediction Score: {pred_score})")
        else:
            st.info("Unable to determine the authenticity of the bill.")

# Batch Prediction
st.subheader("Upload a CSV for Multiple Predictions")
feature_csv = st.file_uploader("Upload CSV file",
                               type=["csv"],
                               accept_multiple_files=True)

if feature_csv and st.button("Get multiple predictions"):
    predictions = send_batch_prediction_request(feature_csv, "webapp")
    if not predictions.empty:
        st.write("Prediction Results:")
        st.dataframe(predictions.style.applymap(lambda
                                                    x: 'background-color: lightgreen' if x == 1 else 'background-color: lightcoral' if x == 0 else '',
                                                subset=['pred_result']))