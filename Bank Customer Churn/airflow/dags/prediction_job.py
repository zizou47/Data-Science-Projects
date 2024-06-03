from datetime import timedelta
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
import pandas as pd
import logging
import requests
import os

API_URL = "http://host.docker.internal:8050/predict/"
folder_path = "/opt/data/good_data"


@dag(
    dag_id='prediction_job',
    description='Take files and output predictions',
    tags=['dsp', 'prediction_job'],
    schedule=timedelta(minutes=2),
    start_date=days_ago(n=0, hour=1),
    catchup=False
)
def prediction_job():
    @task
    def check_for_new_data(path):
        csv_files = [file for file in os.listdir(path) if
                     file.endswith(".csv") and
                     not file.startswith("predicted_")]

        if not csv_files:
            return None

        df_list = []
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            df_list.append(pd.read_csv(file_path))
            processed_file_path = os.path.join(folder_path,
                                               f'predicted_{file}')
            os.rename(file_path, processed_file_path)

        merged_df = pd.concat(df_list, ignore_index=True)
        return merged_df

    @task
    def make_predictions(df):
        prediction_data = []
        for _, row in df.iterrows():
            row_data = {
                "is_genuine": row["is_genuine"],
                "diagonal": row["diagonal"],
                "height_left": row["height_left"],
                "height_right": row["height_right"],
                "margin_low": row["margin_low"],
                "margin_up": row["margin_up"],
                "length": row["length"],
                "PredictionSource": "scheduled"
            }
            prediction_data.append(row_data)

        response = requests.post(
            API_URL,
            json=prediction_data
        )

        response_data = response.json()
        prediction = response_data["prediction"]
        logging.info(f'{prediction}')

    df_to_predict = check_for_new_data(folder_path)
    make_predictions(df_to_predict)


scheduled_job_dag = prediction_job()
