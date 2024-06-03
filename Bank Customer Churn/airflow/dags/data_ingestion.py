from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import timedelta
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from utils import *
import datetime
import random
import glob
import os
import pandas as pd
import shutil
import logging

"""import sys
sys.path.append('/opt/api')
from models import Base, ProblemStats"""


@dag(
    dag_id='data_ingestion',
    description='Take files and validate the quality',
    tags=['dsp', 'validate', 'ingestion'],
    schedule=timedelta(minutes=1),
    start_date=days_ago(n=0, hour=1),
    catchup=False
)
def data_ingestion():
    default_folder = "/opt/data/raw_data"
    good_folder = "/opt/data/good_data"
    failed_folder = "/opt/data/bad_data"

    @task
    def read_file():
        file_pattern = os.path.join(default_folder, "*.csv")
        file_paths = glob.glob(file_pattern)
        logging.info(f'{file_paths}')
        file_paths = [f for f in file_paths if
                      not os.path.basename(f).startswith('processed_')]

        file_path = random.choice(file_paths)
        logging.info(f'Chosen file: {file_path}')

        # Define the new name for the processed file
        processed_file_name = "processed_" + os.path.basename(file_path)
        processed_file_path = os.path.join(default_folder,
                                           processed_file_name)

        os.rename(file_path, processed_file_path)

        return processed_file_path

    @task
    def validate_data(file):
        validation_outputs = gx_validation(file)
        return validation_outputs

    """@task
    def raise_alert(validator_output):
        validator_result = validator_output["validator_result"]
        sender = user_email
        recipient = recipient_email
        subject = "Data Quality Issues"

        failed_tests = [result for result in validator_result["results"] if not result["success"]]
        
        if failed_tests:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"Dear Engineering team,\n\n"
            message += f"Data Quality Alert - {timestamp}\n\n"
            message += "The following data quality checks have failed:\n\n"

            for test in failed_tests:
                message += f"- Expectation: {test['expectation_config']['expectation_type']}\n"
                message += f"- Column: {test['expectation_config']['kwargs']['column']}\n"
                
                # Check which key is available and use it
                unexpected_list = test['result'].get('partial_unexpected_list') or test['result'].get('unexpected_list', [])
                message += f"- Details: {unexpected_list}\n"
                message += f"- Number of rows: {len(unexpected_list)}\n\n"

            message += ("Please review the validation results and address the issues as soon as possible.\n")
            message += "\nBest Regards,\n[Name]\nML engineer\n[Company]"
            send_email(sender, recipient, subject, message)
            logging.info('Email sent!')
        else:
            logging.info('No data quality issues detected.')"""


    @task
    def split_file(validator_output, folder_b, folder_c):
        file = validator_output["file"]
        validator_result = validator_output["validator_result"]
        df = pd.read_csv(file)
        problem_rows = []

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        for result in validator_result["results"]:
            if not result["success"]:
                problem_rows.extend(
                    result["result"]["partial_unexpected_index_list"])

        if not problem_rows:
            shutil.move(file, folder_c)
        else:
            df_problems = df.loc[problem_rows]
            df_no_problems = df.drop(problem_rows)

            problems_file_path = (
                os.path.join(folder_b,
                             f"file_with_problems_"
                             f"{timestamp}_{os.path.basename(file)}"))
            no_problems_file_path = (
                os.path.join(folder_c,
                             f"file_without_problems_"
                             f"{os.path.basename(file)}"))

            df_problems.to_csv(problems_file_path, index=False)
            df_no_problems.to_csv(no_problems_file_path, index=False)

    """@task
    def save_quality_issues(validator_output, db_url):
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        validator_result = validator_output["validator_result"]
        file_name = (
            os.path.basename(validator_result
                             ["meta"]["batch_spec"]
                             ["reader_options"]["filepath_or_buffer"]))
        for result in validator_result["results"]:
            if not result["success"]:
                column = result["expectation_config"]["kwargs"]["column"]
                expectation_type = result["expectation_config"][
                    "expectation_type"]
                unexpected_values = (
                    str(result["result"]["partial_unexpected_list"]))

                stat = ProblemStats(
                    file_name=file_name,
                    column=column,
                    expectation_type=expectation_type,
                    unexpected_values=unexpected_values
                )
                session.add(stat)
            session.commit()"""

    # Task
    chosen_file = read_file()
    validate = validate_data(chosen_file)
    #raise_alert(validate)
    split_file(validate, failed_folder, good_folder)
    #save_quality_issues(validate, DB_URL)


ingestion_dag = data_ingestion()
