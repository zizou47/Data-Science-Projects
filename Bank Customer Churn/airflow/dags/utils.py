from email.mime.text import MIMEText
import great_expectations as gx
import smtplib


user_email = "yazidhit@gmail.com"
recipient_email = "yazidforads@gmail.com"
DB_URL = "postgresql://postgres:yazid@host.docker.internal:5432/mydbs"


def send_email(sender, recipient, subject, message):
    # Create the message
    message = MIMEText(message)
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient

    # Establish a connection with the SMTP server
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(user_email, "ulws pdlo avlh oggs")
        server.sendmail(sender, recipient, message.as_string())



def gx_validation(file):
    context = gx.get_context()
    validator = context.sources.pandas_default.read_csv(file)

    # Validate 'diagonal' column
    validator.expect_column_values_to_be_of_type(
        "diagonal", "float64",
        result_format={'result_format': 'SUMMARY'}
    )
    validator.expect_column_values_to_be_between(
        "diagonal", min_value=171.04, max_value=173.01,
        result_format={'result_format': 'SUMMARY'}
    )

    
    # Validate 'height_right' column
    validator.expect_column_values_to_be_of_type(
        "height_right", "float64",
        result_format={'result_format': 'SUMMARY'}
    )
    validator.expect_column_values_to_be_between(
        "height_right", min_value=102.82, max_value=104.95,
        result_format={'result_format': 'SUMMARY'}
    )

    # Validate 'margin_low' column
    validator.expect_column_values_to_be_of_type(
        "margin_low", "float64",
        result_format={'result_format': 'SUMMARY'}
    )
    validator.expect_column_values_to_be_between(
        "margin_low", min_value=2.98, max_value=6.90,
        result_format={'result_format': 'SUMMARY'}
    )

    # Validate 'margin_up' column
    validator.expect_column_values_to_be_of_type(
        "margin_up", "float64",
        result_format={'result_format': 'SUMMARY'}
    )
    validator.expect_column_values_to_be_between(
        "margin_up", min_value=2.27, max_value=3.91,
        result_format={'result_format': 'SUMMARY'}
    )

    # Validate 'length' column
    validator.expect_column_values_to_be_of_type(
        "length", "float64",
        result_format={'result_format': 'SUMMARY'}
    )
    validator.expect_column_values_to_be_between(
        "length", min_value=109.49, max_value=114.44,
        result_format={'result_format': 'SUMMARY'}
    )

    # Validate the data and return the results
    validator_result = validator.validate()
    return {"file": file, "validator_result": validator_result}

