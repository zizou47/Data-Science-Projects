import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from archive.utils import convert_time_format, get_predictions
from archive.utils import GET_API_URL


st.set_page_config(
    page_title="ML Legends",
    page_icon="🤖",
    layout="wide"
)

st.title("🕰️  Predictions History")
st.sidebar.info("📄 In this page, we will show all the filtered prediction history.")
st.sidebar.write("© A product of ML Legends")


# Calculate the default start date & end time
default_start_date = datetime.now() - timedelta(days=3)
default_end_time = datetime.now() + timedelta(minutes=2)

col1, col2 = st.columns(2)

# Left column for start_date and start_time
with col1:
    start_date = st.date_input("Start Date", default_start_date)
    start_time = st.time_input("Start Time")

# Right column for end_date and end_time
with col2:
    end_date = st.date_input("End Date", pd.Timestamp.now().date())
    end_time = st.time_input("End Time", default_end_time)

# 2 columns for filter Ratings & Type
col3, col4 = st.columns(2)

with col3:
    unique_ratings = [1, 2, 3, 4, 5]
    selected_ratings = st.multiselect('Select Ratings to Filter', unique_ratings, default=unique_ratings)

with col4:
    unique_types = ['App', 'Job']
    selected_types = st.multiselect('Select Prediction Type to Filter', unique_types, default=unique_types)


if st.button("🖨️ Show History"):

    response = get_predictions(GET_API_URL, start_date, end_date, start_time, end_time, selected_ratings, selected_types)

    columns_list = ["ID", "Review", "Rating Prediction", "Predict Time", "Predict Type"]

    df = pd.DataFrame(response, columns=columns_list)
    df = df.set_index(df.columns[0])

    df["Predict Time"] = df["Predict Time"].apply(convert_time_format)
    df["Predict Time"] = pd.to_datetime(df["Predict Time"])

    st.dataframe(df, use_container_width=True)
