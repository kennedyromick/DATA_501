import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from helper import data, data_cleaning, create_parallel_coordinates_plot

st.set_page_config(
     page_title="Data Analysis Web App",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://github.com/everydaycodings/Data-Analysis-Web-App',
         'Report a bug': "https://github.com/everydaycodings/Data-Analysis-Web-App/issues/new",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
)

st.sidebar.title("Beyond the Track")

file_format_type = ["csv", "txt", "xls", "xlsx", "ods", "odt"]
functions = ["Overview", "Outliers", "Drop Columns", "Drop Categorical Rows", "Drop Numeric Rows", "Rename Columns", "Display Plot", "Handling Missing Data", "Data Wrangling"]
excel_type =["vnd.ms-excel","vnd.openxmlformats-officedocument.spreadsheetml.sheet", "vnd.oasis.opendocument.spreadsheet", "vnd.oasis.opendocument.text"]

uploaded_file = st.sidebar.file_uploader("Upload Your file", type=file_format_type)

if uploaded_file is not None:
    races = pd.read_csv(uploaded_file)

    # Data processing steps using helper functions from helper.py
    races = data_cleaning(races)  # Example function from helper.py

    # Select columns for parallel coordinate plot
    selected_columns = ['TrackStatus', 'SpeedST', 'LapTimeInSeconds', 'Sector2TimeInSeconds', 'SpeedI1', 'Sector1TimeInSeconds', 'SpeedI2', 'Sector3TimeInSeconds', 'SpeedFL']

    # Create parallel coordinates plot
    fig = create_parallel_coordinates_plot(races, selected_columns)

    # Show the plot
    st.plotly_chart(fig)