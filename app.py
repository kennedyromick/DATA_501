import streamlit as st
import pandas as pd
import plotly.express as px
from helper import data_cleaning, correlation_matrix, create_parallel_coordinates_plot, calculate_safety_car_probability, monte_carlo_simulation, plot_monte_carlo_simulation, seconddata

st.set_page_config(
    page_title="Beyond the Track App",
    page_icon="🏎",
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

uploaded_file = st.sidebar.file_uploader("Upload Your file", type=file_format_type)

if uploaded_file is not None:
    races = pd.read_csv(uploaded_file)

    # Data processing
    races = data_cleaning(races)

    # Create correlation matrix
    corr_matrix = correlation_matrix(races)

    # Display correlation matrix results
    st.subheader("Correlation Matrix Results")

    # Display the correlation matrix plot
    st.pyplot(corr_matrix)

    # Select columns for parallel coordinate plot
    selected_columns = ['TrackStatus', 'SpeedST', 'LapTimeInSeconds', 'Sector2TimeInSeconds', 'SpeedI1', 'Sector1TimeInSeconds', 'SpeedI2', 'Sector3TimeInSeconds', 'SpeedFL']

    # Create parallel coordinates plot
    fig_PCP = create_parallel_coordinates_plot(races, selected_columns)

    st.subheader("Interactive Parallel Coordinate Plot (Highly Correlated Variables)")
    # Show the plot
    st.plotly_chart(fig_PCP)

    # Get unique race names for selection
    race_names = races['EventName'].unique()

    # Select race to plot
    selected_race = st.sidebar.selectbox("Select Race", race_names)

    # Filter data for selected race
    selected_race_data = races[races['EventName'] == selected_race]

    # Calculate safety car probability
    safety_car_data = calculate_safety_car_probability(selected_race_data)

    # Plot the predicted probability of safety car for each lap
    st.subheader(f"Predicted Probability of Safety Car for {selected_race}")
    monte_carlo_df = monte_carlo_simulation(selected_race_data, selected_race)
    fig_monte_carlo = plot_monte_carlo_simulation(monte_carlo_df, selected_race)

    # Display the plot
    st.plotly_chart(fig_monte_carlo)
