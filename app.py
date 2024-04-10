import streamlit as st
import pandas as pd
import plotly.express as px
from helper import data_cleaning, correlation_matrix, create_parallel_coordinates_plot, calculate_safety_car_probability, monte_carlo_simulation, plot_monte_carlo_simulation, plot_monte_carlo_evaluation

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

uploaded_file_1 = st.sidebar.file_uploader("Upload 2018 Season Data for Training", type=file_format_type)
uploaded_file_2 = st.sidebar.file_uploader("Upload 2019 Season Data for Testing", type=file_format_type)

selected_race_2018 = None  # Initialize selected_race_2018 at the beginning

if uploaded_file_1 is not None and uploaded_file_2 is not None:
    races_2018 = pd.read_csv(uploaded_file_1)
    races_2019 = pd.read_csv(uploaded_file_2)

    # Data processing for 2018 data
    races_2018 = data_cleaning(races_2018)

    # Data processing for 2019 data
    races_2019 = data_cleaning(races_2019)

    # Display the 2018 dataframe
    st.subheader("2018 Season Data")
    st.dataframe(races_2018)

    # Display the 2019 dataframe
    st.subheader("2019 Season Data")
    st.dataframe(races_2019)

    # Create correlation matrix for 2018 data
    corr_matrix_2018 = correlation_matrix(races_2018)

    # Display correlation matrix results for 2018 data
    st.subheader("Correlation Matrix Results")
    st.pyplot(corr_matrix_2018)

    selected_columns = ['TrackStatus', 'SpeedST', 'LapTimeInSeconds', 'Sector2TimeInSeconds', 'SpeedI1', 'Sector1TimeInSeconds', 'SpeedI2', 'Sector3TimeInSeconds', 'SpeedFL']

    # Create parallel coordinates plot for 2018 data
    fig_PCP_2018 = create_parallel_coordinates_plot(races_2018, selected_columns)

    st.subheader("Interactive Parallel Coordinate Plot (Highly Correlated Variables)")
    st.plotly_chart(fig_PCP_2018)

    # Get unique race names for selection from 2018 data
    race_names_2018 = races_2018['EventName'].unique()

    # Select race to plot from 2018 data
    selected_race_2018 = st.sidebar.selectbox("Select Race (2018 Season)", race_names_2018)

    # Update selected race in session state
    st.session_state.selected_race_2018 = selected_race_2018

    # Filter data for selected race from 2018 data
    selected_race_data_2018 = races_2018[races_2018['EventName'] == selected_race_2018]

    # Calculate safety car probability for 2018 data
    safety_car_data_2018 = calculate_safety_car_probability(selected_race_data_2018)

    # Plot the predicted probability of safety car for each lap for 2018 data
    st.subheader(f"Predicted Probability of Safety Car for {selected_race_2018} (2018 Season)")
    monte_carlo_df_2018 = monte_carlo_simulation(selected_race_data_2018, selected_race_2018)
    fig_monte_carlo_2018 = plot_monte_carlo_simulation(monte_carlo_df_2018, selected_race_2018)

    # Display the plot for 2018 data
    st.plotly_chart(fig_monte_carlo_2018)

# Button to trigger evaluation
if monte_carlo_df_2018 is None:
    st.warning("No occurrences of a safety car in the selected 2018 race. Evaluation is not possible.")
else:
    if st.button('Evaluate Simulation Accuracy with 2019 Data'):
        if selected_race_2018 is None:
            st.error("Please select a race first.")
        else:
            # Get unique race names for selection from 2019 data
            race_names_2019 = races_2019['EventName'].unique()

            # Filter data for selected race from 2019 data
            selected_race_data_2019 = races_2019[races_2019['EventName'] == selected_race_2018]

            # Plot the predicted probability of safety car for each lap for 2019 data
            st.subheader(f"Predicted vs Actual Safety Car Probability for {selected_race_2018}")
            fig_evaluation = plot_monte_carlo_evaluation(monte_carlo_df_2018, races_2019, selected_race_2018)

            if fig_evaluation is not None:
                # Display the evaluation plot
                st.plotly_chart(fig_evaluation)
