from sys import prefix
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go


excel_type =["vnd.ms-excel","vnd.openxmlformats-officedocument.spreadsheetml.sheet", "vnd.oasis.opendocument.spreadsheet", "vnd.oasis.opendocument.text"]


def data(data, file_type, seperator=None):

    if file_type == "csv":
        data = pd.read_csv(data)
    
    elif file_type in excel_type:
        data = pd.read_excel(data)
        st.sidebar.info("If you are using Excel file so there could be chance of getting minor error(temporary sollution: avoid the error by removing overview option from input box) so bear with it. It will be fixed soon")
    
    elif file_type == "plain":
        try:
            data = pd.read_table(data, sep=seperator)
        except ValueError:
            st.info("If you haven't Type the separator then dont worry about the error this error will go as you type the separator value and hit Enter.")

    return data

def seconddata(data, file_type, seperator=None):

    if file_type == "csv":
        data = pd.read_csv(data)

   # elif file_type == "json":
    #    data = pd.read_json(data)
    #    data = (data["devices"].apply(pd.Series))
    
    elif file_type in excel_type:
        data = pd.read_excel(data)
        st.sidebar.info("If you are using Excel file so there could be chance of getting minor error(temporary sollution: avoid the error by removing overview option from input box) so bear with it. It will be fixed soon")
    
    elif file_type == "plain":
        try:
            data = pd.read_table(data, sep=seperator)
        except ValueError:
            st.info("If you haven't Type the separator then dont worry about the error this error will go as you type the separator value and hit Enter.")

    return data

def data_cleaning(df):
    # Drop Deleted column
    df.drop('Deleted', axis=1, inplace=True)
    
    # DNF (y/n)
    df['DNF (y/n)'] = df['DNF (y/n)'].replace({'yes': 1, 'no': 0})
    
    # Compound
    df['Compound'] = df['Compound'].replace({'ULTRASOFT': 0, 'SOFT': 1, 'SUPERSOFT': 2, 'MEDIUM': 3, 'WET': 4, 'INTERMEDIATE': 5, 'HYPERSOFT': 6, 'HARD': 7})
    
    # FreshTyre
    df['FreshTyre'] = df['FreshTyre'].astype(int)
    
    # IsPersonalBest
    df['IsPersonalBest'] = df['IsPersonalBest'].astype(int)
    
    # Rainfall
    df['Rainfall'] = df['Rainfall'].astype(int)
    
    # TeamName
    df['Team'] = df['Team'].replace({'Ferrari': 0, 'Force India': 1, 'Haas F1 Team': 2, 'McLaren': 3, 'Mercedes': 4, 'Racing Point': 5, 'Red Bull Racing': 6, 'Renault': 7, 'Sauber': 8, 'Toro Rosso': 9, 'Williams': 10})
    
    # Convert the time columns to timedelta
    time_columns = ['Time', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime', 'LapStartTime']
    df[time_columns] = df[time_columns].apply(pd.to_timedelta, errors='coerce')
    
    # Extract the total seconds for each timedelta column
    for column in time_columns:
        df[f'{column}InSeconds'] = df[column].dt.total_seconds()
    
    return df


def correlation_matrix(df, title='Correlation Heatmap'):

    # Exclude non-numeric columns from correlation calculation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr()

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)

    return fig


def create_parallel_coordinates_plot(df, selected_columns):
    # Separate categorical and numeric columns
    numeric_columns = selected_columns[1:]

    # Normalize numeric columns
    scaler = MinMaxScaler()
    df_subset = df[selected_columns]
    df_subset[numeric_columns] = scaler.fit_transform(df_subset[numeric_columns])

    # Create parallel coordinates plot
    fig = px.parallel_coordinates(
        df_subset,
        color='TrackStatus',
        color_continuous_scale=['plum', 'aqua'],  # Set custom colors for TrackStatus
        dimensions=selected_columns[1:],  # Exclude TrackStatus from dimensions
        labels={'TrackStatus': 'Status'},
        range_color=[4, 6],  # Set the range for color scale
    )

    # Customize the layout
    fig.update_layout(
        height=600,
        width=900,
        title='',
        margin=dict(l=50),
        coloraxis_colorbar=dict(
            title="Status",
            tickvals=[4, 6],
            ticktext=["4", "6"]
        )
    )

    return fig

def monte_carlo_simulation(df, selected_race, num_simulations=1000):
    # Convert num_simulations to an integer if it's a string
    if isinstance(num_simulations, str):
        num_simulations = int(num_simulations)

    # Get unique lap numbers
    all_lap_numbers = df['LapNumber'].unique()

    # Create a DataFrame to store simulation results
    simulation_df = pd.DataFrame({'LapNumber': all_lap_numbers})

    # Calculate safety car probabilities for each lap
    safety_car_probabilities = calculate_safety_car_probability(df)

    # Perform Monte Carlo simulation to predict the probability of safety car over each lap
    for _ in range(num_simulations):
        # Generate random probabilities for safety car occurrence based on calculated probabilities
        probabilities = np.random.rand(len(all_lap_numbers))
        
        # Interpolate probabilities to match all lap numbers
        interpolated_probabilities = np.interp(all_lap_numbers, safety_car_probabilities['LapNumber'], safety_car_probabilities['Probability_Safety_Car'])
        
        safety_car_occurrence = probabilities <= interpolated_probabilities
        simulation_df[f'Simulation_{_+1}'] = safety_car_occurrence.astype(int)

    # Assign the selected race name to each row
    simulation_df['Race'] = selected_race

    return simulation_df


def calculate_safety_car_probability(df):
    # Filter data to include only rows with track status 4 or 6
    safety_car_data = df[df['TrackStatus'].isin([4, 6])]
    
    # Group by LapNumber and count the occurrences of track status 4 or 6 for each lap
    safety_car_counts = safety_car_data.groupby('LapNumber').size().reset_index(name='SafetyCarCount')
    
    # Calculate the total number of laps
    total_laps = df['LapNumber'].nunique()
    
    # Calculate the probability of safety car for each lap
    safety_car_counts['Probability_Safety_Car'] = safety_car_counts['SafetyCarCount'] / total_laps
    
    return safety_car_counts


def plot_monte_carlo_simulation(simulation_df, selected_race):
    fig = go.Figure()

    # Calculate the average probability across all simulations
    avg_probability = np.mean(simulation_df.filter(like='Simulation_'), axis=1)

    # Add trace for the averaged probability
    fig.add_trace(go.Scatter(x=simulation_df['LapNumber'], y=avg_probability, mode='lines', name='Average Probability'))

    # Update layout
    fig.update_layout(
        xaxis_title='Lap Number',
        yaxis_title='Probability of Safety Car',
        height=600,
        width=900,
        yaxis=dict(range=[0, 1]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig
