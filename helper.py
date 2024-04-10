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
    fig, ax = plt.subplots(figsize=(20, 20))

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

    if len(safety_car_probabilities) == 0:
        return None

    # Perform Monte Carlo simulation to predict the probability of safety car over each lap
    simulation_columns = []

    for _ in range(num_simulations):
        # Generate random probabilities for safety car occurrence based on calculated probabilities
        probabilities = np.random.rand(len(all_lap_numbers))
        
        # Interpolate probabilities to match all lap numbers
        interpolated_probabilities = np.interp(all_lap_numbers, safety_car_probabilities['LapNumber'], safety_car_probabilities['Probability_Safety_Car'])
        
        safety_car_occurrence = probabilities <= interpolated_probabilities
        simulation_columns.append(pd.Series(safety_car_occurrence.astype(int), name=f'Simulation_{_+1}'))

    # Concatenate all simulation columns into the DataFrame
    simulation_df = pd.concat([simulation_df] + simulation_columns, axis=1)

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
    if simulation_df is None:
        fig = go.Figure()
        fig.update_layout(
            annotations=[
                dict(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text="No occurrences of a safety car - cannot make predictions",
                    showarrow=False,
                    font=dict(size=16)
                )
            ]
        )
        return fig

    fig = go.Figure()

    # Calculate the average probability across all simulations
    avg_probability = simulation_df.drop(columns=['LapNumber', 'Race']).mean(axis=1)

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

def plot_monte_carlo_evaluation(monte_carlo_df_2018, races_2019, race_name):
    # Check if the selected race is in the 2019 data
    if race_name not in races_2019['EventName'].values:
        fig = go.Figure()
        fig.update_layout(
            annotations=[
                dict(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text="Race data not available in 2019",
                    showarrow=False,
                    font=dict(size=16)
                )
            ]
        )
        return fig

    fig = go.Figure()

    # Add traces for 2018 predicted probabilities
    simulation_columns_2018 = [col for col in monte_carlo_df_2018.columns if col.startswith('Simulation_')]
    avg_probability_2018 = monte_carlo_df_2018[simulation_columns_2018].mean(axis=1)
    fig.add_trace(go.Scatter(x=monte_carlo_df_2018['LapNumber'], y=avg_probability_2018,
                             mode='lines', name='Predicted Probability (2018)', line=dict(color='blue')))

    # Add traces for 2019 actual track status
    # Filter races_2019 for the selected race
    selected_race_data_2019 = races_2019[races_2019['EventName'] == race_name]

    # Filter the selected race data for safety car laps with TrackStatus 4 or 6
    safety_car_laps_2019 = selected_race_data_2019[selected_race_data_2019['TrackStatus'].isin([4, 6])]

    fig.add_trace(go.Scatter(x=safety_car_laps_2019['LapNumber'], y=[0.5] * len(safety_car_laps_2019),
                             mode='markers', name='Actual or Virtual Safety Car (2019)', marker=dict(color='green', symbol='triangle-down')))

    # Update layout
    fig.update_layout(
        height=600,
        width=900,
        xaxis_title='Lap Number',
        yaxis_title='Probability of Safety Car',
        yaxis=dict(range=[0, 1]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig
