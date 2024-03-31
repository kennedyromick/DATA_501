from sys import prefix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px


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

    # Original dimensions
    original_dimensions = [
        dict(range=[0, 1], tickvals=[0, 1], label=col) for col in selected_columns[1:]
    ]

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
        ),
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        args=[
                            {
                                "dimensions": original_dimensions  # Reset dimensions to original values
                            }
                        ],
                        label="Reset",
                        method="relayout"
                    )
                ],
                direction="down",
                showactive=True,
                x=0.02,  # Adjust x-coordinate for left position
                xanchor="left",
                y=0.02,  # Adjust y-coordinate for top position
                yanchor="top"
            ),
        ]
    )

    return fig


