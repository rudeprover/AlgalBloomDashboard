import os
import logging
import streamlit as st
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import io
from PIL import Image

def setup_logger():
    """
    Configure and return a logger for the application
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler('logs/cyanobacteria_app.log')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate that coordinates are within valid ranges
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        bool: True if coordinates are valid, False otherwise
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180

def get_density_color_scale():
    """
    Get a color scale for cyanobacteria density visualization
    
    Returns:
        list: Color scale definitions for plotly
    """
    return [
        [0, 'rgb(0, 51, 153)'],     # Dark blue (very low)
        [0.2, 'rgb(0, 153, 255)'],  # Light blue (low)
        [0.4, 'rgb(0, 204, 102)'],  # Green (moderate)
        [0.6, 'rgb(255, 255, 0)'],  # Yellow (moderate-high)
        [0.8, 'rgb(255, 153, 51)'], # Orange (high)
        [1.0, 'rgb(255, 0, 0)']     # Red (very high)
    ]

def get_density_risk_level(density: float) -> str:
    """
    Get the risk level based on cyanobacteria density
    
    Args:
        density (float): Cyanobacteria density in cells/ml
        
    Returns:
        str: Risk level description
    """
    if density < 10000:
        return "Very Low"
    elif density < 20000:
        return "Low"
    elif density < 100000:
        return "Moderate"
    elif density < 1000000:
        return "High"
    else:
        return "Very High"

def create_time_series_plot(time_series_df: pd.DataFrame) -> go.Figure:
    """
    Create a plotly figure for the time series data
    
    Args:
        time_series_df (pd.DataFrame): Time series data with 'date' and 'density' columns
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Create a copy of the dataframe
    df = time_series_df.copy()
    
    # Add risk level to the dataframe
    df['risk_level'] = df['density'].apply(get_density_risk_level)
    
    # Create the figure
    fig = px.line(
        df, 
        x='date', 
        y='density',
        labels={'date': 'Date', 'density': 'Cyanobacteria Density (cells/ml)'},
        title='Historical Monthly Cyanobacteria Density',
        color_discrete_sequence=['#2D68C4']
    )
    
    # Update the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Density (cells/ml)',
        hovermode='x unified',
        legend_title='',
        template='plotly_white',
        height=500
    )
    
    # Add markers colored by risk level
    risk_colors = {
        'Very Low': 'rgb(0, 51, 153)',
        'Low': 'rgb(0, 153, 255)',
        'Moderate': 'rgb(0, 204, 102)',
        'High': 'rgb(255, 153, 51)',
        'Very High': 'rgb(255, 0, 0)'
    }
    
    # Add markers with risk level colors
    for risk_level, color in risk_colors.items():
        risk_df = df[df['risk_level'] == risk_level]
        if not risk_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=risk_df['date'],
                    y=risk_df['density'],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=risk_level,
                    hovertemplate='%{x}<br>Density: %{y:,.2f} cells/ml<br>Risk: ' + risk_level
                )
            )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    # Format y-axis with commas
    fig.update_yaxes(tickformat=",")
    
    return fig

def create_forecast_plot(forecast_df: pd.DataFrame) -> go.Figure:
    """
    Create a plotly figure for the forecast data
    
    Args:
        forecast_df (pd.DataFrame): Forecast data with 'date' and 'predicted_density' columns
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Create a copy of the dataframe
    df = forecast_df.copy()
    
    # Add risk level to the dataframe
    df['risk_level'] = df['predicted_density'].apply(get_density_risk_level)
    
    # Create the figure
    fig = px.line(
        df, 
        x='date', 
        y='predicted_density',
        labels={'date': 'Date', 'predicted_density': 'Predicted Density (cells/ml)'},
        title='7-Day Cyanobacteria Density Forecast',
        color_discrete_sequence=['#1F77B4']
    )
    
    # Add markers colored by risk level
    risk_colors = {
        'Very Low': 'rgb(0, 51, 153)',
        'Low': 'rgb(0, 153, 255)',
        'Moderate': 'rgb(0, 204, 102)',
        'High': 'rgb(255, 153, 51)',
        'Very High': 'rgb(255, 0, 0)'
    }
    
    # Add markers with risk level colors
    for risk_level, color in risk_colors.items():
        risk_df = df[df['risk_level'] == risk_level]
        if not risk_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=risk_df['date'],
                    y=risk_df['predicted_density'],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=risk_level,
                    hovertemplate='%{x}<br>Predicted Density: %{y:,.2f} cells/ml<br>Risk: ' + risk_level
                )
            )
    
    # Update the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Predicted Density (cells/ml)',
        hovermode='x unified',
        legend_title='',
        template='plotly_white',
        height=400
    )
    
    # Format x-axis to show days of week
    fig.update_xaxes(
        tickformat='%a %b %d'
    )
    
    # Format y-axis with commas
    fig.update_yaxes(tickformat=",")
    
    return fig

def display_map_with_density(lat: float, lon: float, time_series_df: pd.DataFrame) -> None:
    """
    Create and display a folium map with cyanobacteria density visualization
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        time_series_df (pd.DataFrame): Time series data with 'date' and 'density' columns
    """
    # Create map centered at coordinates
    m = folium.Map(location=[lat, lon], zoom_start=10)
    
    # Latest density from the time series
    latest_density = time_series_df.iloc[-1]['density']
    
    # Determine color based on density
    def get_color(density):
        if density < 10000:
            return 'blue'
        elif density < 20000:
            return 'green'
        elif density < 100000:
            return 'yellow'
        elif density < 1000000:
            return 'orange'
        else:
            return 'red'
    
    color = get_color(latest_density)
    
    # Add marker
    folium.CircleMarker(
        location=[lat, lon],
        radius=15,
        color='black',
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        tooltip=f"Density: {latest_density:,.2f} cells/ml<br>Risk: {get_density_risk_level(latest_density)}"
    ).add_to(m)
    
    # Create a heat map effect for the density
    # Generate points in a grid around the center
    radius = 0.05  # Approximately 5km
    grid_size = 10
    heat_data = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate position
            grid_lat = lat + radius * (i - grid_size/2) / (grid_size/2)
            grid_lon = lon + radius * (j - grid_size/2) / (grid_size/2)
            
            # Calculate distance from center
            dist = ((grid_lat - lat)**2 + (grid_lon - lon)**2)**0.5
            
            # Weight based on distance (inverse)
            weight = max(0, 1 - (dist / radius)) * latest_density / 10000
            
            heat_data.append([grid_lat, grid_lon, weight])
    
    # Add heat map layer
    HeatMap(heat_data, radius=20, blur=15, max_zoom=13).add_to(m)
    
    # Render the map
    st_data = st.components.v1.html(m._repr_html_(), height=400)
    
    return st_data
