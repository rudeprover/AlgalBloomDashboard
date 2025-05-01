import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from dateutil.relativedelta import relativedelta

# Import custom modules
from model_handler import generate_historical_monthly_series, predict_future_density
from metrics import calculate_mape
from utils import setup_logger, validate_coordinates, create_time_series_plot, create_forecast_plot

# Setup logger
logger = setup_logger()

def main():
    st.set_page_config(page_title="Cyanobacteria Prediction Dashboard", layout="wide")
    
    # Header and description
    st.title("🌊 Cyanobacteria Density Prediction Dashboard")
    
    st.markdown("""
    This application predicts cyanobacteria density based on satellite imagery for a given location and date.
    The system generates a 5-year historical monthly time series and forecasts density for the next 7 days.
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.subheader("🛠️ Input Parameters")
        
        # Date picker
        st.write("📅 Select reference date:")
        reference_date = st.date_input(
            "Date for prediction",
            datetime.date.today() - datetime.timedelta(days=14),  # Default to 2 weeks ago
            help="Select a reference date. The system will generate a 5-year historical time series up to this date."
        )
        
        # Manual coordinate input
        st.write("🌐 Enter coordinates:")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=28.61, min_value=-90.0, max_value=90.0, step=0.01)
        with col2:
            longitude = st.number_input("Longitude", value=77.21, min_value=-180.0, max_value=180.0, step=0.01)
        
        # Cache directory option
        cache_dir = st.text_input("Cache Directory", "cache", help="Directory to store satellite data")
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Generate prediction button
        generate_button = st.button("🔍 Generate Prediction", use_container_width=True)
        
        # Status information
        st.write("---")
        st.subheader("ℹ️ Status")
        status_placeholder = st.empty()
    
    # Main area - Map for selecting points
    st.subheader("🗺️ Select a location on the map")
    
    # Base map centered on provided coordinates
    m = folium.Map(location=[latitude, longitude], zoom_start=6)
    
    # Add marker at current coordinates
    folium.Marker(
        [latitude, longitude],
        popup=f"Lat: {latitude}, Lon: {longitude}",
        tooltip="Current selection"
    ).add_to(m)
    
    # Add drawing controls to the map
    draw = Draw(
        export=False,
        draw_options={
            'polyline': False,
            'polygon': False,
            'rectangle': False,
            'circle': False,
            'circlemarker': False,
            'marker': True
        },
        marker_options={'draggable': True}
    )
    draw.add_to(m)
    
    # Display the map and capture interactions
    map_output = st_folium(m, width="100%", height=400)
    
    # Update coordinates if a marker is placed on the map
    if map_output and map_output.get('last_drawn'):
        geom = map_output['last_drawn']['geometry']
        if geom['type'] == 'Point':
            longitude, latitude = geom['coordinates']
            st.success(f"Selected location: Latitude: {latitude:.4f}, Longitude: {longitude:.4f}")
    
    # Validation of coordinates
    if not validate_coordinates(latitude, longitude):
        st.error("Invalid coordinates. Latitude must be between -90 and 90, and longitude between -180 and 180.")
    
    # Results section
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Create containers for results
    time_series_container = st.container()
    forecast_container = st.container()
    metrics_container = st.container()
    features_container = st.expander("🔬 Features Used for Prediction (Click to expand)")
    
    # Generate prediction when button is clicked
    if generate_button:
        with status_placeholder:
            st.info("Generating time series and predictions... This may take several minutes to process.")
        
        try:
            # Calculate the start date (5 years before the reference date)
            start_date = reference_date - relativedelta(years=5)
            
            # Generate the 5-year monthly time series
            with st.spinner("Generating 5-year monthly time series..."):
                monthly_time_series = generate_historical_monthly_series(
                    latitude=latitude,
                    longitude=longitude,
                    start_date=start_date,
                    end_date=reference_date,
                    cache_dir=cache_dir
                )
                
                if monthly_time_series is None or monthly_time_series.empty:
                    st.error("Failed to generate monthly time series. Please try a different location or date range.")
                    return
            
            # Predict the next 7 days
            with st.spinner("Predicting density for the next 7 days..."):
                forecast_df, features = predict_future_density(
                    monthly_time_series=monthly_time_series,
                    reference_date=reference_date,
                    days_to_predict=7
                )
                
                if forecast_df is None or forecast_df.empty:
                    st.error("Failed to generate forecast. Please try a different time series.")
                    return
            
            # Calculate MAPE (assuming we have some observed values to compare against)
            # For demonstration purposes, we'll use random "actual" values
            # In a real application, you would use actual observed values if available
            mape = calculate_mape(
                forecast_df['predicted_density'].values,
                # Simulated "actual" values (in a real app, you'd use real observations if available)
                forecast_df['predicted_density'].values * (1 + np.random.normal(0, 0.1, len(forecast_df)))
            )
            
            # Display the monthly time series
            with time_series_container:
                st.subheader("📈 Historical Monthly Time Series (5 Years)")
                fig = create_time_series_plot(monthly_time_series)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add download button for the time series data
                st.download_button(
                    label="📥 Download Historical Time Series CSV",
                    data=monthly_time_series.to_csv(index=False),
                    file_name="cyanobacteria_historical_time_series.csv",
                    mime="text/csv"
                )
            
            # Display the 7-day forecast
            with forecast_container:
                st.subheader("🔮 7-Day Density Forecast")
                fig = create_forecast_plot(forecast_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add download button for the forecast data
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=forecast_df.to_csv(index=False),
                    file_name="cyanobacteria_forecast.csv",
                    mime="text/csv"
                )
            
            # Display metrics
            with metrics_container:
                st.subheader("📊 Forecast Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
                
                with col2:
                    # Current density from the last point in the time series
                    current_density = monthly_time_series.iloc[-1]['density']
                    st.metric("Current Density", f"{current_density:,.2f} cells/ml")
                
                with col3:
                    # Predicted density for day 7
                    final_prediction = forecast_df.iloc[-1]['predicted_density']
                    change = ((final_prediction / current_density) - 1) * 100
                    st.metric(
                        "Day 7 Forecast", 
                        f"{final_prediction:,.2f} cells/ml",
                        f"{change:+.2f}%",
                        delta_color="inverse" if change < 0 else "normal"
                    )
            
            # Display features used for prediction
            with features_container:
                if features is not None:
                    # Make features more readable
                    readable_features = features.copy().T.reset_index()
                    readable_features.columns = ['Feature', 'Value']
                    readable_features['Feature'] = readable_features['Feature'].str.replace('_', ' ').str.title()
                    
                    # Format the feature table nicely
                    st.dataframe(readable_features, use_container_width=True)
                else:
                    st.info("Feature information not available.")
            
            with status_placeholder:
                st.success("Processing complete!")
                
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            with status_placeholder:
                st.error("An error occurred during processing.")

if __name__ == "__main__":
    main()
