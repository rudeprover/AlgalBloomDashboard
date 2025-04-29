import streamlit as st
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import numpy as np
import requests
from io import BytesIO
import base64

st.set_page_config(page_title="Cyanobacteria Dashboard", layout="wide")
st.title("🌊 Cyanobacteria Density Prediction Dashboard")

# Helper function to get location information from coordinates
def get_location_info(lat, lon):
    """
    Get location information using reverse geocoding.
    Returns a dictionary with available location information.
    """
    try:
        # Using OpenStreetMap Nominatim API for reverse geocoding
        # Note: For production, should use a more robust service with proper API key
        response = requests.get(
            f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10",
            headers={"User-Agent": "CyanobacteriaDashboard/1.0"}
        )
        
        if response.status_code == 200:
            data = response.json()
            info = {
                "display_name": data.get("display_name", "Unknown location"),
                "country": data.get("address", {}).get("country", "Unknown"),
                "state": data.get("address", {}).get("state", 
                         data.get("address", {}).get("county", "Unknown")),
                "water_body": "Unknown" # OSM doesn't reliably provide water body names
            }
            
            # Check if location is near water (simplified)
            address = data.get("address", {})
            for key in address:
                if any(water_term in key for water_term in 
                      ["sea", "ocean", "lake", "river", "water", "bay"]):
                    info["water_body"] = address[key]
                    break
            
            return info
        else:
            return {
                "display_name": "Location lookup failed",
                "country": "Unknown",
                "state": "Unknown",
                "water_body": "Unknown"
            }
    except Exception as e:
        st.error(f"Error getting location data: {e}")
        return {
            "display_name": "Error retrieving location data",
            "country": "Unknown",
            "state": "Unknown",
            "water_body": "Unknown"
        }

st.markdown("""
Upload a CSV containing sample points with latitude, longitude, and date.
This app will extract band values (manually or pre-loaded), and use a trained LightGBM model to predict cyanobacteria abundance.
""")

# Section 1: Map-based point selector
st.subheader("🗺️ Choose a Location on the Map")
map_center = [28.61, 77.21]  # Default location (Delhi)

# Create a Folium map with drawing controls
m = folium.Map(location=map_center, zoom_start=4)
draw = Draw(
    export=False,
    draw_options={
        'polyline': False,
        'polygon': False,
        'rectangle': False,
        'circle': False,
        'circlemarker': False,
        'marker': True
    }
)
draw.add_to(m)

# Render map in Streamlit and capture drawing
output = st_folium(m, width=700, height=500)

selected_point = None
if output and output.get('last_drawn'):
    geom = output['last_drawn']['geometry']
    if geom['type'] == 'Point':
        lon, lat = geom['coordinates']
        selected_point = (lat, lon)
        
        # Display geographic information
        st.success(f"Selected point: Latitude: {lat:.4f}, Longitude: {lon:.4f}")
        
        # Get location information
        with st.spinner("Fetching location information..."):
            location_info = get_location_info(lat, lon)
        
        # Create expander for more detailed geographic info
        with st.expander("Geographic Information", expanded=True):
            st.markdown(f"### 📍 {location_info['display_name']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Coordinates")
                st.markdown(f"**Decimal Degrees:** {lat:.6f}°, {lon:.6f}°")
                
                # Convert to degrees, minutes, seconds
                lat_deg = int(abs(lat))
                lat_min = int((abs(lat) - lat_deg) * 60)
                lat_sec = ((abs(lat) - lat_deg) * 60 - lat_min) * 60
                lat_dir = "N" if lat >= 0 else "S"
                
                lon_deg = int(abs(lon))
                lon_min = int((abs(lon) - lon_deg) * 60)
                lon_sec = ((abs(lon) - lon_deg) * 60 - lon_min) * 60
                lon_dir = "E" if lon >= 0 else "W"
                
                st.markdown(f"**DMS:** {lat_deg}° {lat_min}' {lat_sec:.2f}\" {lat_dir}, {lon_deg}° {lon_min}' {lon_sec:.2f}\" {lon_dir}")
            
            with col2:
                st.markdown("#### Region Information")
                st.markdown(f"* **Country:** {location_info['country']}")
                st.markdown(f"* **State/Region:** {location_info['state']}")
                st.markdown(f"* **Nearest Water Body:** {location_info['water_body']}")
                
                # Calculate rough estimate of water body type based on location
                if -60 <= lat <= 60:  # Tropical/temperate regions
                    if abs(lon) > 150 or abs(lon) < 30:  # Rough estimate for Pacific and Atlantic
                        water_type = "Ocean (estimated)"
                    else:
                        water_type = "Inland/Coastal (estimated)"
                else:  # Polar regions
                    water_type = "Polar waters (estimated)"
                
                st.markdown(f"* **Water Type:** {water_type}")
                
            # Try to get a satellite preview image using Google Static Maps
            st.markdown("#### Satellite Preview")
            map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=13&size=600x300&maptype=satellite&key=YOUR_API_KEY"
            st.markdown("*Satellite imagery preview would appear here with a valid API key*")
            
            # Create a small embedded map centered on the point
            preview_map = folium.Map(location=[lat, lon], zoom_start=12, width=600, height=300)
            folium.Marker([lat, lon], popup="Selected Point").add_to(preview_map)
            st_folium(preview_map, width=600, height=300, key="preview_map")

# Upload CSV
df = None
uploaded_file = st.file_uploader("📂 Upload your sample CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("### 🔍 Sample Data Preview")
    st.dataframe(df.head())

    # Parse date column if numeric YYYYMMDD
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')

    st.write("### 📆 Date Range in Data:")
    st.write(f"From {df['date'].min().date()} to {df['date'].max().date()}")

# Use selected point from map for prediction
if 'use_for_prediction' not in st.session_state:
    st.session_state.use_for_prediction = False

if st.session_state.selected_point and st.session_state.use_for_prediction:
    lat, lon = st.session_state.selected_point
    location_info = st.session_state.location_info
    current_date = datetime.now().strftime("%Y%m%d")
    
    # Create a dataframe for the selected point
    point_df = pd.DataFrame({
        'uid': ['map_selection'],
        'lat': [lat],
        'lon': [lon],
        'date': [pd.to_datetime(current_date, format='%Y%m%d')]
    })
    
    # In a real application, you would fetch these values from satellite data
    # For now, we'll use placeholder values or attempt to estimate based on location
    
    # Simple example of location-based band value estimation (just for demonstration)
    # In a real app, you would use satellite imagery APIs or databases
    water_intensity = 0.15  # Base value
    
    # Adjust based on latitude (simplified example)
    if abs(lat) < 30:  # Tropical regions often have different reflectance
        water_intensity += 0.05
    
    # Adjust based on detected water body type (simplified example)
    if location_info and "ocean" in location_info['water_body'].lower():
        water_intensity -= 0.02  # Oceans tend to have different spectral properties
    elif location_info and "lake" in location_info['water_body'].lower():
        water_intensity += 0.03  # Lakes may have different properties
    
    # Create synthetic band values - in reality these come from remote sensing
    point_df['B2'] = water_intensity + 0.02  # Blue band
    point_df['B3'] = water_intensity         # Green band
    point_df['B4'] = water_intensity - 0.02  # Red band
    point_df['B8'] = water_intensity * 2     # NIR band
    
    # Add geographic metadata
    if location_info:
        point_df['location_name'] = location_info['display_name']
        point_df['country'] = location_info['country']
        point_df['state_region'] = location_info['state']
        point_df['water_body'] = location_info['water_body']
    
    # Calculate water type based on location
    if -60 <= lat <= 60:  # Tropical/temperate regions
        if abs(lon) > 150 or abs(lon) < 30:  # Rough estimate for Pacific and Atlantic
            point_df['water_type'] = "Ocean (estimated)"
        else:
            point_df['water_type'] = "Inland/Coastal (estimated)"
    else:  # Polar regions
        point_df['water_type'] = "Polar waters (estimated)"
    
    # Use this instead of the uploaded CSV
    df = point_df
    
    # Show a horizontal rule to separate sections
    st.markdown("---")
    
    # Show the dataframe with detailed information
    st.write("### 📍 Location Information for Prediction")
    
    if location_info:
        st.markdown(f"**Selected Location:** {location_info['display_name']}")
    
    # Display information in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Geographic Details")
        st.markdown(f"* **Coordinates:** {lat:.4f}°, {lon:.4f}°")
        if location_info:
            st.markdown(f"* **Country:** {location_info['country']}")
            st.markdown(f"* **Region:** {location_info['state']}")
            st.markdown(f"* **Water Body:** {location_info['water_body']}")
        st.markdown(f"* **Water Type:** {point_df['water_type'].iloc[0]}")
    
    with col2:
        st.markdown("#### Satellite Data (Synthetic)")
        st.markdown("*In a production environment, these values would be fetched from actual satellite data:*")
        
        spectral_data = pd.DataFrame({
            'Band': ['B2 (Blue)', 'B3 (Green)', 'B4 (Red)', 'B8 (NIR)'],
            'Value': [point_df['B2'].iloc[0], point_df['B3'].iloc[0], 
                      point_df['B4'].iloc[0], point_df['B8'].iloc[0]]
        })
        
        st.dataframe(spectral_data, hide_index=True)
        
        # Show small bar chart of band values
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.bar(spectral_data['Band'], spectral_data['Value'], color=['blue', 'green', 'red', 'purple'])
        ax.set_ylabel('Reflectance')
        ax.set_title('Spectral Band Values')
        st.pyplot(fig)

# Load LightGBM model
if df is not None:
    st.markdown("---")
    st.subheader("🤖 Predict Cyanobacteria Density")
    model_file = st.file_uploader("Upload trained LightGBM model (.txt)", type='txt')

    if model_file:
        # Load model safely
        try:
            model = lgb.Booster(model_file=model_file)
            st.success("Model loaded!")

            # Check required features in DataFrame
            required_features = ['lat', 'lon', 'B2', 'B3', 'B4', 'B8']
            missing = [f for f in required_features if f not in df.columns]
            
            if missing:
                st.warning(f"Missing required features: {', '.join(missing)}")
            else:
                X = df[required_features]
                df['predicted_abun'] = model.predict(X)

                st.write("### 📊 Predictions")
                st.dataframe(df[['uid', 'lat', 'lon', 'date', 'predicted_abun']])

                # Time series plot for selected UID if multiple entries exist
                if len(df['uid'].unique()) > 1:
                    st.write("### 📈 Time Series (Pick a UID)")
                    selected_uid = st.selectbox("Choose sample ID", df['uid'].unique())
                    subset = df[df['uid'] == selected_uid].sort_values('date')

                    fig, ax = plt.subplots()
                    ax.plot(subset['date'], subset['predicted_abun'], marker='o')
                    ax.set_title(f"Predicted Abundance for {selected_uid}")
                    ax.set_ylabel("Abundance (cells/ml)")
                    ax.set_xlabel("Date")
                    st.pyplot(fig)
                else:
                    # For single point from map selection, show a simple bar chart
                    if 'predicted_abun' in df.columns:
                        fig, ax = plt.subplots()
                        ax.bar(["Selected Location"], df['predicted_abun'])
                        ax.set_title("Predicted Cyanobacteria Abundance")
                        ax.set_ylabel("Abundance (cells/ml)")
                        st.pyplot(fig)

                # Download predictions
                st.download_button(
                    "📥 Download Predictions CSV",
                    df.to_csv(index=False),
                    file_name="cyanobacteria_predictions.csv",
                    mime='text/csv'
                )
        except Exception as e:
            st.error(f"Failed to load or use model: {str(e)}")
else:
    st.info("Please upload a CSV file or select a point on the map to begin.")
