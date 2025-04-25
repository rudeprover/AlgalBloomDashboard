import streamlit as st
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

st.set_page_config(page_title="Cyanobacteria Dashboard", layout="wide")
st.title("🌊 Cyanobacteria Density Prediction Dashboard")

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
    },
    marker_options={'draggable': True}
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
        st.success(f"Selected point: Latitude: {lat:.4f}, Longitude: {lon:.4f}")

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

    # Load LightGBM model
    st.markdown("---")
    st.subheader("🤖 Predict Cyanobacteria Density")
    model_file = st.file_uploader("Upload trained LightGBM model (.txt)", type='txt')

    if model_file and 'lat' in df.columns and 'lon' in df.columns:
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

                # Time series plot for selected UID
                st.write("### 📈 Time Series (Pick a UID)")
                selected_uid = st.selectbox("Choose sample ID", df['uid'].unique())
                subset = df[df['uid'] == selected_uid].sort_values('date')

                fig, ax = plt.subplots()
                ax.plot(subset['date'], subset['predicted_abun'], marker='o')
                ax.set_title(f"Predicted Abundance for {selected_uid}")
                ax.set_ylabel("Abundance (cells/ml)")
                ax.set_xlabel("Date")
                st.pyplot(fig)

                # Download predictions
                st.download_button(
                    "📥 Download Predictions CSV",
                    df.to_csv(index=False),
                    file_name="cyanobacteria_predictions.csv",
                    mime='text/csv'
                )
        except Exception as e:
            st.error(f"Failed to load or use model: {e}")
else:
    st.info("Please upload a CSV file to begin.")
