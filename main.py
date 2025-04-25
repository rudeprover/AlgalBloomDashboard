import streamlit as st
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import datetime
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
map_center = [28.61, 77.21]  # Default location (e.g. Delhi)
m = leafmap.Map(center=map_center, zoom=4)
m.add_draw_control()
m.to_streamlit(height=500)

if m.user_roi:
    coords = m.user_roi.centroid().coordinates().getInfo()
    lon, lat = coords
    st.success(f"Selected point: Latitude: {lat:.4f}, Longitude: {lon:.4f}")

# Upload CSV
df = None
uploaded_file = st.file_uploader("📂 Upload your sample CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("### 🔍 Sample Data Preview")
    st.dataframe(df.head())

    if 'date' in df.columns and df['date'].dtype != 'O':
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')

    st.write("### 📆 Date Range in Data:")
    st.write(f"From {df['date'].min().date()} to {df['date'].max().date()}")

    # Load model
    st.markdown("---")
    st.subheader("🤖 Predict Cyanobacteria Density")
    model_file = st.file_uploader("Upload trained LightGBM model (.txt)", type='txt')

    if model_file:
        try:
            model = lgb.Booster(model_file=model_file)
            st.success("Model loaded!")

            required_features = ['lat', 'lon', 'B2', 'B3', 'B4', 'B8']
            if all(col in df.columns for col in required_features):
                X = df[required_features]
                predictions = model.predict(X)
                df['predicted_abun'] = predictions

                st.write("### 📊 Predictions")
                st.dataframe(df[['uid', 'lat', 'lon', 'date', 'predicted_abun']])

                st.write("### 📈 Time Series (Pick a UID)")
                selected_uid = st.selectbox("Choose sample ID", df['uid'].unique())
                subset = df[df['uid'] == selected_uid].sort_values('date')

                fig, ax = plt.subplots()
                ax.plot(subset['date'], subset['predicted_abun'], marker='o')
                ax.set_title(f"Predicted Abundance for {selected_uid}")
                ax.set_ylabel("Abundance (cells/ml)")
                ax.set_xlabel("Date")
                st.pyplot(fig)

                st.download_button("📥 Download Predictions CSV", df.to_csv(index=False), file_name="cyanobacteria_predictions.csv")

            else:
                st.warning(f"Missing required features: {', '.join(set(required_features) - set(df.columns))}")

        except Exception as e:
            st.error(f"Failed to load or use model: {e}")
else:
    st.info("Please upload a CSV file to begin.")
