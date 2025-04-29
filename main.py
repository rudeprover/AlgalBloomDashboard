import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

# Set page config
st.set_page_config(page_title="Coordinate Finder", layout="wide")
st.title("📍 Coordinate Finder")
st.markdown("Drop a marker on the map to see the exact coordinates.")

# Create a Folium map
map_center = [20, 0]  # Default center (roughly middle of the world)
m = folium.Map(location=map_center, zoom_start=2)

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
    }
)
draw.add_to(m)

# Display the map
st.subheader("🗺️ Click anywhere on the map to drop a marker")
map_data = st_folium(m, width=800, height=500)

# Check if a point has been selected on the map
if map_data and map_data.get('last_drawn'):
    geom = map_data['last_drawn']['geometry']
    if geom['type'] == 'Point':
        lon, lat = geom['coordinates']
        
        # Show the coordinates in a box
        st.success(f"Marker dropped at coordinates: Latitude: {lat:.6f}°, Longitude: {lon:.6f}°")
        
        # Display coordinate information in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Decimal Degrees")
            st.markdown(f"**Latitude:** {lat:.6f}°")
            st.markdown(f"**Longitude:** {lon:.6f}°")
            
        with col2:
            # Convert to degrees, minutes, seconds
            lat_deg = int(abs(lat))
            lat_min = int((abs(lat) - lat_deg) * 60)
            lat_sec = ((abs(lat) - lat_deg) * 60 - lat_min) * 60
            lat_dir = "N" if lat >= 0 else "S"
            
            lon_deg = int(abs(lon))
            lon_min = int((abs(lon) - lon_deg) * 60)
            lon_sec = ((abs(lon) - lon_deg) * 60 - lon_min) * 60
            lon_dir = "E" if lon >= 0 else "W"
            
            st.markdown("### Degrees, Minutes, Seconds")
            st.markdown(f"**Latitude:** {lat_deg}° {lat_min}' {lat_sec:.2f}\" {lat_dir}")
            st.markdown(f"**Longitude:** {lon_deg}° {lon_min}' {lon_sec:.2f}\" {lon_dir}")
else:
    st.info("👆 Click on the map to drop a marker and view its coordinates.")
