import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import requests

# Set page config
st.set_page_config(page_title="Location Explorer", layout="wide")
st.title("📍 Location Explorer")
st.markdown("Select a point on the map to see details about that location.")

# Function to get location information
def get_location_info(lat, lon):
    """Get location information using OpenStreetMap Nominatim API"""
    try:
        response = requests.get(
            f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10",
            headers={"User-Agent": "LocationExplorer/1.0"}
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "display_name": data.get("display_name", "Unknown location"),
                "country": data.get("address", {}).get("country", "Unknown"),
                "state": data.get("address", {}).get("state", 
                         data.get("address", {}).get("county", "Unknown")),
                "water_body": next((data.get("address", {}).get(key) for key in data.get("address", {}) 
                                if any(water_term in key.lower() for water_term in 
                                ["sea", "ocean", "lake", "river", "water", "bay"])), "Unknown")
            }
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
st.subheader("🗺️ Click anywhere on the map to select a location")
map_data = st_folium(m, width=800, height=500)

# Check if a point has been selected on the map
if map_data and map_data.get('last_drawn'):
    geom = map_data['last_drawn']['geometry']
    if geom['type'] == 'Point':
        lon, lat = geom['coordinates']
        
        # Show the coordinates
        st.success(f"Selected coordinates: {lat:.6f}°, {lon:.6f}°")
        
        # Get location information
        with st.spinner("Fetching location details..."):
            location_info = get_location_info(lat, lon)
        
        # Display location information
        st.header("📍 Location Details")
        
        # Location name
        st.subheader(location_info['display_name'])
        
        # Display information in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Coordinates")
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
            
            # UTM coordinate estimate
            utm_zone = int((lon + 180) / 6) + 1
            hemisphere = "Northern" if lat >= 0 else "Southern"
            st.markdown(f"**Estimated UTM Zone:** {utm_zone} ({hemisphere} Hemisphere)")
        
        with col2:
            st.markdown("### Geographic Information")
            st.markdown(f"**Country:** {location_info['country']}")
            st.markdown(f"**State/Region:** {location_info['state']}")
            
            # Water body detection
            if location_info['water_body'] != "Unknown":
                st.markdown(f"**Nearest Water Body:** {location_info['water_body']}")
            
            # Estimated timezone
            timezone_offset = int(lon / 15)
            timezone_str = f"UTC{'+' if timezone_offset >= 0 else ''}{timezone_offset}" 
            st.markdown(f"**Estimated Timezone:** {timezone_str} (approximate)")
            
            # Climate zone estimate (very approximate)
            if abs(lat) < 23.5:
                climate = "Tropical"
            elif abs(lat) < 35:
                climate = "Subtropical"
            elif abs(lat) < 66.5:
                climate = "Temperate"
            else:
                climate = "Polar"
            st.markdown(f"**Estimated Climate Zone:** {climate}")
        
        # Show a small focused map of the selected location
        st.markdown("### Location Preview")
        preview_map = folium.Map(location=[lat, lon], zoom_start=10)
        folium.Marker([lat, lon], popup=f"{lat:.4f}, {lon:.4f}").add_to(preview_map)
        st_folium(preview_map, width=800, height=400)
else:
    st.info("👆 Click on the map to select a location and view its details.")
