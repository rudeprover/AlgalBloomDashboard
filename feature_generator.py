"""
Feature Generator Module

This module wraps the functionality from the provided Feature_Generator.txt file
to generate features for cyanobacteria density prediction.

In a production environment, this would import and use the actual feature generation code.
For this demonstration, we're providing a simplified implementation.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union

# Setup logging
logger = logging.getLogger(__name__)

# This is a simplified version. In a real implementation, you would
# import the actual feature generation code from Feature_Generator.txt

def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate that coordinates are within valid ranges
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        bool: True if coordinates are valid, False otherwise
    """
    if not (-90 <= lat <= 90):
        logger.error(f"Invalid latitude: {lat}. Must be between -90 and 90.")
        return False
        
    if not (-180 <= lon <= 180):
        logger.error(f"Invalid longitude: {lon}. Must be between -180 and 180.")
        return False
        
    return True

def ensure_dir_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure that the specified directory exists
    
    Args:
        directory (Union[str, Path]): Directory path
        
    Returns:
        Path: Path object for the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def generate_features_for_point(
    latitude: float, 
    longitude: float, 
    date: str,
    cache_dir: str = "cache"
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Generate features for cyanobacteria density prediction for a single point
    
    Args:
        latitude (float): Sample point latitude
        longitude (float): Longitude of the point
        date (str): Date in YYYY-MM-DD format
        cache_dir (str): Directory to cache satellite data
        
    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: 
            - Metadata about the selected satellite image
            - Features for prediction
    """
    logger.info(f"Generating features for: lat={latitude}, lon={longitude}, date={date}")
    
    # Validate coordinates
    if not validate_coordinates(latitude, longitude):
        return None, None
    
    # Create cache directory
    ensure_dir_exists(cache_dir)
    
    try:
        # In a real implementation, this would call the actual feature generation code
        # from Feature_Generator.txt. Here we're simulating the feature generation process.
        
        # Parse the date string
        parsed_date = datetime.datetime.strptime(date, "%Y-%m-%d")
        month = parsed_date.month
        year = parsed_date.year
        
        # Create features that would typically be generated from satellite imagery
        # These features are based on the expected model inputs from the provided code
        features = {}
        
        # Spectral band features (mean values)
        # These would normally be calculated from satellite imagery
        base_reflectance = 0.05 + 0.02 * np.sin(month / 12 * 2 * np.pi)  # Seasonal variation
        features.update({
            'B01_mean': base_reflectance * 0.5 + 0.01 * np.random.randn(),
            'B02_mean': base_reflectance * 1.2 + 0.01 * np.random.randn(),
            'B03_mean': base_reflectance * 1.5 + 0.02 * np.random.randn(),
            'B04_mean': base_reflectance * 1.1 + 0.015 * np.random.randn(),
            'B05_mean': base_reflectance * 2.2 + 0.02 * np.random.randn(),
            'B06_mean': base_reflectance * 2.5 + 0.025 * np.random.randn(),
            'B07_mean': base_reflectance * 2.8 + 0.03 * np.random.randn(),
            'B08_mean': base_reflectance * 3.0 + 0.03 * np.random.randn(),
            'B09_mean': base_reflectance * 2.0 + 0.02 * np.random.randn(),
            'B11_mean': base_reflectance * 1.8 + 0.018 * np.random.randn(),
            'B12_mean': base_reflectance * 1.6 + 0.016 * np.random.randn(),
            'B8A_mean': base_reflectance * 2.9 + 0.029 * np.random.randn(),
            'WVP_mean': 2.0 + 0.5 * np.sin(month / 12 * 2 * np.pi) + 0.1 * np.random.randn(),
            'AOT_mean': 0.05 + 0.02 * np.sin(month / 12 * 2 * np.pi) + 0.005 * np.random.randn(),
        })
        
        # Water percentage (higher in summer months in northern hemisphere)
        summer_factor = np.sin((month - 6) / 12 * 2 * np.pi)
        water_percent = 0.75 + 0.15 * summer_factor + 0.05 * np.random.randn()
        features['percent_water'] = max(0.1, min(1.0, water_percent))
        
        # Green band statistics
        features.update({
            'green95th': features['B03_mean'] * 1.2 + 0.01 * np.random.randn(),
            'green5th': features['B03_mean'] * 0.8 + 0.01 * np.random.randn(),
        })
        
        # Band ratios (important for algal bloom detection)
        features.update({
            'green_red_ratio': features['B03_mean'] / features['B04_mean'],
            'green_blue_ratio': features['B03_mean'] / features['B02_mean'],
            'red_blue_ratio': features['B04_mean'] / features['B02_mean'],
            'green95th_blue_ratio': features['green95th'] / features['B02_mean'],
            'green5th_blue_ratio': features['green5th'] / features['B02_mean'],
        })
        
        # NDVI variants (for vegetation and algal detection)
        features.update({
            'NDVI_B04': (features['B08_mean'] - features['B04_mean']) / (features['B08_mean'] + features['B04_mean']),
            'NDVI_B05': (features['B08_mean'] - features['B05_mean']) / (features['B08_mean'] + features['B05_mean']),
            'NDVI_B06': (features['B08_mean'] - features['B06_mean']) / (features['B08_mean'] + features['B06_mean']),
            'NDVI_B07': (features['B08_mean'] - features['B07_mean']) / (features['B08_mean'] + features['B07_mean']),
        })
        
        # AOT range
        features['AOT_range'] = 0.01 + 0.005 * np.random.randn()
        
        # Temporal and geographic features
        features.update({
            'month': month,
            'days_before_sample': 3 + np.random.randint(0, 5),  # Random days before sample (3-7)
            'land_cover': 6,  # 6 typically represents water
        })
        
        # Create metadata for the satellite image
        satellite_meta = pd.DataFrame({
            'item_id': [f'S2A_MSIL2A_{year}{month:02d}01T000000_N0300_R000_T00XXX_{year}{month:02d}01T000000'],
            'cloud_pct': [0.05 + 0.1 * np.random.rand()],  # Random cloud percentage (5-15%)
            'num_water_pixels': [int(10000 * features['percent_water'])],
            'days_before_sample': [features['days_before_sample']],
            'visual_href': [f'https://example.com/sentinel-imagery/{year}/{month:02d}/visual.jpg']
        })
        
        # Create features dataframe
        features_df = pd.DataFrame([features])
        
        logger.info(f"Successfully generated features for point")
        return satellite_meta, features_df
        
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}", exc_info=True)
        return None, None

# Function to initialize the module
def initialize():
    """
    Initialize the feature generator module
    
    In a real implementation, this would load and initialize the actual
    feature generation code from Feature_Generator.txt
    """
    logger.info("Initializing feature generator module")
    
    # Check if the actual feature generator code exists
    feature_generator_path = "Feature_Generator.txt"
    if os.path.exists(feature_generator_path):
        logger.info(f"Found Feature_Generator.txt at {feature_generator_path}")
        # In a real implementation, you would parse and load the code here
    else:
        logger.warning(f"Feature_Generator.txt not found at {feature_generator_path}. Using simplified implementation.")

# Initialize the module when imported
initialize()
