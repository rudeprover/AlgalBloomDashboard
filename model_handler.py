import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import datetime
from dateutil.relativedelta import relativedelta
import logging
import catboost
import xgboost as xgb

# Import feature generator module
from feature_generator import generate_features_for_point

# Get the logger
logger = logging.getLogger(__name__)

def load_catboost_model() -> catboost.CatBoost:
    """
    Load the pre-trained CatBoost model for density estimation
    
    Returns:
        catboost.CatBoost: Loaded model
    """
    model_path = os.path.join('models', 'cyanobacteria_catboost_model.cbm')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CatBoost model file not found at {model_path}. Please ensure the model file exists.")
    
    try:
        model = catboost.CatBoost()
        model.load_model(model_path)
        logger.info(f"Successfully loaded CatBoost model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load CatBoost model: {str(e)}")
        raise

def load_xgboost_model() -> xgb.Booster:
    """
    Load the pre-trained XGBoost model for time series forecasting
    
    Returns:
        xgb.Booster: Loaded model
    """
    model_path = os.path.join('models', 'cyanobacteria_xgboost_model.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"XGBoost model file not found at {model_path}. Please ensure the model file exists.")
    
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        logger.info(f"Successfully loaded XGBoost model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load XGBoost model: {str(e)}")
        raise

def predict_density_for_date(
    latitude: float, 
    longitude: float, 
    date: datetime.date,
    cache_dir: str = "cache",
) -> Optional[Tuple[float, pd.DataFrame]]:
    """
    Generate prediction for cyanobacteria density at the given coordinates and date
    
    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
        date (datetime.date): Date for prediction
        cache_dir (str): Directory to cache satellite data
        
    Returns:
        Optional[Tuple[float, pd.DataFrame]]: 
            - Predicted density (cells/ml)
            - Features dataframe
    """
    try:
        # Format date for feature generator
        formatted_date = date.strftime("%Y-%m-%d")
        
        # Generate features using the feature generator
        logger.info(f"Generating features for lat={latitude}, lon={longitude}, date={formatted_date}")
        _, features = generate_features_for_point(
            latitude=latitude,
            longitude=longitude,
            date=formatted_date,
            cache_dir=cache_dir
        )
        
        # Check if features were successfully generated
        if features is None or features.empty:
            logger.error(f"Failed to generate features for date {formatted_date}")
            return None
        
        # Load the CatBoost model
        model = load_catboost_model()
        
        # Make prediction
        logger.info(f"Making prediction with CatBoost for date {formatted_date}")
        raw_prediction = model.predict(features)
        
        # Convert prediction to density (cells/ml)
        # Depending on how your model was trained, you might need to transform the raw prediction
        # For example, if you trained on log-transformed values, you need to exponentiate
        density_prediction = np.exp(raw_prediction[0]) if isinstance(raw_prediction, np.ndarray) else np.exp(raw_prediction)
        
        logger.info(f"Generated prediction for {formatted_date}: {density_prediction:.2f} cells/ml")
        
        return density_prediction, features
        
    except Exception as e:
        logger.error(f"Error in prediction process for date {date}: {str(e)}", exc_info=True)
        return None

def generate_historical_monthly_series(
    latitude: float,
    longitude: float,
    start_date: datetime.date,
    end_date: datetime.date,
    cache_dir: str = "cache"
) -> Optional[pd.DataFrame]:
    """
    Generate a monthly time series of cyanobacteria density for the given location
    over the specified date range
    
    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
        start_date (datetime.date): Start date for the time series
        end_date (datetime.date): End date for the time series
        cache_dir (str): Directory to cache satellite data
        
    Returns:
        Optional[pd.DataFrame]: DataFrame with monthly time series data including:
            - date: Date (first day of each month)
            - density: Predicted cyanobacteria density (cells/ml)
    """
    try:
        # Generate a sequence of monthly dates
        current_date = start_date.replace(day=1)  # Start from the first day of the month
        end_date = end_date.replace(day=1)        # End on the first day of the month
        
        dates = []
        while current_date <= end_date:
            dates.append(current_date)
            current_date = (current_date + relativedelta(months=1))
        
        logger.info(f"Generating monthly time series from {start_date} to {end_date} ({len(dates)} points)")
        
        # Create time series dataframe
        time_series = []
        
        # For each date, generate features and predict density
        for date in dates:
            result = predict_density_for_date(
                latitude=latitude,
                longitude=longitude,
                date=date,
                cache_dir=cache_dir
            )
            
            if result is not None:
                density, _ = result
                time_series.append({
                    'date': date,
                    'density': density
                })
            else:
                logger.warning(f"No prediction available for {date}. Using interpolated value.")
                # If prediction fails, use an interpolated value based on the average of previous values
                # This is a simple handling strategy for missing values
                if time_series:
                    avg_density = sum(item['density'] for item in time_series) / len(time_series)
                    time_series.append({
                        'date': date,
                        'density': avg_density
                    })
                else:
                    # If we have no data yet, use a default value
                    time_series.append({
                        'date': date,
                        'density': 10000  # Default value (adjust as needed)
                    })
        
        # Convert to DataFrame
        time_series_df = pd.DataFrame(time_series)
        
        # Add rolling averages and other features that might be useful for the XGBoost model
        time_series_df['rolling_avg_3m'] = time_series_df['density'].rolling(window=3, min_periods=1).mean()
        time_series_df['rolling_avg_6m'] = time_series_df['density'].rolling(window=6, min_periods=1).mean()
        time_series_df['rolling_max_6m'] = time_series_df['density'].rolling(window=6, min_periods=1).max()
        time_series_df['month'] = time_series_df['date'].dt.month
        time_series_df['year'] = time_series_df['date'].dt.year
        
        # Add seasonal decomposition if the time series is long enough
        if len(time_series_df) >= 12:
            # Calculate seasonal component (simple approach)
            monthly_avg = time_series_df.groupby('month')['density'].transform('mean')
            time_series_df['seasonal'] = time_series_df['density'] / monthly_avg
        
        logger.info(f"Successfully generated time series with {len(time_series_df)} points")
        return time_series_df
        
    except Exception as e:
        logger.error(f"Error generating time series: {str(e)}", exc_info=True)
        return None

def prepare_xgboost_features(time_series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for the XGBoost forecast model from the time series
    
    Args:
        time_series_df (pd.DataFrame): Time series DataFrame
        
    Returns:
        pd.DataFrame: Features for XGBoost
    """
    # Get the latest data point (most recent month)
    latest_data = time_series_df.iloc[-1].copy()
    
    # Create features for XGBoost
    features = {}
    
    # Add the most recent density values
    for i in range(1, min(13, len(time_series_df)) + 1):
        idx = -i if i <= len(time_series_df) else 0
        features[f'density_lag_{i}'] = time_series_df.iloc[idx]['density']
    
    # Add rolling averages
    features['rolling_avg_3m'] = latest_data['rolling_avg_3m']
    features['rolling_avg_6m'] = latest_data['rolling_avg_6m']
    features['rolling_max_6m'] = latest_data['rolling_max_6m']
    
    # Add trend (simple linear trend based on last 6 months)
    if len(time_series_df) >= 6:
        last_6m = time_series_df.iloc[-6:]['density'].values
        features['trend'] = (last_6m[-1] - last_6m[0]) / last_6m[0] if last_6m[0] > 0 else 0
    else:
        features['trend'] = 0
    
    # Add seasonality features
    features['month'] = latest_data['month']
    if 'seasonal' in latest_data:
        features['seasonal'] = latest_data['seasonal']
    else:
        features['seasonal'] = 1.0
    
    return pd.DataFrame([features])

def predict_future_density(
    monthly_time_series: pd.DataFrame,
    reference_date: datetime.date,
    days_to_predict: int = 7
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Predict future daily cyanobacteria density based on historical monthly data
    
    Args:
        monthly_time_series (pd.DataFrame): DataFrame with monthly time series
        reference_date (datetime.date): Reference date (end of historical data)
        days_to_predict (int): Number of days to predict forward
        
    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
            - DataFrame with daily predictions
            - Features used for prediction
    """
    try:
        # Load the XGBoost model
        model = load_xgboost_model()
        
        # Prepare features for XGBoost
        features = prepare_xgboost_features(monthly_time_series)
        
        # Generate dates for the forecast
        forecast_dates = [reference_date + datetime.timedelta(days=i) for i in range(1, days_to_predict + 1)]
        
        # Make predictions
        # The XGBoost model is expected to predict the density for each of the next 7 days
        # Convert features to DMatrix format for XGBoost
        dmatrix = xgb.DMatrix(features)
        
        # Get raw predictions
        raw_predictions = model.predict(dmatrix)
        
        # If the model returns a matrix (multiple days at once), use it directly
        # Otherwise, we need to transform the single prediction to a time series
        if isinstance(raw_predictions, np.ndarray) and len(raw_predictions.shape) > 1 and raw_predictions.shape[1] >= days_to_predict:
            # Model directly predicted multiple days
            daily_predictions = raw_predictions[0, :days_to_predict]
        else:
            # Model predicted a single value, we'll use a simple time series transformation
            # to generate daily predictions based on the monthly pattern
            latest_density = monthly_time_series.iloc[-1]['density']
            
            # Apply a simple transformation to create daily predictions
            # This is a simplified approach - in a real model, you'd have daily predictions
            daily_scale_factors = np.linspace(1.0, raw_predictions[0] / latest_density, days_to_predict)
            daily_predictions = latest_density * daily_scale_factors
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'predicted_density': daily_predictions
        })
        
        logger.info(f"Successfully generated {days_to_predict}-day forecast")
        return forecast_df, features
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}", exc_info=True)
        return None, None
