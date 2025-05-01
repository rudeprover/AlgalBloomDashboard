import numpy as np
from typing import Union, List, Tuple
import logging

# Get the logger
logger = logging.getLogger(__name__)

def calculate_mape(
    y_true: Union[List[float], np.ndarray], 
    y_pred: Union[List[float], np.ndarray]
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    
    Args:
        y_true (Union[List[float], np.ndarray]): Actual values
        y_pred (Union[List[float], np.ndarray]): Predicted values
        
    Returns:
        float: MAPE value as a percentage
    """
    try:
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Ensure the arrays are the same length
        if len(y_true) != len(y_pred):
            raise ValueError(f"Arrays must be the same length. Got {len(y_true)} and {len(y_pred)}.")
        
        # Filter out where y_true is zero to avoid division by zero
        non_zero_indices = y_true != 0
        
        if not np.any(non_zero_indices):
            logger.warning("All true values are zero, cannot calculate MAPE")
            return 0.0
        
        # Calculate percentage errors
        percentage_errors = np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])
        
        # Calculate MAPE
        mape = 100.0 * np.mean(percentage_errors)
        
        return mape
    
    except Exception as e:
        logger.error(f"Error calculating MAPE: {str(e)}")
        return 0.0

def calculate_rmse(
    y_true: Union[List[float], np.ndarray], 
    y_pred: Union[List[float], np.ndarray]
) -> float:
    """
    Calculate Root Mean Square Error (RMSE)
    
    Args:
        y_true (Union[List[float], np.ndarray]): Actual values
        y_pred (Union[List[float], np.ndarray]): Predicted values
        
    Returns:
        float: RMSE value
    """
    try:
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Ensure the arrays are the same length
        if len(y_true) != len(y_pred):
            raise ValueError(f"Arrays must be the same length. Got {len(y_true)} and {len(y_pred)}.")
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        return rmse
    
    except Exception as e:
        logger.error(f"Error calculating RMSE: {str(e)}")
        return 0.0

def calculate_mae(
    y_true: Union[List[float], np.ndarray], 
    y_pred: Union[List[float], np.ndarray]
) -> float:
    """
    Calculate Mean Absolute Error (MAE)
    
    Args:
        y_true (Union[List[float], np.ndarray]): Actual values
        y_pred (Union[List[float], np.ndarray]): Predicted values
        
    Returns:
        float: MAE value
    """
    try:
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Ensure the arrays are the same length
        if len(y_true) != len(y_pred):
            raise ValueError(f"Arrays must be the same length. Got {len(y_true)} and {len(y_pred)}.")
        
        # Calculate MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        return mae
    
    except Exception as e:
        logger.error(f"Error calculating MAE: {str(e)}")
        return 0.0

def calculate_all_metrics(
    y_true: Union[List[float], np.ndarray], 
    y_pred: Union[List[float], np.ndarray]
) -> Tuple[float, float, float]:
    """
    Calculate all metrics: MAPE, RMSE, and MAE
    
    Args:
        y_true (Union[List[float], np.ndarray]): Actual values
        y_pred (Union[List[float], np.ndarray]): Predicted values
        
    Returns:
        Tuple[float, float, float]: MAPE, RMSE, and MAE values
    """
    mape = calculate_mape(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    
    return mape, rmse, mae
