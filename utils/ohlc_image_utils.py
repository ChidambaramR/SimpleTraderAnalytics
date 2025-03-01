import os
import mplfinance as mpf
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot

def create_ohlc_chart(df: pd.DataFrame, output_folder: str, filename: str = None, backward_window: int = 120, forward_window: int = 60) -> str:
    """
    Creates an OHLC chart from the given DataFrame and saves it to the specified folder.
    Uses the 'Agg' backend to avoid GUI-related issues.
    
    Args:
        df (pd.DataFrame): DataFrame containing OHLC data with columns: ['Open', 'High', 'Low', 'Close']
                          and index as DateTimeIndex
        output_folder (str): Path to the folder where the image should be saved
        filename (str, optional): Name of the output file. If None, generates a timestamp-based name
        backward_window (int): Number of data points to show before the center point (default: 120)
        forward_window (int): Number of data points to show after the center point (default: 60)
        
    Returns:
        str: Path to the saved image file
        
    Raises:
        ValueError: If required columns are missing or if the folder path is invalid
    """
    # Validate input DataFrame
    required_columns = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain all OHLC columns: {required_columns}")
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ohlc_chart_{timestamp}.png"
    elif not filename.endswith('.png'):
        filename += '.png'
    
    output_path = os.path.join(output_folder, filename)
    
    # Get the center point (assuming it's a trading day start at 9:15)
    center_idx = None
    if '09:15' in df.index.strftime('%H:%M').values:
        center_idx = df.index.get_indexer(df.index[df.index.strftime('%H:%M') == '09:15'])[0]
    else:
        # If 9:15 not found, use the middle point
        center_idx = len(df) // 2
    
    # Calculate start and end indices for the window
    start_idx = max(0, center_idx - backward_window)
    end_idx = min(len(df), center_idx + forward_window)
    
    # Get the windowed data
    plot_df = df.iloc[start_idx:end_idx]
    
    # Create the OHLC chart with specific style and size
    mpf.plot(
        plot_df,
        type='candle',
        style='charles',
        title='OHLC Chart',
        volume=True if 'Volume' in df.columns else False,
        savefig=dict(
            fname=output_path,
            dpi=300,
            bbox_inches='tight'
        ),
        figsize=(12, 8),
        returnfig=False  # Don't return the figure to avoid memory issues
    )
    
    return output_path 